/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/ProcessArgs.h"
#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/Utils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"

static constexpr const char *DEBUG_TYPE = "ProcessArgs";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(...) \
LLVM_DEBUG({ \
  DBGS(); \
  llvm::dbgs() << __VA_ARGS__; \
  llvm::dbgs() << "\n"; \
})

using namespace llvm;
using namespace mlir;
using namespace triton;

// For each shared iter_arg, we need to track:
// - Which block_ids use it
// - Who is the owner (first block_id in order)
// - For each non-owner block, what new iter_arg index to use
struct SharedArgInfo {
  int argIndex;           // original iter_arg index (0, 1, 2...)
  Value iterArg;          // the original iter_arg value
  int ownerBlockId;       // block_id that "owns" this arg (first in order)
  int newArgIndex;        // new iter_arg index in the new for op (for this specific block)
  int nonOwnerBlockId;    // the non-owner block that needs a clone
};

// Find iter_args shared across multiple block_ids
static LogicalResult findSharedIterArgs(
    scf::ForOp forOp,
    SmallVector<int> &idsInOrder,
    SmallVector<SharedArgInfo> &sharedArgsInfo,
    llvm::DenseMap<int, int> &oldArgIndexToNewArgIndexBase)
{
  Block *body = forOp.getBody();
  if (!body || !body->mightHaveTerminator()) {
    LDBG("[Error]: forOp body is invalid or has no terminator\n");
    return failure();
  }

  // Find which block_ids use which iter_args
  llvm::DenseMap<int, llvm::DenseSet<int>> argIndexToBlockIds;
  for (Operation &op : body->without_terminator()) {
    auto blockIdAttr = op.getAttrOfType<IntegerAttr>("ssbuffer.block_id");
    if (!blockIdAttr) continue;
    int blockId = blockIdAttr.getInt();

    for (OpOperand &operand : op.getOpOperands()) {
      Value v = operand.get();
      for (unsigned i = 0; i < forOp.getNumRegionIterArgs(); ++i) {
        if (v == forOp.getRegionIterArgs()[i]) {
          argIndexToBlockIds[i].insert(blockId);
        }
      }
    }
  }

  // Find iter_args used by multiple block_ids
  int extraArgCount = 0;
  for (auto &p : argIndexToBlockIds) {
    int argIndex = p.first;
    const llvm::DenseSet<int> &blockIds = p.second;

    if (blockIds.size() > 1) {
      int ownerBlockId = -1;
      for (int id : idsInOrder) {
        if (blockIds.contains(id)) {
          ownerBlockId = id;
          break;
        }
      }
      if (ownerBlockId == -1) continue;

      oldArgIndexToNewArgIndexBase[argIndex] = forOp.getNumRegionIterArgs() + extraArgCount;
      extraArgCount++;

      for (int bid : blockIds) {
        if (bid != ownerBlockId) {
          SharedArgInfo info;
          info.argIndex = argIndex;
          info.iterArg = forOp.getRegionIterArgs()[argIndex];
          info.ownerBlockId = ownerBlockId;
          info.newArgIndex = oldArgIndexToNewArgIndexBase[argIndex];
          info.nonOwnerBlockId = bid;
          sharedArgsInfo.push_back(info);
        }
      }
    }
  }

  return success();
}

// Find the computation chain for each shared arg
static LogicalResult findComputationChains(
    scf::ForOp forOp,
    SmallVector<SharedArgInfo> &sharedArgsInfo,
    llvm::DenseMap<int, Operation*> &sharedArgToCompOp,
    llvm::DenseMap<int, llvm::DenseSet<Operation*>> &sharedArgToChainOps)
{
  Block *body = forOp.getBody();

  for (auto &info : sharedArgsInfo) {
    int argIndex = info.argIndex;
    if (sharedArgToCompOp.contains(argIndex)) continue;

    Value iterArg = forOp.getRegionIterArgs()[argIndex];

    // Find the operation in owner block that produces yield input
    Operation *compOp = nullptr;
    for (Operation &op : body->without_terminator()) {
      auto blockIdAttr = op.getAttrOfType<IntegerAttr>("ssbuffer.block_id");
      if (!blockIdAttr || blockIdAttr.getInt() != info.ownerBlockId) continue;

      bool usesIterArg = false;
      for (Value operand : op.getOperands()) {
        if (operand == iterArg) {
          usesIterArg = true;
          break;
        }
      }
      if (!usesIterArg) continue;

      for (Value result : op.getResults()) {
        for (OpOperand &use : result.getUses()) {
          if (isa<scf::YieldOp>(use.getOwner())) {
            compOp = &op;
            break;
          }
        }
        if (compOp) break;
      }
      if (compOp) break;
    }

    if (!compOp) {
      LDBG("[Error]: Could not find comp op for arg " << iterArg << "\n");
      continue;
    }

    sharedArgToCompOp[argIndex] = compOp;

    // Collect the computation chain (backward traversal)
    llvm::DenseSet<Operation*> chainOps;
    SmallVector<Operation*> worklist;
    worklist.push_back(compOp);

    while (!worklist.empty()) {
      Operation *op = worklist.pop_back_val();
      if (chainOps.contains(op)) continue;
      chainOps.insert(op);

      for (Value operand : op->getOperands()) {
        if (auto *defOp = operand.getDefiningOp()) {
          if (defOp->getParentOp() == forOp && !chainOps.contains(defOp)) {
            worklist.push_back(defOp);
          }
        }
      }
    }

    sharedArgToChainOps[argIndex] = chainOps;
  }

  return success();
}

// Find the last operation in the for body with the given block_id
static Operation *findLastOpInBlock(Block *body, int blockId)
{
  Operation *lastOp = nullptr;
  for (Operation &op : body->without_terminator()) {
    auto blockIdAttr = op.getAttrOfType<IntegerAttr>("ssbuffer.block_id");
    if (blockIdAttr && blockIdAttr.getInt() == blockId) {
      lastOp = &op;
    }
  }
  return lastOp;
}

// Create new for op with extra iter_args
static scf::ForOp createNewForOp(
    scf::ForOp forOp,
    llvm::DenseMap<int, int> &oldArgIndexToNewArgIndexBase)
{
  OpBuilder builder(forOp);
  SmallVector<Value> newInitArgs(forOp.getInitArgs().begin(), forOp.getInitArgs().end());

  for (auto &p : oldArgIndexToNewArgIndexBase) {
    int oldArgIndex = p.first;
    newInitArgs.push_back(forOp.getInitArgs()[oldArgIndex]);
  }

  scf::ForOp newForOp = builder.create<scf::ForOp>(
      forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
      forOp.getStep(), newInitArgs);

  for (auto &attr : forOp->getAttrs()) {
    newForOp->setAttr(attr.getName(), attr.getValue());
  }

  return newForOp;
}

// Migrate body from old block to new block
static void migrateBody(Block *oldBlock, Block *newBlock)
{
  // Save old block arguments
  SmallVector<Value> oldBlockArgs;
  for (unsigned i = 0; i < oldBlock->getNumArguments(); ++i) {
    oldBlockArgs.push_back(oldBlock->getArgument(i));
  }

  // Redirect block arguments
  for (unsigned i = 0; i < oldBlock->getNumArguments(); ++i) {
    oldBlock->getArgument(i).replaceAllUsesWith(newBlock->getArgument(i));
  }

  // Move all operations
  for (Operation &op : llvm::make_early_inc_range(oldBlock->without_terminator())) {
    op.moveBefore(newBlock, newBlock->end());
  }
}

// Clone computation chain for a non-owner block
static Value cloneChainForBlock(
    Block *newBlock,
    Operation *lastOpInBlock,
    llvm::DenseSet<Operation*> &chainOps,
    Operation *compOp,
    Value oldBlockArg,
    Value newBlockArg,
    int nonOwnerBlockId,
    int argIndex,
    OpBuilder &builder)
{
  SmallVector<Operation *> sortedChain(chainOps.begin(), chainOps.end());
  if (failed(topologicalSort(sortedChain))) return nullptr;

  if (lastOpInBlock) {
    builder.setInsertionPointAfter(lastOpInBlock);
  }

  IRMapping argMapper;
  argMapper.map(oldBlockArg, newBlockArg);

  IRMapping resultMapper;
  for (Operation *op : sortedChain) {
    IRMapping opMapper;
    for (OpOperand &operand : op->getOpOperands()) {
      Value oldVal = operand.get();
      Value newVal = argMapper.contains(oldVal) ? argMapper.lookup(oldVal) : oldVal;
      opMapper.map(oldVal, newVal);
    }

    if (resultMapper.contains(op->getResult(0))) continue;

    Operation *cloned = builder.clone(*op, opMapper);
    cloned->setAttr("ssbuffer.block_id", builder.getI32IntegerAttr(nonOwnerBlockId));
    cloned->setAttr("ssbuffer.arg", builder.getI32IntegerAttr(argIndex));

    resultMapper.map(op->getResult(0), cloned->getResult(0));
    builder.setInsertionPointAfter(cloned);
  }

  return resultMapper.lookup(compOp->getResult(0));
}

// Replace uses of old iter_arg with clone in non-owner block
static void replaceUsesInBlock(
    Block *block,
    int nonOwnerBlockId,
    Value originalArg,
    Value newBlockArg,
    OpBuilder &builder,
    int argIndex)
{
  for (Operation &op : block->without_terminator()) {
    auto blockIdAttr = op.getAttrOfType<IntegerAttr>("ssbuffer.block_id");
    if (!blockIdAttr || blockIdAttr.getInt() != nonOwnerBlockId) continue;

    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      if (op.getOperand(i) == originalArg) {
        op.setOperand(i, newBlockArg);
        op.setAttr("ssbuffer.arg", builder.getI32IntegerAttr(argIndex));
      }
    }
  }
}

// Build new yield with cloned results
static void buildNewYield(
    Block *oldBlock,
    Block *newBlock,
    SmallVector<Value> &clonedResults,
    OpBuilder &builder)
{
  auto oldYield = cast<scf::YieldOp>(oldBlock->getTerminator());
  SmallVector<Value> yieldOperands;
  for (unsigned i = 0; i < oldYield.getNumOperands(); ++i) {
    yieldOperands.push_back(oldYield.getOperand(i));
  }
  for (auto &result : clonedResults) {
    yieldOperands.push_back(result);
  }

  builder.setInsertionPointToEnd(newBlock);
  builder.create<scf::YieldOp>(newBlock->getParentOp()->getLoc(), yieldOperands);
  oldYield.erase();
}

// Finalize: replace uses and erase old for op
static void finalizeForOp(scf::ForOp forOp, scf::ForOp newForOp)
{
  if (forOp.getNumResults() > 0) {
    SmallVector<Value> newResults;
    for (unsigned i = 0; i < forOp.getNumResults(); ++i) {
      newResults.push_back(newForOp.getResult(i));
    }
    forOp.replaceAllUsesWith(newResults);
  }
  forOp.erase();
}

// Process one shared arg: clone chain and replace uses
static Value processOneSharedArg(
    Block *newBlock,
    Block *oldBlock,
    SharedArgInfo &info,
    llvm::DenseMap<int, Operation*> &sharedArgToCompOp,
    llvm::DenseMap<int, llvm::DenseSet<Operation*>> &sharedArgToChainOps)
{
  Operation *compOp = sharedArgToCompOp[info.argIndex];
  if (!compOp) return nullptr;

  llvm::DenseSet<Operation*> &chainOps = sharedArgToChainOps[info.argIndex];
  if (chainOps.empty()) return nullptr;

  Operation *lastOpInBlock = findLastOpInBlock(newBlock, info.nonOwnerBlockId);

  Value oldBlockArg = oldBlock->getArgument(info.argIndex + 1);
  Value newBlockArg = newBlock->getArgument(info.newArgIndex + 1);

  OpBuilder cloneBuilder(newBlock, newBlock->end());
  Value clonedResult = cloneChainForBlock(
      newBlock, lastOpInBlock, chainOps, compOp,
      oldBlockArg, newBlockArg,
      info.nonOwnerBlockId, info.argIndex, cloneBuilder);

  if (clonedResult) {
    Value originalArg = newBlock->getArgument(info.argIndex + 1);
    replaceUsesInBlock(newBlock, info.nonOwnerBlockId, originalArg, newBlockArg, cloneBuilder, info.argIndex);
  }

  return clonedResult;
}

static LogicalResult processSharedIterArgsInForOp(scf::ForOp forOp)
{
  SmallVector<int> idsInOrder = getBlockIdsInOrder(forOp);

  // Step 1: Find shared iter_args
  SmallVector<SharedArgInfo> sharedArgsInfo;
  llvm::DenseMap<int, int> oldArgIndexToNewArgIndexBase;

  if (failed(findSharedIterArgs(forOp, idsInOrder, sharedArgsInfo, oldArgIndexToNewArgIndexBase))) {
    return failure();
  }
  if (sharedArgsInfo.empty()) return success();

  LDBG("Found " << sharedArgsInfo.size() << " shared iter_args to process\n");

  // Step 2: Find computation chains
  llvm::DenseMap<int, Operation*> sharedArgToCompOp;
  llvm::DenseMap<int, llvm::DenseSet<Operation*>> sharedArgToChainOps;

  if (failed(findComputationChains(forOp, sharedArgsInfo, sharedArgToCompOp, sharedArgToChainOps))) {
    return failure();
  }

  // Step 3: Create new for op
  scf::ForOp newForOp = createNewForOp(forOp, oldArgIndexToNewArgIndexBase);
  Block *oldBlock = forOp.getBody();
  Block *newBlock = newForOp.getBody();

  // Step 4: Migrate body
  migrateBody(oldBlock, newBlock);

  // Step 5: Clone chain and replace uses for each non-owner block
  OpBuilder builder(forOp);
  SmallVector<Value> clonedResults;

  for (auto &info : sharedArgsInfo) {
    Value clonedResult = processOneSharedArg(
        newBlock, oldBlock, info, sharedArgToCompOp, sharedArgToChainOps);
    if (clonedResult) clonedResults.push_back(clonedResult);
  }

  // Step 6: Build new yield
  buildNewYield(oldBlock, newBlock, clonedResults, builder);

  // Step 7: Replace uses and erase old for op
  finalizeForOp(forOp, newForOp);

  return success();
}

LogicalResult ProcessArgsPass::processSharedIterArgs(ModuleOp module)
{
  WalkResult result = module.walk([&](Operation *op) -> WalkResult {
    if (!op->hasAttr("ssbuffer.main_loop")) {
      return WalkResult::advance();
    }
    auto forOp = dyn_cast<scf::ForOp>(op);
    if (!forOp) {
      LDBG("[Error]: op with ssbuffer.main_loop is not a scf::ForOp\n");
      return WalkResult::interrupt();
    }

    if (failed(processSharedIterArgsInForOp(forOp))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (result.wasInterrupted()) {
    return failure();
  }
  return success();
}

void ProcessArgsPass::runOnOperation()
{
  ModuleOp module = getOperation();

  LDBG("before processArgs:\n" << module << "\n");

  if (failed(processSharedIterArgs(module))) {
    signalPassFailure();
    return;
  }

  LDBG("after processArgs:\n" << module << "\n");
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createProcessArgsPass()
{
  return std::make_unique<ProcessArgsPass>();
}

} // namespace triton
} // namespace mlir