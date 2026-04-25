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

#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition.h"
#include "llvm/Support/Debug.h"

static constexpr const char *DEBUG_TYPE = "AddControlFlowCondition";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << (X) << "\n")

using namespace mlir;
using namespace triton;

void AddControlFlowConditionPass::runOnOperation()
{
    ModuleOp module = getOperation();

    LDBG("Enter add controlflow condition pass.");
    // Step1:Fill in the intraCoreDependentMap and crossCoreDependentMap

    // Step2:Create an ifOp wrapper block based on the block_id

    // Step3:Fill in blockCounters innerDepConds and insertInterCorePipeS

    // Step4:Update the conditions of ifOp based on the intraCoreDependentMap and crossCoreDependentMap

    // Step5:Update the iteration count of forOp
    LDBG("Exit add controlflow condition pass.");
}

namespace mlir {
namespace triton {
std::unique_ptr<OperationPass<ModuleOp>> createAddControlFlowConditionPass()
{
    return std::make_unique<AddControlFlowConditionPass>();
}
} // namespace triton
} // namespace mlir