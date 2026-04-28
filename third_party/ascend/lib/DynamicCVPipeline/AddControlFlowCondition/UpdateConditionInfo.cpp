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
#include "third_party/ascend/include/DynamicCVPipeline/AddControlFlowCondition/UpdateConditionInfo.h"
#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/IR/HIVMInterfaces.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/Debug.h"

static constexpr const char *DEBUG_TYPE = "UpdateConditionInfoPass";
static constexpr const char *SSBUFFER_Main_LOOP = "ssbuffer.main_loop";
static constexpr const char *SSBUFFER_IF = "ssbuffer.if";

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << (X) << "\n")
using namespace mlir;
using namespace triton;
using namespace hivm;

void UpdateConditionInfoPass::runOnOperation()
{
    ModuleOp module = getOperation();

    LDBG("Enter UpdateConditionInfo pass.");
    // Update the conditions of ifOp based on the intraCoreDependentMap and crossCoreDependentMap
    LDBG("Exit UpdateConditionInfo pass.");
}

namespace mlir {
namespace triton {
std::unique_ptr<OperationPass<ModuleOp> > createUpdateConditionInfoPass()
{
    return std::make_unique<UpdateConditionInfoPass>();
}
} // namespace triton
} // namespace mlir
