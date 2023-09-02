#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include <memory>

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

using namespace mlir;

class MakeScanDimContiguous : public mlir::OpRewritePattern<triton::ScanOp> {
public:
  MakeScanDimContiguous(mlir::MLIRContext *context)
      : OpRewritePattern<triton::ScanOp>(context, 1) {}

  mlir::LogicalResult
  matchAndRewrite(triton::ScanOp scanOp,
                  mlir::PatternRewriter &rewriter) const override {
    using mlir::Value;
    using triton::gpu::BlockedEncodingAttr;

    auto axis = scanOp.getAxis();
    auto exampleType = llvm::dyn_cast<RankedTensorType>(scanOp.getType(0));
    auto srcEncoding = exampleType.getEncoding().cast<BlockedEncodingAttr>();
    auto shape = exampleType.getShape();

    auto threadsPerWarp = srcEncoding.getThreadsPerWarp();
    auto warpsPerCTA = srcEncoding.getWarpsPerCTA();

    auto desiredElemsPerThread =
        shape[axis] / (threadsPerWarp[axis] * warpsPerCTA[axis]);
    llvm::errs() << "Here: " << threadsPerWarp[axis] << " " << warpsPerCTA[axis]
                 << " " << desiredElemsPerThread << "\n";

    auto srcElemsPerThread = srcEncoding.getSizePerThread();
    if (srcElemsPerThread[axis] >= desiredElemsPerThread) {
      return mlir::failure();
    }

    SmallVector<unsigned> newSizePerThread(shape.size(), 1);
    newSizePerThread[axis] = desiredElemsPerThread;

    auto ctaLayout = triton::gpu::getCTALayout(srcEncoding);
    auto newEncoding = BlockedEncodingAttr::get(
        getContext(), newSizePerThread, threadsPerWarp, warpsPerCTA,
        srcEncoding.getOrder(), ctaLayout);

    auto loc = scanOp.getLoc();
    SmallVector<Value> newOperands;
    for (auto op : scanOp.getOperands()) {
      auto elemTy = op.getType().cast<RankedTensorType>().getElementType();
      auto newOp = rewriter.create<triton::gpu::ConvertLayoutOp>(
          loc, RankedTensorType::get(shape, elemTy, newEncoding), op);

      newOperands.push_back(newOp);
    }

    auto newScan = rewriter.create<triton::ScanOp>(loc, newOperands, axis);
    auto &newCombineOp = newScan.getCombineOp();
    rewriter.cloneRegionBefore(scanOp.getCombineOp(), newCombineOp,
                               newCombineOp.end());

    for (unsigned i = 0; i < newOperands.size(); ++i) {
      auto newRes = rewriter.create<triton::gpu::ConvertLayoutOp>(
          loc, scanOp.getType(i), newScan.getResult()[i]);
      scanOp.getResult()[i].replaceAllUsesWith(newRes);
    }
    return mlir::success();
  }
};

class TritonGPUOptimizeScanOperandsPass
    : public TritonGPUOptimizeScanOperandsBase<
          TritonGPUOptimizeScanOperandsPass> {
public:
  TritonGPUOptimizeScanOperandsPass() = default;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    mlir::RewritePatternSet patterns(context);
    patterns.add<MakeScanDimContiguous>(context);

    ModuleOp m = getOperation();
    if (applyPatternsAndFoldGreedily(m, std::move(patterns)).failed())
      signalPassFailure();
  }
};

std::unique_ptr<Pass> mlir::createTritonGPUOptimizeScanOperandsPass() {
  return std::make_unique<TritonGPUOptimizeScanOperandsPass>();
}
