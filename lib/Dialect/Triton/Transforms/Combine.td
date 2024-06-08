#ifndef TRITON_PATTERNS
#define TRITON_PATTERNS

include "mlir/Dialect/Arith/IR/ArithOps.td"
include "triton/Dialect/Triton/IR/TritonOps.td"
include "mlir/IR/PatternBase.td"


// AddIOp(DotOp(a, b, c), d) and c==0 => DotOp(a, b, d)
// AddFOp(DotOp(a, b, c), d) and c==0 => DotOp(a, b, d)

// AddIOp(d, DotOp(a, b, c)) and c==0 => DotOp(a, b, d)
// AddFOp(d, DotOp(a, b, c)) and c==0 => DotOp(a, b, d)
def CombineDotAddIPattern : Pat<
        (Arith_AddIOp $d, (TT_DotOp:$res $a, $b, $c, $inputPrecision, $maxNumImpreciseAcc), $overflow),
        (TT_DotOp $a, $b, $d, $inputPrecision, $maxNumImpreciseAcc, (location $res)),
        [(Constraint<CPred<"isZero($0)">> $c),
         (Constraint<CPred<"res->hasOneUse()">, "dot result has a single use">)]>;
def CombineDotAddFPattern : Pat<
        (Arith_AddFOp $d, (TT_DotOp:$res $a, $b, $c, $inputPrecision, $maxNumImpreciseAcc), $fastmath),
        (TT_DotOp $a, $b, $d, $inputPrecision, $maxNumImpreciseAcc, (location $res)),
        [(Constraint<CPred<"isZero($0)">> $c),
         (Constraint<CPred<"::llvm::cast<::mlir::IntegerAttr>($0).getInt() == 0">> $maxNumImpreciseAcc),
         (Constraint<CPred<"res->hasOneUse()">, "dot result has a single use">)]>;

def CombineDotAddIRevPattern : Pat<
        (Arith_AddIOp (TT_DotOp:$res $a, $b, $c, $inputPrecision, $maxNumImpreciseAcc), $d, $overflow),
        (TT_DotOp $a, $b, $d, $inputPrecision, $maxNumImpreciseAcc, (location $res)),
        [(Constraint<CPred<"isZero($0)">> $c),
         (Constraint<CPred<"res->hasOneUse()">, "dot result has a single use">)]>;
def CombineDotAddFRevPattern : Pat<
        (Arith_AddFOp (TT_DotOp:$res $a, $b, $c, $inputPrecision, $maxNumImpreciseAcc), $d, $fastmath),
        (TT_DotOp $a, $b, $d, $inputPrecision, $maxNumImpreciseAcc, (location $res)),
        [(Constraint<CPred<"isZero($0)">> $c),
         (Constraint<CPred<"::llvm::cast<::mlir::IntegerAttr>($0).getInt() == 0">> $maxNumImpreciseAcc),
         (Constraint<CPred<"res->hasOneUse()">, "dot result has a single use">)]>;

// addptr(addptr(%ptr, %idx0), %idx1) => addptr(%ptr, AddI(%idx0, %idx1))
//   Note: leave (sub %c0, %c0) canceling to ArithDialect
//         (ref: ArithCanonicalization.td)
defvar DefOverflow = ConstantEnumCase<Arith_IntegerOverflowAttr, "none">;
def CombineAddPtrPattern : Pat<
        (TT_AddPtrOp (TT_AddPtrOp $ptr, $idx0), $idx1),
        (TT_AddPtrOp $ptr, (Arith_AddIOp $idx0, $idx1, DefOverflow)),
        [(Constraint<CPred<"isAddPtrOffsetCombinable($0, $1)">> $idx0, $idx1)]>;

// select(cond, load(ptrs, splat(cond), ???), other)
//   => load(ptrs, splat(cond), other)
def HasOneUse: Constraint<CPred<"$_self.hasOneUse()">, "has one use">;
def CombineSelectMaskedLoadSplatPattern: Pat<
    (TT_SelectOp
        $cond,
        (TT_LoadOp:$load $ptrs, (TT_SplatOp:$splat_cond $cond), $other_load,
            $boundaryCheck, $padding, $cache, $evict, $isVolatile),
        $other_select),
    (TT_LoadOp $ptrs, $splat_cond, $other_select,
        $boundaryCheck, $padding, $cache, $evict, $isVolatile),
    [HasOneUse $load]>;

// select(cond, load(ptrs, cond, ???), other)
//   => load(ptrs, cond, other)
def CombineSelectMaskedLoadPattern: Pat<
    (TT_SelectOp
        $cond,
        (TT_LoadOp:$load $ptrs, $cond, $other_load,
            $boundaryCheck, $padding, $cache, $evict, $isVolatile),
        $other_select),
    (TT_LoadOp $ptrs, $cond, $other_select,
        $boundaryCheck, $padding, $cache, $evict, $isVolatile),
    [HasOneUse $load]>;

#endif
