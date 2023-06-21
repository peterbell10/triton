// RUN: triton-opt %s -split-input-file -triton-reorder-broadcast | FileCheck %s

// CHECK-LABEL: @test_splat_elementwise_pattern
tt.func @test_splat_elementwise_pattern(%arg0: f32) -> (tensor<128x128xf32>, tensor<128x128x!tt.ptr<f32>>) {
    // CHECK-DAG: %[[a:.*]] = arith.constant 1.000000e+00 : f32
    // CHECK-DAG: %[[c1:.*]] = arith.constant 1 : i64
    %c1 = arith.constant 1 : i64
    %a = arith.constant dense<1.0> : tensor<128x128xf32>

    // CHECK-DAG: %[[add:.*]] = arith.addf %arg0, %[[a]] : f32
    // CHECK-NEXT: %[[splat:.*]] = tt.splat %[[add]] : (f32) -> tensor<128x128xf32>
    %b = tt.splat %arg0 : (f32) -> tensor<128x128xf32>
    %add = arith.addf %a, %b : tensor<128x128xf32>


    // CHECK-NEXT: %[[ptr:.*]] = tt.int_to_ptr %[[c1]] : i64 -> !tt.ptr<f32>
    // CHECK-NEXT: %{{.*}} = tt.splat %[[ptr]] : (!tt.ptr<f32>) -> tensor<128x128x!tt.ptr<f32>>
    %c1_t = tt.splat %c1 : (i64) -> tensor<128x128xi64>
    %ptr = tt.int_to_ptr %c1_t : tensor<128x128xi64> -> tensor<128x128x!tt.ptr<f32>>

    tt.return %add, %ptr : tensor<128x128xf32>, tensor<128x128x!tt.ptr<f32>>
}

// CHECK-LABEL: @test_broadcast_elementwise_pattern
tt.func @test_broadcast_elementwise_pattern(%arg0: tensor<128x1xf32>) -> tensor<128x128xf32> {

    // CHECK-NEXT: %[[abs:.*]] = math.absf %arg0 : tensor<128x1xf32>
    // CHECK-NEXT: %{{.*}} = tt.broadcast %[[abs]] : (tensor<128x1xf32>) -> tensor<128x128xf32>
    %broadcast = tt.broadcast %arg0 : (tensor<128x1xf32>) -> tensor<128x128xf32>
    %abs = math.absf %broadcast : tensor<128x128xf32>

    tt.return %abs : tensor<128x128xf32>
}
