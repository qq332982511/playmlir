func @matmul(%arg0: tensor<128x64xf32>,
                          %arg1: tensor<64x256xf32>, %arg2 : tensor<128x256xf32>) ->tensor<128x256xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<128x64xf32>, tensor<64x256xf32>) outs(%arg2 : tensor<128x256xf32>) -> tensor<128x256xf32>
  return %0 : tensor<128x256xf32>
}

func @print_perf(%iters: index, %total_time: f64) {
  %c2 = arith.constant 2 : index
  %cM = arith.constant 256 : index
  %cN = arith.constant 256 : index
  %cK = arith.constant 256 : index

  %mn = arith.muli %cM, %cN : index
  %mnk = arith.muli %mn, %cK : index

  // 2*M*N*K.
  %flops_per_iter = arith.muli %c2, %mnk : index
  %flops = arith.muli %iters, %flops_per_iter : index
  %flops_i64 = arith.index_cast %flops : index to i64
  %flops_f = arith.sitofp %flops_i64 : i64 to f64
  %flops_per_s = arith.divf %flops_f, %total_time : f64
  vector.print %flops_per_s : f64

  return
}

func @main() {
  %ca = linalg.init_tensor [128, 64] : tensor<128x64xf32>
  %cb = linalg.init_tensor [64, 256] : tensor<64x256xf32>
  %cc = linalg.init_tensor [128, 256] : tensor<128x256xf32>
    
  %f1 = arith.constant 1.0 : f32
  %f0 = arith.constant 0.0 : f32

  %fill_ca = linalg.fill(%f1, %ca) : f32, tensor<128x64xf32> -> tensor<128x64xf32>
  %fill_cb = linalg.fill(%f1, %cb) : f32, tensor<64x256xf32> -> tensor<64x256xf32>
  %fill_cc = linalg.fill(%f0, %cc) : f32, tensor<128x256xf32> -> tensor<128x256xf32>

  %c0 = arith.constant 0: index
  %c1 = arith.constant 1: index
  %iters = arith.constant 200: index

  %t_start_matmul = call @rtclock() : () -> f64
  scf.for %arg0 = %c0 to %iters step %c1 {
    call @matmul(%fill_ca, %fill_cb, %fill_cc) : (tensor<128x64xf32>, tensor<64x256xf32>, tensor<128x256xf32>) -> tensor<128x256xf32>
  }
  %t_end_matmul = call @rtclock() : () -> f64
  %tmatmul = arith.subf %t_end_matmul, %t_start_matmul: f64
  call @print_perf(%iters, %tmatmul) : (index, f64) -> ()

  %f64 = arith.constant 64.0 : f32
  %ct = linalg.init_tensor [128, 256] : tensor<128x256xf32>
  %fill_ct = linalg.fill(%f64, %ct) : f32, tensor<128x256xf32> -> tensor<128x256xf32>
  %fct = bufferization.to_memref %fill_ct : memref<128x256xf32>
  %ufct = memref.cast %fct :  memref<128x256xf32> to memref<*xf32>

  %cc_res = call @matmul(%fill_ca, %fill_cb, %fill_cc) : (tensor<128x64xf32>, tensor<64x256xf32>, tensor<128x256xf32>) -> tensor<128x256xf32>
  %res = bufferization.to_memref %cc_res : memref<128x256xf32>
  %uc = memref.cast %res :  memref<128x256xf32> to memref<*xf32>
  %errors = call @verifyMemRefF32(%ufct, %uc) : (memref<*xf32>, memref<*xf32>) -> i64
  vector.print %errors : i64

  return
}

func private @rtclock() -> f64
func private @verifyMemRefF32(memref<*xf32>, memref<*xf32>) -> i64 attributes { llvm.emit_c_interface }
func private @print_memref_f32(memref<*xf32>)
