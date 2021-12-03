!elem_type_a = type f32
!elem_type_b = type f32
!elem_type_c = type f32
!row_major_A = type memref<256x256x!elem_type_a>
!row_major_B = type memref<256x256x!elem_type_b>
!row_major_C = type memref<256x256x!elem_type_c>

func @matmul(%a: !row_major_A, %b: !row_major_B, %c: !row_major_C)
// TODO: activate manually for now.
// attributes { passthrough = [["target-cpu", "skylake-avx512"], ["prefer-vector-width", "512"]]}
{
  linalg.matmul ins(%a, %b : !row_major_A, !row_major_B)
    outs(%c: !row_major_C)
  return
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
  %v0 = arith.constant 0.0 : !elem_type_a
  %v1 = arith.constant 1.0 : !elem_type_a

  %A = memref.alloc() : !row_major_A
  %B = memref.alloc() : !row_major_B
  %C = memref.alloc() : !row_major_C

  linalg.fill(%v1, %A) : !elem_type_a, !row_major_A
  linalg.fill(%v1, %B) : !elem_type_b, !row_major_B
  linalg.fill(%v0, %C) : !elem_type_c, !row_major_C

  %c0 = arith.constant 0: index
  %c1 = arith.constant 1: index
  %iters = arith.constant 100: index

  /// Run and dump performance for matmul.
  /// Preheating run:
  scf.for %arg0 = %c0 to %iters step %c1 {
    %z = arith.constant 0.0 : !elem_type_c
    linalg.fill(%z, %C) : !elem_type_c, !row_major_C
    call @matmul(%A, %B, %C) : (!row_major_A, !row_major_B, !row_major_C) -> ()
  }
  %t_start_matmul = call @rtclock() : () -> f64
  scf.for %arg0 = %c0 to %iters step %c1 {
    // linalg.matmul writes %C in place, need to reset it to zero every time.
    // This is accounts for about 10-15% perf hit on small sizes.
    // Once linalg on tensors is ready, fusing fill at the register level will
    // be easy.
    %z = arith.constant 0.0 : !elem_type_c
    linalg.fill(%z, %C) : !elem_type_c, !row_major_C
    call @matmul(%A, %B, %C) : (!row_major_A, !row_major_B, !row_major_C) -> ()
  }
  %t_end_matmul = call @rtclock() : () -> f64
  %tmatmul = arith.subf %t_end_matmul, %t_start_matmul: f64
  call @print_perf(%iters, %tmatmul) : (index, f64) -> ()

  // CHECK: {{^0$}}
  %C_ref = memref.alloc() : !row_major_C
  linalg.fill(%v0, %C_ref) : !elem_type_c, !row_major_C
  linalg.matmul ins(%A, %B : !row_major_A, !row_major_B)
    outs(%C_ref: !row_major_C)
  %act = memref.cast %C : !row_major_C to memref<*xf32>
  %exp = memref.cast %C_ref : !row_major_C to memref<*xf32>
  %errors = call @verifyMemRefF32(%act, %exp) : (memref<*xf32>, memref<*xf32>) -> i64
  vector.print %errors : i64
  memref.dealloc %C_ref : !row_major_C

  memref.dealloc %A : !row_major_A
  memref.dealloc %B : !row_major_B
  memref.dealloc %C : !row_major_C

  return
}

func private @rtclock() -> f64
func private @verifyMemRefF32(memref<*xf32>, memref<*xf32>) -> i64 attributes { llvm.emit_c_interface }
