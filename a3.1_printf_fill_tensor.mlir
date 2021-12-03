func @main() {
    %c = linalg.init_tensor [16] : tensor<16xf32>
    %f1 = arith.constant 1.0 : f32
    %fill_ct = linalg.fill(%f1, %c) : f32, tensor<16xf32> -> tensor<16xf32>
    %fct = bufferization.to_memref %fill_ct : memref<16xf32>
    %uc = memref.cast %fct :  memref<16xf32> to memref<*xf32>
    call @print_memref_f32(%uc): (memref<*xf32>) -> ()
    return
}

func private @print_memref_f32(memref<*xf32>)