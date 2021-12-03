func @main() {
    %c = memref.alloc() : memref<8xf32> 
    %f0 = arith.constant 1.0 : f32
    linalg.fill(%f0, %c) : f32, memref<8xf32>
    %uc = memref.cast %c :  memref<8xf32> to memref<*xf32>
    call @print_memref_f32(%uc): (memref<*xf32>) -> ()
    return
}

func private @print_memref_f32(memref<*xf32>)