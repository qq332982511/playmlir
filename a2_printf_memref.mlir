func @main() {
    %c = memref.alloc() : memref<8xf32> 
    %uc = memref.cast %c :  memref<8xf32> to memref<*xf32>
    call @print_memref_f32(%uc): (memref<*xf32>) -> ()
    return
}

func private @print_memref_f32(memref<*xf32>)