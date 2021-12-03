#map0 = affine_map<(d0) -> (d0)>
func @main() {
    %c = memref.alloc() : memref<8xf32> 
    %c2 = memref.alloc() : memref<8xf32>
    %res = memref.alloc() : memref<8xf32>
    %f1 = arith.constant 1.0 : f32
    %f2 = arith.constant 2.0 : f32
    linalg.fill(%f1, %c) : f32, memref<8xf32>
    linalg.fill(%f2, %c2) : f32, memref<8xf32>
    affine.for %i0 = 0 to 8 {
      %t0 = affine.load %c[%i0] : memref<8xf32>
      %t2 = affine.load %c2[%i0] : memref<8xf32>
      %t3 = arith.addf %t0, %t2 : f32
      affine.store %t3, %res[%i0] : memref<8xf32>
    }
    
    %uc = memref.cast %res :  memref<8xf32> to memref<*xf32>
    call @print_memref_f32(%uc): (memref<*xf32>) -> ()
    return
}

func private @print_memref_f32(memref<*xf32>)