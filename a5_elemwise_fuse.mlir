#map0 = affine_map<(d0) -> (d0)>
func @main() {
    %c = memref.alloc() : memref<10xf32> 
    %c2 = memref.alloc() : memref<10xf32>
    %c5 = memref.alloc() : memref<10xf32>
    %res = memref.alloc() : memref<10xf32>
    %f1 = arith.constant 1.0 : f32
    %f2 = arith.constant 2.0 : f32
    %f5 = arith.constant 5.0 : f32
    linalg.fill(%f1, %c) : f32, memref<10xf32>
    linalg.fill(%f2, %c2) : f32, memref<10xf32>
    linalg.fill(%f5, %c5) : f32, memref<10xf32>
    affine.for %i0 = 0 to 10 {
      %t0 = affine.load %c[%i0] : memref<10xf32>
      %t2 = affine.load %c2[%i0] : memref<10xf32>      
      %t3 = arith.addf %t0, %t2 : f32
      affine.store %t3, %res[%i0] : memref<10xf32>
    }

    affine.for %i0 = 0 to 10 {
      %t0 = affine.load %res[%i0] : memref<10xf32>
      %t2 = affine.load %c5[%i0] : memref<10xf32>      
      %t3 = arith.mulf %t0, %t2 : f32
      affine.store %t3, %res[%i0] : memref<10xf32>
    }
    
    
    %uc = memref.cast %res :  memref<10xf32> to memref<*xf32>
    call @print_memref_f32(%uc): (memref<*xf32>) -> ()
    return
}

func private @print_memref_f32(memref<*xf32>)

func @sibling_fusion(%arg0: memref<10x10xf32>, %arg1: memref<10x10xf32>,
                         %arg2: memref<10x10xf32>, %arg3: memref<10x10xf32>,
                         %arg4: memref<10x10xf32>) {
      affine.for %arg5 = 0 to 3 {
        affine.for %arg6 = 0 to 3 {
          %0 = affine.load %arg0[%arg5, %arg6] : memref<10x10xf32>
          %1 = affine.load %arg1[%arg5, %arg6] : memref<10x10xf32>
          %2 = arith.mulf %0, %1 : f32
          affine.store %2, %arg3[%arg5, %arg6] : memref<10x10xf32>
        }
      }
      affine.for %arg5 = 0 to 3 {
        affine.for %arg6 = 0 to 3 {
          %0 = affine.load %arg0[%arg5, %arg6] : memref<10x10xf32>
          %1 = affine.load %arg2[%arg5, %arg6] : memref<10x10xf32>
          %2 = arith.addf %0, %1 : f32
          affine.store %2, %arg4[%arg5, %arg6] : memref<10x10xf32>
        }
      }
      return
    }