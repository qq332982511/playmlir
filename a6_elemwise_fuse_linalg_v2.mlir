
#map0 = affine_map<(d0) -> (d0)>

func @main() {
    %c1 = linalg.init_tensor [10] : tensor<10xf32>
    %c2 = linalg.init_tensor [10] : tensor<10xf32>
    %c5 = linalg.init_tensor [10] : tensor<10xf32>
    
    %f1 = arith.constant 1.0 : f32
    %f2 = arith.constant 2.0 : f32
    %f5 = arith.constant 5.0 : f32
    %fill_c1 = linalg.fill(%f1, %c1) : f32, tensor<10xf32> -> tensor<10xf32>
    %fill_c2 = linalg.fill(%f2, %c2) : f32, tensor<10xf32> -> tensor<10xf32>
    %fill_c5 = linalg.fill(%f5, %c5) : f32, tensor<10xf32> -> tensor<10xf32>

    %res_tensor = linalg.init_tensor [10] : tensor<10xf32>
    %res_mid = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]}
      ins(%fill_c1, %fill_c2 : tensor<10xf32>,  tensor<10xf32>)
     outs(%res_tensor : tensor<10xf32>) {
    ^bb0(%a: f32, %b: f32, %c: f32) :
      %4 = arith.addf %a, %b : f32
      linalg.yield %4 : f32
    } -> tensor<10xf32>

    %res_mid2 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]}
      ins(%res_mid, %fill_c5 : tensor<10xf32>,  tensor<10xf32>)
     outs(%res_tensor : tensor<10xf32>) {
    ^bb0(%a: f32, %b: f32, %c: f32) :
      %4 = arith.addf %a, %b : f32
      linalg.yield %4 : f32
    } -> tensor<10xf32>


    %res = bufferization.to_memref %res_mid2 : memref<10xf32>
    %uc = memref.cast %res :  memref<10xf32> to memref<*xf32>
    call @print_memref_f32(%uc): (memref<*xf32>) -> ()
    return
}


func private @print_memref_f32(memref<*xf32>)

