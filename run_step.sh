alias run_mlir_cpu_template="mlir-cpu-runner -O3 -e main -entry-point-result=void -shared-libs=/home/liujunjie/github/deeplearning/framework/mlir/build/lib/libmlir_runner_utils.so -shared-libs=/home/liujunjie/github/deeplearning/framework/mlir/build/lib/libmlir_c_runner_utils.so"

# mlir-opt a0_printf_scale.mlir 

# mlir-opt a0_printf_scale.mlir -convert-std-to-llvm -convert-vector-to-llvm

# mlir-opt a0_printf_scale.mlir -convert-std-to-llvm -convert-vector-to-llvm  | run_mlir_cpu_template

## swap vec and std will occur bug
# mlir-opt a1_printf_vec.mlir -convert-vector-to-llvm  -convert-std-to-llvm  | run_mlir_cpu_template

# mlir-opt a2_printf_memref.mlir -convert-memref-to-llvm  -convert-std-to-llvm  -reconcile-unrealized-casts | run_mlir_cpu_template

# mlir-opt a3_printf_fill.mlir --finalizing-bufferize -convert-linalg-to-loops  -convert-scf-to-std -convert-memref-to-llvm  -convert-std-to-llvm -reconcile-unrealized-casts | run_mlir_cpu_template

# mlir-opt a3.1_printf_fill_tensor.mlir --linalg-bufferize -convert-linalg-to-loops  -convert-scf-to-std -convert-memref-to-llvm  -convert-std-to-llvm  -reconcile-unrealized-casts | run_mlir_cpu_template

# mlir-opt a4_elemwise_add.mlir --lower-affine --finalizing-bufferize -convert-linalg-to-loops  -convert-scf-to-std -convert-memref-to-llvm  -convert-std-to-llvm -reconcile-unrealized-casts | run_mlir_cpu_template

# mlir-opt a5_elemwise_fuse.mlir --affine-loop-fusion="fusion-maximal" #--lower-affine --finalizing-bufferize -convert-linalg-to-loops  -convert-scf-to-std -convert-memref-to-llvm  -convert-std-to-llvm -reconcile-unrealized-casts | run_mlir_cpu_template

# mlir-opt a6_elemwise_fuse_linalg_v2.mlir -linalg-fuse-elementwise-ops --linalg-bufferize -convert-linalg-to-loops  -convert-scf-to-std  -convert-memref-to-llvm  -convert-std-to-llvm  -reconcile-unrealized-casts | run_mlir_cpu_template

# mlir-opt a7_matmul_base.mlir -linalg-fuse-elementwise-ops --linalg-bufferize -convert-linalg-to-loops  -convert-scf-to-std -convert-vector-to-llvm -convert-memref-to-llvm  -convert-std-to-llvm  -reconcile-unrealized-casts | run_mlir_cpu_template

mlir-opt a8_matmul_tile.mlir  --linalg-strategy-tile-pass="anchor-func=matmul anchor-op=linalg.matmul" --linalg-tile="tile-sizes=8"  --linalg-bufferize -tensor-bufferize -func-bufferize  --linalg-bufferize --scf-bufferize -convert-linalg-to-loops  -convert-scf-to-std  -convert-vector-to-llvm -convert-memref-to-llvm  -convert-std-to-llvm  -reconcile-unrealized-casts | run_mlir_cpu_template
