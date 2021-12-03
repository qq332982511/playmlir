#!/bin/bash

LLVM_SRC_DIR=llvm-project/
build_dir="./build/"
install_dir="${build_dir}/install/"
cmake -GNinja \
  "-H$LLVM_SRC_DIR/llvm" \
  "-B$build_dir" \
  -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_ENABLE_LLD=OFF \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD="X86" \
  -DLLVM_INCLUDE_TOOLS=ON \
  -DLLVM_BUILD_TOOLS=ON \
  -DLLVM_INCLUDE_TESTS=OFF \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_BUILD_LLVM_DYLIB=ON \
  -DLLVM_LINK_LLVM_DYLIB=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$install_dir

cmake --build "$build_dir" -j$(nproc) --target all && cmake --install "$build_dir"