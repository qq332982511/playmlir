func @main() {
    %c = arith.constant 1.0 : f32
    vector.print %c : f32
    return
}