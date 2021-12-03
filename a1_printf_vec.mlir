func @main() {
    %c = arith.constant dense<[0, 1, 2, 3]> : vector<4xi32>
    vector.print %c : vector<4xi32>
    return
}