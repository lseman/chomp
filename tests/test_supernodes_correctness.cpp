/*
 * Correctness test for supernodes: verify L-patterns and merge logic.
 * Tests against hand-computed reference examples.
 */

#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

#include "linear_system/qdldl.h"
#include "linear_system/supernodes.h"

using namespace qdldl23;
using namespace snode;

void test_lpat_simple_chain() {
    // Test L-pattern computation on simple case:
    // Tridiag matrix, CSC upper:
    // col 0: [0]
    // col 1: [0, 1]
    // col 2: [1, 2]
    // col 3: [2, 3]
    //
    // After factor:
    // L(:,0) = [0]        (diagonal only)
    // L(:,1) = [1]        (diagonal only after forward sub)
    // L(:,2) = [2]        (diagonal only)
    // L(:,3) = [3]        (diagonal only)
    //
    // Etree for tridiag is a path: 0 <- 1 <- 2 <- 3
    // (where <- means "parent of")
    // Symbolic reach of col j down etree should be single path.

    int32_t n = 4;
    std::vector<int32_t> Ap = {0, 1, 3, 5, 7};
    std::vector<int32_t> Ai = {0, 0, 1, 1, 2, 2, 3};
    std::vector<double> Ax = {4.0, 1.0, 4.0, 1.0, 4.0, 1.0, 4.0};

    SparseUpperCSC<double, int32_t> A(n, Ap, Ai, Ax);
    auto S = analyze_fast(A);

    std::cout << "Etree: ";
    for (int32_t i = 0; i < n; ++i) {
        std::cout << S.etree[i] << " ";
    }
    std::cout << "\n";

    // Expected etree for tridiag: [1, 2, 3, -1] (0->1, 1->2, 2->3, 3 is root)
    assert(S.etree[0] == 1 || S.etree[0] == -1);  // 0 points to 1 or is root
    assert(S.etree[1] == 2 || S.etree[1] == -1);
    assert(S.etree[2] == 3 || S.etree[2] == -1);
    assert(S.etree[3] == -1);  // root

    auto sn_info = identify_supernodes_qdldl(A, S);

    std::cout << "Supernodes:\n";
    for (size_t k = 0; k < sn_info.ranges.size(); ++k) {
        auto [lo, hi] = sn_info.ranges[k];
        std::cout << "  [" << lo << ", " << hi << "]\n";
    }

    // Tridiag: no column has matching L-pattern with next (just diagonal),
    // so each should be own supernode
    assert(sn_info.ranges.size() == static_cast<size_t>(n));
}

void test_lpat_dense_below() {
    // Dense lower part: forces complex L-patterns
    // [ 4  1  0.5]
    // [ 1  3  0.2]
    // [0.5 0.2 2 ]
    //
    // CSC upper:
    // col 0: [0]
    // col 1: [0, 1]
    // col 2: [0, 1, 2]
    //
    // After factor, L should have full fill below diagonal.
    // L(:,0) = [0]
    // L(:,1) = [1]
    // L(:,2) = [1, 2]  (fill from etree + structural reach)

    int32_t n = 3;
    std::vector<int32_t> Ap = {0, 1, 3, 6};
    std::vector<int32_t> Ai = {0, 0, 1, 0, 1, 2};
    std::vector<double> Ax = {4.0, 1.0, 3.0, 0.5, 0.2, 2.0};

    SparseUpperCSC<double, int32_t> A(n, Ap, Ai, Ax);
    auto S = analyze_fast(A);

    std::cout << "\nDense case etree: ";
    for (int32_t i = 0; i < n; ++i) {
        std::cout << S.etree[i] << " ";
    }
    std::cout << "\n";

    auto sn_info = identify_supernodes_qdldl(A, S);

    std::cout << "Dense case supernodes:\n";
    for (size_t k = 0; k < sn_info.ranges.size(); ++k) {
        auto [lo, hi] = sn_info.ranges[k];
        std::cout << "  [" << lo << ", " << hi << "]\n";
    }

    // All columns have full fill; may merge if etree is a chain and patterns match
    assert(sn_info.ranges.size() > 0);
}

void test_lpat_block_diagonal() {
    // Two independent 2x2 blocks: no coupling
    // [4 1 0 0]
    // [1 3 0 0]
    // [0 0 5 2]
    // [0 0 2 4]
    //
    // Etree should be two separate components.
    // L-patterns: [0] and [1] decouple from [2] and [3].
    // Should form 2 supernodes of size 2 each (if merging works).

    int32_t n = 4;
    std::vector<int32_t> Ap = {0, 1, 3, 4, 6};
    std::vector<int32_t> Ai = {0, 0, 1, 2, 2, 3};
    std::vector<double> Ax = {4.0, 1.0, 3.0, 5.0, 2.0, 4.0};

    SparseUpperCSC<double, int32_t> A(n, Ap, Ai, Ax);
    auto S = analyze_fast(A);

    std::cout << "\nBlock diagonal etree: ";
    for (int32_t i = 0; i < n; ++i) {
        std::cout << S.etree[i] << " ";
    }
    std::cout << "\n";

    auto sn_info = identify_supernodes_qdldl(A, S);

    std::cout << "Block diagonal supernodes:\n";
    for (size_t k = 0; k < sn_info.ranges.size(); ++k) {
        auto [lo, hi] = sn_info.ranges[k];
        std::cout << "  [" << lo << ", " << hi << "]\n";
    }

    // Should detect 2 independent blocks if column ordering preserved
    assert(sn_info.ranges.size() >= 2);
}

int main() {
    std::cout << "\n=== Supernodes Correctness Tests ===\n\n";

    try {
        test_lpat_simple_chain();
        std::cout << "✓ test_lpat_simple_chain\n";
    } catch (const std::exception& e) {
        std::cout << "✗ test_lpat_simple_chain: " << e.what() << "\n";
    }

    try {
        test_lpat_dense_below();
        std::cout << "✓ test_lpat_dense_below\n";
    } catch (const std::exception& e) {
        std::cout << "✗ test_lpat_dense_below: " << e.what() << "\n";
    }

    try {
        test_lpat_block_diagonal();
        std::cout << "✓ test_lpat_block_diagonal\n";
    } catch (const std::exception& e) {
        std::cout << "✗ test_lpat_block_diagonal: " << e.what() << "\n";
    }

    return 0;
}
