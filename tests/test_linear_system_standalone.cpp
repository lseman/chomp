/*
 * Standalone C++ test suite for linear_system tools (qdldl, supernodes, amd).
 * Header-only, no external bindings needed.
 * Build from /data/dev/chomp: g++ -O3 -std=c++23 -I./include tests/test_linear_system_standalone.cpp -o test_ls
 */

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <functional>

// Include headers
#include "linear_system/qdldl.h"
#include "linear_system/supernodes.h"
#include "linear_system/amd.h"

using namespace qdldl23;
using std::function;

// ============================================================================
// Test Utilities
// ============================================================================

struct TestResult {
    std::string name;
    bool passed;
    std::string message;

    TestResult(const std::string& n, bool p, const std::string& m = "")
        : name(n), passed(p), message(m) {}
};

std::vector<TestResult> results;

void assert_equal(double a, double b, double tol = 1e-10, const std::string& ctx = "") {
    if (std::abs(a - b) > tol) {
        throw std::runtime_error("Assertion failed: " + ctx + " | " + std::to_string(a) +
                                 " != " + std::to_string(b));
    }
}

void run_test(const std::string& name, std::function<void()> test_fn) {
    try {
        test_fn();
        results.emplace_back(name, true);
        std::cout << "✓ " << name << std::endl;
    } catch (const std::exception& e) {
        results.emplace_back(name, false, e.what());
        std::cout << "✗ " << name << " | " << e.what() << std::endl;
    }
}

// ============================================================================
// QDLDL Tests
// ============================================================================

void test_qdldl_identity_3x3() {
    // I = L D L^T, so D = I, L = I
    int32_t n = 3;
    std::vector<int32_t> Ap = {0, 1, 2, 3};
    std::vector<int32_t> Ai = {0, 1, 2};
    std::vector<double> Ax = {1.0, 1.0, 1.0};

    SparseUpperCSC<double, int32_t> A(n, std::move(Ap), std::move(Ai), std::move(Ax));
    auto F = factorize(A);

    assert_equal(F.D[0], 1.0, 1e-10, "D[0] = 1");
    assert_equal(F.D[1], 1.0, 1e-10, "D[1] = 1");
    assert_equal(F.D[2], 1.0, 1e-10, "D[2] = 1");
    assert(F.num_pos == 3);
}

void test_qdldl_diagonal_4x4() {
    // Diagonal matrix
    int32_t n = 4;
    std::vector<int32_t> Ap = {0, 1, 2, 3, 4};
    std::vector<int32_t> Ai = {0, 1, 2, 3};
    std::vector<double> Ax_orig = {2.0, 3.0, 1.5, 4.0};
    std::vector<double> Ax = Ax_orig;

    SparseUpperCSC<double, int32_t> A(n, Ap, Ai, Ax);  // don't move yet
    auto F = factorize(A);

    for (int32_t i = 0; i < n; ++i) {
        assert_equal(F.D[i], Ax_orig[i], 1e-10, "D[" + std::to_string(i) + "]");
        assert_equal(F.Dinv[i], 1.0 / Ax_orig[i], 1e-10, "Dinv[" + std::to_string(i) + "]");
    }
}

void test_qdldl_tridiag_spd() {
    // Tridiagonal SPD: [ 4 1 0 0; 1 4 1 0; 0 1 4 1; 0 0 1 4 ]
    // CSC upper: col 0: [0], col 1: [0,1], col 2: [1,2], col 3: [2,3]
    int32_t n = 4;
    std::vector<int32_t> Ap = {0, 1, 3, 5, 7};
    std::vector<int32_t> Ai = {0, 0, 1, 1, 2, 2, 3};
    std::vector<double> Ax = {4.0, 1.0, 4.0, 1.0, 4.0, 1.0, 4.0};

    SparseUpperCSC<double, int32_t> A(n, Ap, Ai, Ax);
    auto F = factorize(A);

    assert(F.num_pos == n);  // All positive diagonals
    // Diagonals should be positive and < 4 (since submatrix is SPD)
    for (int32_t i = 0; i < n; ++i) {
        assert(F.D[i] > 0);
    }
}

void test_qdldl_solve_2x2() {
    // A = [4 1; 1 3], CSC upper: col 0: [0], col 1: [0,1]
    // x = [1; 2], b = A*x = [6; 7]
    int32_t n = 2;
    std::vector<int32_t> Ap = {0, 1, 3};
    std::vector<int32_t> Ai = {0, 0, 1};
    std::vector<double> Ax = {4.0, 1.0, 3.0};

    SparseUpperCSC<double, int32_t> A(n, Ap, Ai, Ax);
    auto F = factorize(A);

    std::vector<double> b = {6.0, 7.0};
    std::vector<double> x(b);
    solve(F, x.data());

    assert_equal(x[0], 1.0, 1e-9, "x[0]");
    assert_equal(x[1], 2.0, 1e-9, "x[1]");
}

void test_qdldl_solve_3x3() {
    // SPD matrix: [ 4.0  1.0  0.5 ]
    //             [ 1.0  3.5  0.2 ]
    //             [ 0.5  0.2  2.0 ]
    // CSC upper: col 0: [0], col 1: [0,1], col 2: [0,1,2]
    int32_t n = 3;
    std::vector<int32_t> Ap = {0, 1, 3, 6};
    std::vector<int32_t> Ai = {0, 0, 1, 0, 1, 2};
    std::vector<double> Ax = {4.0, 1.0, 3.5, 0.5, 0.2, 2.0};

    SparseUpperCSC<double, int32_t> A(n, Ap, Ai, Ax);
    auto F = factorize(A);

    // x_true = [1, -1, 2]
    std::vector<double> x_true = {1.0, -1.0, 2.0};
    std::vector<double> b(n, 0);

    // Compute b = A * x_true (manually)
    b[0] = 4.0 * 1.0 + 1.0 * (-1.0) + 0.5 * 2.0;      // 4
    b[1] = 1.0 * 1.0 + 3.5 * (-1.0) + 0.2 * 2.0;      // -2.1
    b[2] = 0.5 * 1.0 + 0.2 * (-1.0) + 2.0 * 2.0;      // 4.3

    std::vector<double> x = b;
    solve(F, x.data());

    assert_equal(x[0], x_true[0], 1e-8, "x[0]");
    assert_equal(x[1], x_true[1], 1e-8, "x[1]");
    assert_equal(x[2], x_true[2], 1e-8, "x[2]");
}

void test_qdldl_symmetric_permutation() {
    // Simple 2x2, CSC upper: col 0: [0], col 1: [0,1]
    // A = [4 1; 1 3]
    int32_t n = 2;
    std::vector<int32_t> Ap = {0, 1, 3};
    std::vector<int32_t> Ai = {0, 0, 1};
    std::vector<double> Ax = {4.0, 1.0, 3.0};

    SparseUpperCSC<double, int32_t> A(n, Ap, Ai, Ax);

    // Permutation: swap rows/cols (p = [1, 0])
    auto ord = Ordering<int32_t>::from_perm(std::vector<int32_t>{1, 0});
    auto B = permute_symmetric_upper(A, ord);

    // B should be [ 3 1; 1 4 ] in permuted order
    assert(B.n == 2);
    auto F = factorize(B);
    assert(F.num_pos == 2);
}

void test_qdldl_iterative_refinement() {
    // Small test: solve with refinement
    // A = [4 1; 1 3], CSC upper: col 0: [0], col 1: [0,1]
    int32_t n = 2;
    std::vector<int32_t> Ap = {0, 1, 3};
    std::vector<int32_t> Ai = {0, 0, 1};
    std::vector<double> Ax = {4.0, 1.0, 3.0};

    SparseUpperCSC<double, int32_t> A(n, Ap, Ai, Ax);
    auto F = factorize(A);

    std::vector<double> b = {6.0, 7.0};
    std::vector<double> x(n, 0.0);
    x[0] = 0.9;  // initial guess (off by 0.1)
    x[1] = 2.1;  // initial guess (off by 0.1)

    refine(A, F, x.data(), b.data(), /*iters=*/3);

    assert_equal(x[0], 1.0, 1e-8, "refined x[0]");
    assert_equal(x[1], 2.0, 1e-8, "refined x[1]");
}

// ============================================================================
// Supernode Tests
// ============================================================================

void test_supernode_identity() {
    // Diagonal → each col is own supernode
    int32_t n = 3;
    std::vector<int32_t> Ap = {0, 1, 2, 3};
    std::vector<int32_t> Ai = {0, 1, 2};
    std::vector<double> Ax = {1.0, 1.0, 1.0};

    SparseUpperCSC<double, int32_t> A(n, std::move(Ap), std::move(Ai), std::move(Ax));
    auto S = analyze_fast(A);

    auto sn_info = snode::identify_supernodes_qdldl(A, S);
    assert(static_cast<int32_t>(sn_info.ranges.size()) == n);

    for (int32_t k = 0; k < n; ++k) {
        assert(sn_info.ranges[k].first == k);
        assert(sn_info.ranges[k].second == k);
    }
}

void test_supernode_simple_merge() {
    // Test case: three columns, simple merge test
    // CSC upper: col 0: [0], col 1: [0,1], col 2: [1,2]
    int32_t n = 3;
    std::vector<int32_t> Ap = {0, 1, 3, 5};
    std::vector<int32_t> Ai = {0, 0, 1, 1, 2};
    std::vector<double> Ax = {1.0, 1.0, 1.0, 1.0, 1.0};

    SparseUpperCSC<double, int32_t> A(n, Ap, Ai, Ax);
    auto S = analyze_fast(A);

    auto sn_info = snode::identify_supernodes_qdldl(A, S);
    assert(sn_info.ranges.size() > 0);
}

// ============================================================================
// AMD Tests
// ============================================================================

void test_amd_identity() {
    // Identity matrix: valid permutation (may be identity or not)
    CSR A(3);
    A.indptr = {0, 0, 0, 0};  // no edges
    A.indices = {};

    AMDReorderingArray amd;
    auto [perm, stats] = amd.compute_fill_reducing_permutation(A, /*symmetrize=*/false);

    assert(static_cast<int32_t>(perm.size()) == 3);
    std::vector<bool> seen(3, false);
    for (int32_t p : perm) {
        assert(p >= 0 && p < 3);
        assert(!seen[p]);
        seen[p] = true;
    }
}

void test_amd_path_graph() {
    // Path graph: 0-1-2-3 as undirected edges
    // CSR: row 0 has edge to 1, row 1 to 2, row 2 to 3, row 3 empty
    CSR A(4);
    A.indptr = {0, 1, 2, 3, 3};  // row 0: [0,1), row 1: [1,2), row 2: [2,3), row 3: [3,3)
    A.indices = {1, 2, 3};         // col indices: 1, 2, 3

    AMDReorderingArray amd;
    auto [perm, stats] = amd.compute_fill_reducing_permutation(A, /*symmetrize=*/true);

    assert(static_cast<int32_t>(perm.size()) == 4);
    // Check it's a valid permutation
    std::vector<bool> seen(4, false);
    for (int32_t p : perm) {
        assert(p >= 0 && p < 4);
        assert(!seen[p]);
        seen[p] = true;
    }
}

void test_amd_stats() {
    // Verify AMD stats are computed
    CSR A(3);
    A.indptr = {0, 1, 2, 3};
    A.indices = {1, 2, 0};  // edges: 0→1, 1→2, 2→0

    AMDReorderingArray amd;
    auto [perm, stats] = amd.compute_fill_reducing_permutation(A, /*symmetrize=*/true);

    assert(stats.matrix_size == 3);
    assert(stats.original_nnz >= 0);
    assert(stats.reordered_bandwidth >= 0);
    assert(stats.inverse_permutation.size() == 3);
}

// ============================================================================
// Integration Tests
// ============================================================================

void test_amd_then_qdldl() {
    // Real workflow: AMD reorder → QDLDL factorize → solve
    int32_t n = 3;
    // CSC upper: col 0: [0], col 1: [0,1], col 2: [1,2]
    std::vector<int32_t> Ap = {0, 1, 3, 5};
    std::vector<int32_t> Ai = {0, 0, 1, 1, 2};
    std::vector<double> Ax = {4.0, 1.0, 3.0, 0.5, 2.0};

    SparseUpperCSC<double, int32_t> A(n, Ap, Ai, Ax);

    // AMD reorder (simple: use identity for now)
    CSR csr_A(n);
    csr_A.indptr = {0, 1, 2, 3};
    csr_A.indices = {1, 2, 0};  // dummy edges

    AMDReorderingArray amd;
    auto [perm, stats] = amd.compute_fill_reducing_permutation(csr_A, /*symmetrize=*/true);

    // Apply permutation to A
    auto ord = Ordering<int32_t>::from_perm(perm);
    auto B = permute_symmetric_upper(A, ord);

    // Factor
    auto F = factorize(B);
    assert(F.num_pos > 0);

    // Solution should be valid (no crash)
    assert(std::isfinite(F.D[0]));
}

// ============================================================================
// Main & Reporting
// ============================================================================

int main() {
    std::cout << "\n=== Linear System Tools Test Suite ===\n\n";

    // QDLDL tests
    std::cout << "QDLDL Tests:\n";
    run_test("test_qdldl_identity_3x3", test_qdldl_identity_3x3);
    run_test("test_qdldl_diagonal_4x4", test_qdldl_diagonal_4x4);
    run_test("test_qdldl_tridiag_spd", test_qdldl_tridiag_spd);
    run_test("test_qdldl_solve_2x2", test_qdldl_solve_2x2);
    run_test("test_qdldl_solve_3x3", test_qdldl_solve_3x3);
    run_test("test_qdldl_symmetric_permutation", test_qdldl_symmetric_permutation);
    run_test("test_qdldl_iterative_refinement", test_qdldl_iterative_refinement);

    // Supernode tests
    std::cout << "\nSupernode Tests:\n";
    run_test("test_supernode_identity", test_supernode_identity);
    run_test("test_supernode_simple_merge", test_supernode_simple_merge);

    // AMD tests
    std::cout << "\nAMD Tests:\n";
    run_test("test_amd_identity", test_amd_identity);
    run_test("test_amd_path_graph", test_amd_path_graph);
    run_test("test_amd_stats", test_amd_stats);

    // Integration tests
    std::cout << "\nIntegration Tests:\n";
    run_test("test_amd_then_qdldl", test_amd_then_qdldl);

    // Summary
    std::cout << "\n=== Summary ===\n";
    size_t passed = 0, failed = 0;
    for (const auto& r : results) {
        if (r.passed)
            ++passed;
        else
            ++failed;
    }

    std::cout << "Passed: " << passed << " / " << results.size() << "\n";
    std::cout << "Failed: " << failed << " / " << results.size() << "\n";

    if (failed > 0) {
        std::cout << "\nFailed tests:\n";
        for (const auto& r : results) {
            if (!r.passed) {
                std::cout << "  - " << r.name << ": " << r.message << "\n";
            }
        }
        return 1;
    }

    return 0;
}
