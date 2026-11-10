#pragma once

#include <Eigen/SparseCore>
#include <cmath>

namespace ip_detail {

// Numeric factors belong to one exact matrix. Approximate comparisons can
// silently reuse a different Newton system, especially near convergence.
template <class SparseMatrix>
bool same_factorization_matrix(const SparseMatrix& a, const SparseMatrix& b) {
    if (a.rows() != b.rows() || a.cols() != b.cols() ||
        a.nonZeros() != b.nonZeros()) return false;
    for (Eigen::Index j = 0; j < a.outerSize(); ++j) {
        typename SparseMatrix::InnerIterator ai(a, j), bi(b, j);
        for (; ai && bi; ++ai, ++bi) {
            if (ai.index() != bi.index() || !std::isfinite(ai.value()) ||
                ai.value() != bi.value()) return false;
        }
        if (ai || bi) return false;
    }
    return true;
}

} // namespace ip_detail
