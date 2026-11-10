#pragma once

#include <Eigen/Core>
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <limits>

namespace tr_detail {

// Minimize ||A*x-b||^2 + reg*||x||^2 without forming normal equations.
// Thin factors also avoid an n-by-n system when constraints are few.
inline Eigen::VectorXd regularized_least_squares(const Eigen::MatrixXd& a,
                                                const Eigen::VectorXd& b,
                                                double reg) {
    if (a.rows() != b.size()) throw std::invalid_argument("least squares: rhs dimension mismatch");
    if (a.rows() == 0 || a.cols() == 0) return Eigen::VectorXd::Zero(a.cols());
    if (!a.allFinite() || !b.allFinite()) throw std::invalid_argument("least squares: nonfinite input");
    const double r = std::max(reg, 1e-12);
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(a, Eigen::ComputeThinU | Eigen::ComputeThinV);
    if (svd.info() != Eigen::Success) throw std::runtime_error("least squares: SVD failed");
    Eigen::VectorXd weights = svd.singularValues();
    for (Eigen::Index i = 0; i < weights.size(); ++i) {
        const double sigma = weights[i];
        // sigma/(sigma^2+r), avoiding overflow in sigma^2.
        weights[i] = sigma > std::sqrt(r)
                         ? 1.0 / (sigma + r / sigma)
                         : (sigma / r) / (1.0 + (sigma / std::sqrt(r)) * (sigma / std::sqrt(r)));
    }
    return svd.matrixV() * (weights.array() * (svd.matrixU().transpose() * b).array()).matrix();
}

// Minimize ||A*x-b||^2 + reg*||x||^2, with the first free_columns
// unconstrained and all remaining coordinates nonnegative. Working-set NNLS
// re-solves the free columns after every activation or removal; post-solve
// clipping alone does not solve this problem.
inline Eigen::VectorXd mixed_nonnegative_least_squares(
    const Eigen::MatrixXd& a, const Eigen::VectorXd& b,
    Eigen::Index free_columns, double reg = 1e-12) {
    if (a.rows() != b.size() || free_columns < 0 || free_columns > a.cols())
        throw std::invalid_argument("mixed least squares: dimension mismatch");
    if (!a.allFinite() || !b.allFinite() || !std::isfinite(reg) || reg < 0)
        throw std::invalid_argument("mixed least squares: invalid input");
    const Eigen::Index n = a.cols();
    const double ridge = std::max(reg, 1e-12);
    Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
    std::vector<Eigen::Index> passive;
    for (Eigen::Index j = 0; j < free_columns; ++j) passive.push_back(j);
    auto solve_passive = [&]() {
        Eigen::MatrixXd ap(a.rows(), passive.size());
        for (Eigen::Index k = 0; k < (Eigen::Index)passive.size(); ++k)
            ap.col(k) = a.col(passive[k]);
        Eigen::VectorXd z = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd zp = regularized_least_squares(ap, b, ridge);
        for (Eigen::Index k = 0; k < zp.size(); ++k) z[passive[k]] = zp[k];
        return z;
    };
    x = solve_passive();
    const double tol = 64 * std::numeric_limits<double>::epsilon()
                       * std::max(1.0, a.norm() * b.norm());
    const Eigen::Index budget = std::max<Eigen::Index>(100, 10 * n * n);
    for (Eigen::Index iteration = 0; iteration < budget; ++iteration) {
        Eigen::VectorXd w = a.transpose() * (b - a * x) - ridge * x;
        Eigen::Index entering = -1;
        double best = tol;
        for (Eigen::Index j = free_columns; j < n; ++j) {
            if (std::find(passive.begin(), passive.end(), j) == passive.end()
                && w[j] > best) { best = w[j]; entering = j; }
        }
        if (entering < 0) return x;
        passive.push_back(entering);
        for (;;) {
            Eigen::VectorXd z = solve_passive();
            double alpha = 1.0;
            bool feasible = true;
            for (auto j : passive) if (j >= free_columns && z[j] <= 0) {
                feasible = false;
                alpha = std::min(alpha, (x[j] > 0 ? x[j] / (x[j] - z[j]) : 0.0));
            }
            if (feasible) { x = std::move(z); break; }
            x += alpha * (z - x);
            passive.erase(std::remove_if(passive.begin(), passive.end(),
                [&](Eigen::Index j) {
                    if (j >= free_columns && x[j] <= 0) { x[j] = 0; return true; }
                    return false;
                }), passive.end());
            if (++iteration >= budget)
                throw std::runtime_error("mixed least squares: working set did not converge");
        }
    }
    throw std::runtime_error("mixed least squares: working set did not converge");
}

} // namespace tr_detail
