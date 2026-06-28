#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <trustregion/subproblem.h>

// -----------------------------------------------------------------------------
// Trust-region model utilities
// -----------------------------------------------------------------------------
[[nodiscard]] inline dmat symmetrize(const dmat& A) {
    return 0.5 * (A + A.transpose());
}

[[nodiscard]] inline dmat psd_cholesky_with_shift(const dmat& S,
                                                  double shift_min) {
    Eigen::LLT<dmat> llt(S);
    if (llt.info() == Eigen::Success) return llt.matrixL();

    Eigen::SelfAdjointEigenSolver<dmat> es(0.5 * (S + S.transpose()));
    const auto& vals = es.eigenvalues();
    const auto& V = es.eigenvectors();

    double add = shift_min;
    if (vals.size()) add = std::max(shift_min, 1e-12 - vals.minCoeff());

    dvec pos = (vals.array() + add).max(shift_min).matrix();
    dmat Spos = V * pos.asDiagonal() * V.transpose();

    Eigen::LLT<dmat> llt2(Spos);
    if (llt2.info() != Eigen::Success)
        throw std::runtime_error("Cholesky failed after shift");
    return llt2.matrixL();
}

[[nodiscard]] inline double model_reduction_quad_into(const LinOp& H,
                                                      const dvec& g,
                                                      const dvec& p, dvec& Hp) {
    H_apply(H, p, Hp);
    return -(g.dot(p) + 0.5 * p.dot(Hp));
}

[[nodiscard]] inline double estimate_sigma_into(const LinOp& H, const Metric& M,
                                                const dvec& g, const dvec& p,
                                                dvec& Hp) {
    if (p.size() == 0) return 0.0;
    double den = std::pow(M.norm(p), 2);
    if (den <= 1e-14) return 0.0;
    H_apply(H, p, Hp);
    return std::max(0.0, -(p.dot(Hp) + p.dot(g)) / den);
}

[[nodiscard]] inline double pred_red_cubic_into(const LinOp& H, const Metric& M,
                                                const dvec& g, const dvec& p,
                                                double sigma, dvec& Hp) {
    H_apply(H, p, Hp);
    const double quad = g.dot(p) + 0.5 * p.dot(Hp);
    return -(quad + (sigma / 3.0) * std::pow(M.norm(p), 3));
}
