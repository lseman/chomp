#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace regx {

struct SparseInertiaResult {
    Eigen::SparseMatrix<double> matrix;
    double shift = 0.0;
    int factorizations = 0;
};

// For symmetric H, certify H + shift*I as positive definite using LDL^T
// pivots. No eigenvalue estimate or random probe serves as a certificate.
inline SparseInertiaResult regularize_sparse_inertia(
    const Eigen::SparseMatrix<double>& h, double absolute_floor = 1e-12,
    double target_condition = 1e12) {
    if (h.rows() != h.cols()) throw std::invalid_argument("regularization: matrix must be square");
    if (!(absolute_floor > 0.0) || !std::isfinite(absolute_floor) ||
        !(target_condition >= 1.0) || !std::isfinite(target_condition))
        throw std::invalid_argument("regularization: invalid pivot tolerance");
    SparseInertiaResult result{h};
    if (h.rows() == 0) return result;
    Eigen::VectorXd diagonal = h.diagonal();
    double scale = 0.0;
    for (Eigen::Index col = 0; col < h.outerSize(); ++col) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(h, col); it; ++it) {
            if (!std::isfinite(it.value())) throw std::invalid_argument("regularization: nonfinite matrix");
            scale = std::max(scale, std::abs(it.value()));
        }
        result.matrix.coeffRef(col, col) += 0.0; // fixed diagonal pattern for all trials
    }
    result.matrix.makeCompressed();
    // This controls pivot scale, not an estimate of the spectral condition number.
    const double floor = std::max({absolute_floor, scale / target_condition,
                                  64 * std::numeric_limits<double>::epsilon() * scale});
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> factor;
    factor.analyzePattern(result.matrix);
    auto certify = [&](double shift) {
        for (Eigen::Index i = 0; i < h.rows(); ++i)
            result.matrix.coeffRef(i, i) = diagonal[i] + shift;
        factor.factorize(result.matrix);
        ++result.factorizations;
        return factor.info() == Eigen::Success && factor.vectorD().allFinite() &&
               factor.vectorD().minCoeff() > floor;
    };
    if (certify(0.0)) return result;

    double lower = 0.0;
    double upper = std::max(2 * floor, 1e-6 * scale);
    bool found = false;
    for (int trial = 0; trial < 24; ++trial) {
        if (!std::isfinite(upper)) break;
        if (certify(upper)) { found = true; break; }
        lower = upper;
        upper *= 10.0;
    }
    if (!found) throw std::runtime_error("regularization: failed to certify positive definiteness");
    // Retain a modest margin above the inertia boundary. This avoids both an
    // unnecessarily large shift and a numerically singular Newton system.
    for (int trial = 0; trial < 8; ++trial) {
        const double midpoint = lower + 0.5 * (upper - lower);
        if (certify(midpoint)) upper = midpoint;
        else lower = midpoint;
    }
    if (!certify(upper)) throw std::runtime_error("regularization: inconsistent inertia certificate");
    result.shift = upper;
    return result;
}

} // namespace regx
