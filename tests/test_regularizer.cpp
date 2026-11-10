#include <model/regularizer.h>
#include <blocks/filter.h>
#include <cassert>

static double minimum_eigenvalue(const regx::spmat& a) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es{Eigen::MatrixXd(a)};
    assert(es.info() == Eigen::Success);
    return es.eigenvalues().minCoeff();
}

int main() {
    Filter filter{FilterConfig{}};
    assert(filter.add_if_acceptable(0.0, 1.0));
    assert(!filter.is_acceptable(0.0, 100.0));
    assert(filter.is_acceptable(0.0, 0.5));
    assert(filter.add_if_acceptable(0.0, 1e-10));
    assert(filter.is_acceptable(0.0, 1e-12));
    regx::Regularizer regularizer;
    Eigen::Matrix2d dense;
    dense << 1.0, 2.0, 2.0, 1.0;
    regx::spmat a = dense.sparseView();
    auto [r, info] = regularizer.regularize(a);
    assert(std::abs(info.min_eig_before + 1.0) < 1e-12);
    assert(minimum_eigenvalue(r) > 0.0);

    // Same dimension, nnz and diagonal, different negative curvature.
    dense << 1.0, 4.0, 4.0, 1.0;
    a = dense.sparseView();
    auto [r2, info2] = regularizer.regularize(a);
    assert(std::abs(info2.min_eig_before + 3.0) < 1e-12);
    assert(minimum_eigenvalue(r2) > 0.0);

    dense << 2.0, 0.0, 0.0, 5.0;
    a = dense.sparseView();
    auto [r3, info3] = regularizer.regularize(a);
    assert(info3.min_eig_before == 2.0);
    assert(minimum_eigenvalue(r3) >= 2.0);

    // Force the Gershgorin fallback. Its lower bound must stay below the
    // true minimum eigenvalue, rather than being replaced by min(diagonal).
    regx::RegConfig cfg;
    cfg.small_matrix_threshold = 0;
    cfg.large_matrix_threshold = 0;
    regx::Regularizer fallback(cfg);
    dense << 1.0, 2.0, 2.0, 1.0;
    a = dense.sparseView();
    auto [r4, info4] = fallback.regularize(a);
    assert(info4.min_eig_before <= minimum_eigenvalue(a) + 1e-12);
    assert(minimum_eigenvalue(r4) > 0.0);

    // The default large-matrix path must certify negative curvature even
    // when all diagonal entries are positive.
    regx::spmat large(128, 128);
    large.setIdentity();
    large.coeffRef(0, 1) = 4.0; // deliberately asymmetric: symmetric part has eigenvalue -1
    auto [large_reg, large_info] = regularizer.regularize(large);
    assert(large_info.mode == "SPARSE_INERTIA");
    assert(large_info.sigma > 1.0);
    assert(minimum_eigenvalue(large_reg) > 0.0);
    assert((regx::dmat(large_reg) - regx::dmat(large_reg.transpose())).norm() == 0.0);
}
