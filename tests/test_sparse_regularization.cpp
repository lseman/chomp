#include <model/sparse_regularization.h>
#include <Eigen/Cholesky>
#include <cassert>

static void check_result(const Eigen::SparseMatrix<double>& original,
                         const regx::SparseInertiaResult& result) {
    Eigen::MatrixXd expected(original);
    expected.diagonal().array() += result.shift;
    assert((Eigen::MatrixXd(result.matrix) - expected).norm() == 0.0);
    Eigen::LLT<Eigen::MatrixXd> reference(expected);
    assert(reference.info() == Eigen::Success);
    const Eigen::VectorXd rhs = Eigen::VectorXd::LinSpaced(original.rows(), -1.0, 1.0);
    const Eigen::VectorXd x = reference.solve(rhs);
    const double denominator = expected.norm() * x.norm() + rhs.norm();
    assert((expected * x - rhs).norm() / denominator < 1e-14);
}

int main() {
    Eigen::SparseMatrix<double> h(128, 128);
    for (int i = 0; i < 128; ++i) {
        h.insert(i, i) = 4.0;
        if (i) { h.insert(i - 1, i) = -1.0; h.insert(i, i - 1) = -1.0; }
    }
    h.makeCompressed();
    auto result = regx::regularize_sparse_inertia(h);
    assert(result.shift == 0.0 && result.factorizations == 1);
    check_result(h, result);

    // Positive diagonal, negative eigenvalue: diagonal-only probes miss it.
    h.setIdentity();
    h.coeffRef(0, 1) = h.coeffRef(1, 0) = 2.0;
    h.makeCompressed();
    result = regx::regularize_sparse_inertia(h);
    assert(result.shift > 1.0 && result.shift < 1.1);
    check_result(h, result);

    h *= 1e20;
    result = regx::regularize_sparse_inertia(h);
    assert(result.shift > 1e20 && result.shift < 1.1e20);
    check_result(h, result);

    h.resize(16, 16);
    h.setZero();
    result = regx::regularize_sparse_inertia(h);
    assert(result.shift > 0.0);
    check_result(h, result);

    Eigen::MatrixXd rank_one = Eigen::MatrixXd::Ones(100, 100);
    h = rank_one.sparseView();
    result = regx::regularize_sparse_inertia(h);
    assert(result.shift > 0.0);
    check_result(h, result);

    h.coeffRef(0, 0) = std::numeric_limits<double>::quiet_NaN();
    bool rejected = false;
    try { regx::regularize_sparse_inertia(h); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
}
