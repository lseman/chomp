#include <ip/factorization_cache.h>
#include <ip/dual_step.h>
#include <cassert>
#include <limits>

int main() {
    Eigen::VectorXd z(3), dz(3);
    z << 1.0, 0.0, 2.0;
    dz << -10.0, 0.0, 1.0;
    const double alpha = ip_detail::dual_step_limit(z, dz, 0.995);
    assert(std::abs(alpha - 0.0995) < 1e-15);
    assert((z + alpha * dz).minCoeff() >= 0.0);
    dz.setZero();
    assert(ip_detail::dual_step_limit(z, dz, 0.995) == 1.0);
    using Matrix = Eigen::SparseMatrix<double>;
    Matrix a(2, 2);
    a.insert(0, 0) = 1.0;
    a.insert(1, 1) = 2.0;
    a.makeCompressed();
    Matrix b = a;
    assert(ip_detail::same_factorization_matrix(a, b));
    b.coeffRef(0, 0) += 1e-10;
    assert(!ip_detail::same_factorization_matrix(a, b));
    b = a;
    b.coeffRef(0, 0) = std::numeric_limits<double>::quiet_NaN();
    assert(!ip_detail::same_factorization_matrix(a, b));
    assert(!ip_detail::same_factorization_matrix(b, b));
    b = a;
    b.coeffRef(0, 0) = std::numeric_limits<double>::infinity();
    assert(!ip_detail::same_factorization_matrix(b, b));
    b = a;
    b.uncompress();
    b.reserve(Eigen::Vector2i(4, 4));
    assert(ip_detail::same_factorization_matrix(a, b));
    b.insert(0, 1) = 0.0;
    assert(!ip_detail::same_factorization_matrix(a, b));
    Matrix c(3, 3);
    assert(!ip_detail::same_factorization_matrix(a, c));
    Matrix d(2, 2);
    d.insert(0, 0) = 1.0;
    d.insert(0, 1) = 2.0;
    assert(!ip_detail::same_factorization_matrix(a, d));
}
