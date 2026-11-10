#include <model/hessian_cache.h>
#include <cassert>
#include <limits>
#include <stdexcept>

int main() {
    LagrangianHessianCache cache;
    Eigen::VectorXd lambda(1), nu(1);
    lambda << 2.0;
    nu << -1.0;
    int evaluations = 0;
    // At a fixed x: Hessian of f=x^2, cI=x^2, cE=3*x^2.
    auto evaluate = [&] {
        ++evaluations;
        Eigen::SparseMatrix<double> h(1, 1);
        h.insert(0, 0) = 2.0 + 2.0 * lambda[0] + 6.0 * nu[0];
        return h;
    };
    assert(cache.get(lambda, nu, evaluate).coeff(0, 0) == 0.0);
    cache.get(lambda, nu, evaluate);
    assert(evaluations == 1);
    lambda[0] = 3.0;
    assert(cache.get(lambda, nu, evaluate).coeff(0, 0) == 2.0);
    nu[0] = 1.0;
    assert(cache.get(lambda, nu, evaluate).coeff(0, 0) == 14.0);
    assert(evaluations == 3);

    lambda[0] = 4.0;
    try {
        cache.get(lambda, nu, []() -> Eigen::SparseMatrix<double> {
            throw std::runtime_error("evaluation failed");
        });
        assert(false);
    } catch (const std::runtime_error&) {}
    assert(cache.get(lambda, nu, evaluate).coeff(0, 0) == 16.0);
    cache.reset();
    cache.get(lambda, nu, evaluate);
    assert(evaluations == 5);
    lambda[0] = std::numeric_limits<double>::quiet_NaN();
    cache.get(lambda, nu, evaluate);
    cache.get(lambda, nu, evaluate);
    assert(evaluations == 7);
}
