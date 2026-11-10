#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <optional>
#include <utility>

// One primal point's Lagrangian Hessian. The owning model cache keys x;
// this cache keys the remaining state: inequality and equality multipliers.
class LagrangianHessianCache {
public:
    using Vector = Eigen::VectorXd;
    using Matrix = Eigen::SparseMatrix<double>;

    template <class Evaluate>
    const Matrix& get(const Eigen::Ref<const Vector>& lambda,
                      const Eigen::Ref<const Vector>& nu, Evaluate&& evaluate) {
        if (!matrix_ || !same(lambda, lambda_) || !same(nu, nu_)) {
            Matrix matrix = std::forward<Evaluate>(evaluate)();
            // Leave the existing entry intact if evaluation throws.
            lambda_ = lambda;
            nu_ = nu;
            matrix_ = std::move(matrix);
        }
        return *matrix_;
    }

    void reset() { matrix_.reset(); }

private:
    static bool same(const Eigen::Ref<const Vector>& a, const Vector& b) {
        return a.size() == b.size() && a.allFinite() &&
               (a.array() == b.array()).all();
    }

    std::optional<Matrix> matrix_;
    Vector lambda_, nu_;
};
