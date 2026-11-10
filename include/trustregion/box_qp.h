#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tr_detail {

enum class BoxQPStatus { Optimal, TrustBoundary, IterationLimit };
struct BoxQPResult {
    Eigen::VectorXd step;
    int iterations = 0;
    BoxQPStatus status = BoxQPStatus::IterationLimit;
};

// Feasible active-set/truncated CG for a box intersected with a trust ball.
// Bounds are on the step and must contain zero. Active bounds constrain the
// search space; they are released when their multiplier has the wrong sign.
template <class ApplyHessian>
BoxQPResult box_truncated_cg(const ApplyHessian& apply,
                             const Eigen::VectorXd& g,
                             const Eigen::VectorXd& lower,
                             const Eigen::VectorXd& upper,
                             const Eigen::VectorXd& diagonal,
                             double radius, double tolerance, int max_iterations,
                             const Eigen::MatrixXd* metric = nullptr) {
    using Vector = Eigen::VectorXd;
    const Eigen::Index n = g.size();
    if (lower.size() != n || upper.size() != n || diagonal.size() != n ||
        !(radius > 0.0) || (lower.array() > 0.0).any() || (upper.array() < 0.0).any())
        throw std::invalid_argument("box QP: invalid dimensions or infeasible zero step");
    Vector p = Vector::Zero(n), gradient = g;
    Vector r(n), z(n), d(n), Hd(n), Hp(n);
    const Vector preconditioner = diagonal.cwiseAbs().cwiseMax(1e-12);
    std::vector<int> active(static_cast<size_t>(n), 0);
    for (Eigen::Index i = 0; i < n; ++i) {
        if (lower[i] == upper[i]) active[i] = 2;
        else if (lower[i] == 0.0 && g[i] >= 0.0) active[i] = -1;
        else if (upper[i] == 0.0 && g[i] <= 0.0) active[i] = 1;
    }
    auto project = [&](Vector& v) {
        for (Eigen::Index i = 0; i < n; ++i) if (active[i]) v[i] = 0.0;
    };
    auto restart = [&] {
        apply(p, Hp);
        gradient = g + Hp;
        r = -gradient;
        project(r);
        z = r.cwiseQuotient(preconditioner);
        d = z;
        return r.dot(z);
    };
    auto ball_limit = [&](const Vector& direction) {
        const Vector Md = metric ? Vector(*metric * direction) : direction;
        const double a = direction.dot(Md);
        if (!(a > 0.0)) return std::numeric_limits<double>::infinity();
        const double b = p.dot(Md);
        const double p_norm_squared = metric ? p.dot(*metric * p) : p.squaredNorm();
        const double c = std::min(0.0, p_norm_squared - radius * radius);
        const double root = std::sqrt(std::max(0.0, b * b - a * c));
        return b > 0.0 ? -c / (b + root) : (-b + root) / a;
    };
    double rz = restart();
    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        if (n == 0 || r.lpNorm<Eigen::Infinity>() <= tolerance) {
            // Verify using a fresh residual, including components discarded
            // by the projected CG recurrence.
            rz = restart();
            if (n && r.lpNorm<Eigen::Infinity>() > tolerance) continue;
            int release = -1;
            double worst = -tolerance;
            for (Eigen::Index i = 0; i < n; ++i) {
                const double multiplier = active[i] == -1 ? gradient[i] : -gradient[i];
                if ((active[i] == -1 || active[i] == 1) && multiplier < worst) {
                    worst = multiplier;
                    release = static_cast<int>(i);
                }
            }
            if (release < 0) return {std::move(p), iteration, BoxQPStatus::Optimal};
            active[release] = 0;
            rz = restart();
            continue;
        }
        apply(d, Hd);
        const double curvature = d.dot(Hd);
        const double cg_alpha = curvature > 0.0 ? rz / curvature : std::numeric_limits<double>::infinity();
        double bound_alpha = std::numeric_limits<double>::infinity();
        int blocking = -1, side = 0;
        for (Eigen::Index i = 0; i < n; ++i) {
            double candidate = std::numeric_limits<double>::infinity();
            int candidate_side = 0;
            if (d[i] < 0.0 && std::isfinite(lower[i])) {
                candidate = (lower[i] - p[i]) / d[i]; candidate_side = -1;
            } else if (d[i] > 0.0 && std::isfinite(upper[i])) {
                candidate = (upper[i] - p[i]) / d[i]; candidate_side = 1;
            }
            if (candidate < bound_alpha) {
                bound_alpha = std::max(0.0, candidate);
                blocking = static_cast<int>(i);
                side = candidate_side;
            }
        }
        const double trust_alpha = ball_limit(d);
        const double alpha = std::min({cg_alpha, bound_alpha, trust_alpha});
        if (!std::isfinite(alpha)) throw std::runtime_error("box QP: nonfinite step length");
        p.noalias() += alpha * d;
        p = p.cwiseMax(lower).cwiseMin(upper);
        if (trust_alpha <= std::min(cg_alpha, bound_alpha))
            return {std::move(p), iteration + 1, BoxQPStatus::TrustBoundary};
        if (blocking >= 0 && bound_alpha <= cg_alpha) {
            active[blocking] = side;
            p[blocking] = side < 0 ? lower[blocking] : upper[blocking];
            // Activate simultaneous bound hits in one pass.
            for (Eigen::Index i = 0; i < n; ++i) {
                if (d[i] < 0.0 && p[i] == lower[i]) active[i] = -1;
                if (d[i] > 0.0 && p[i] == upper[i]) active[i] = 1;
            }
            rz = restart();
            continue;
        }
        gradient.noalias() += alpha * Hd;
        r = -gradient;
        project(r);
        z = r.cwiseQuotient(preconditioner);
        const double next_rz = r.dot(z);
        d = z + (next_rz / std::max(rz, std::numeric_limits<double>::min())) * d;
        project(d);
        rz = next_rz;
    }
    return {std::move(p), max_iterations, BoxQPStatus::IterationLimit};
}

} // namespace tr_detail
