#pragma once

#include <Eigen/Core>
#include <algorithm>

namespace ip_detail {

// Largest common step preserving positivity of a nonnegative dual vector.
// Inactive bound entries have z=dz=0 and impose no restriction.
inline double dual_step_limit(const Eigen::VectorXd& z,
                              const Eigen::VectorXd& dz, double tau) {
    double alpha = 1.0;
    for (Eigen::Index i = 0; i < z.size(); ++i) {
        if (dz[i] < 0.0) alpha = std::min(alpha, tau * (-z[i] / dz[i]));
    }
    return std::clamp(alpha, 0.0, 1.0);
}

} // namespace ip_detail
