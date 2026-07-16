#pragma once

#include <algorithm>
#include <optional>

#include <core/definitions.h>

// -----------------------------------------------------------------------------
// Box handling for trust-region steps
// -----------------------------------------------------------------------------
enum class BoxMode { Projection, Alpha };

struct BoxCtx {
    std::optional<dvec> x, lb, ub;
    BoxMode mode{BoxMode::Alpha};

    [[nodiscard]] inline bool active() const noexcept {
        return x && (lb || ub);
    }

    static double alpha_max(const dvec& x, const dvec& d,
                            const std::optional<dvec>& lb,
                            const std::optional<dvec>& ub,
                            double tau = 0.999999) {
        if ((!lb && !ub) || d.size() == 0) return 1.0;
        double amax = 1.0;
        if (lb) {
            for (int i = 0; i < d.size(); ++i)
                if (d[i] < 0.0) amax = std::min(amax, ((*lb)[i] - x[i]) / d[i]);
        }
        if (ub) {
            for (int i = 0; i < d.size(); ++i)
                if (d[i] > 0.0) amax = std::min(amax, ((*ub)[i] - x[i]) / d[i]);
        }
        amax = std::clamp(amax, 0.0, 1.0);
        return std::clamp(tau * amax, 0.0, 1.0);
    }

    static dvec pullback(const dvec& x, const dvec& p,
                         const std::optional<dvec>& lb,
                         const std::optional<dvec>& ub) {
        dvec xt = x + p;
        if (lb) xt = xt.cwiseMax(*lb);
        if (ub) xt = xt.cwiseMin(*ub);
        return xt - x;
    }

    [[nodiscard]] dvec enforce_step(const dvec& p) const {
        if (!active()) return p;
        const dvec& xx = *x;
        if (mode == BoxMode::Projection) return pullback(xx, p, lb, ub);
        const double a = alpha_max(xx, p, lb, ub);
        return a * p;
    }

    [[nodiscard]] dvec enforce_correction(const dvec& x_trial,
                                          const dvec& q) const {
        if (!active()) return q;
        if (mode == BoxMode::Projection) {
            dvec xc = x_trial + q;
            if (lb) xc = xc.cwiseMax(*lb);
            if (ub) xc = xc.cwiseMin(*ub);
            return xc - x_trial;
        }
        const double a = alpha_max(x_trial, q, lb, ub);
        return a * q;
    }
};
