#pragma once
#include <optional>
#include <functional>
#include <core/definitions.h>
#include <Eigen/Core>

// Second-Order Correction (SOC) — IPOPT-style
// After the predictor-corrector step, if the affine step was short,
// the linearization error in the complementarity conditions is large.
// SOC computes a correction direction d_soc that targets the nonlinear
// complementarity residuals:  s_aff * lam_aff - mu_target =/= 0
//
// The SOC RHS is:
//   For inequalities:  JI^T * (Sigma_s * (mu_target - s_aff * lam_aff))
//   For bounds:        (mu_target - sL_aff * zL_aff) / sL_aff  (lower)
//                      (mu_target - sU_aff * zU_aff) / sU_aff  (upper)
//
// The correction is applied as:  d_full = d_base + alpha_soc * d_soc
// where alpha_soc is a fraction-to-boundary step length.

class SOC {
   public:
    struct Config {
        double alpha_threshold = 0.3;  // Trigger SOC when alpha_aff < this
        double tau = 0.995;            // Fraction-to-boundary for SOC step
    };

    explicit SOC(Config config) : config_(config) {}
    explicit SOC(double alpha_threshold = 0.3, double tau = 0.995)
        : config_{alpha_threshold, tau} {}

    // Apply SOC correction to step in-place.
    // Returns true if SOC was applied, false if skipped.
    template <typename StepData, typename Bounds, typename Sigmas>
    bool apply(StepData& step, const spmat& W,
               const std::optional<spmat>& JE,
               const std::optional<spmat>& JI,
               const dvec& s, const dvec& lam,
               const dvec& zL, const dvec& zU, const Bounds& B,
               double alpha_aff, double mu_target,
               bool use_shifted, double tau_shift,
               double bound_shift, const Sigmas& Sg,
               std::function<KKTResult(const spmat&, const dvec&,
                                       const std::optional<spmat>&,
                                       const std::optional<dvec>&,
                                       std::string_view)>
                   solve_kkt) const {
        // Only trigger when affine step was short (linearization error matters)
        if (alpha_aff > config_.alpha_threshold) return false;

        // Build SOC RHS
        dvec rhs_x = build_soc_rhs(s, lam, zL, zU, step, B,
                                   alpha_aff, mu_target,
                                   use_shifted, tau_shift, bound_shift,
                                   Sg, JI);

        // Solve KKT for SOC correction
        auto soc_res = solve_kkt(W, rhs_x, JE, std::nullopt, "soc");

        // Compute fraction-to-boundary step length
        double alpha_soc = compute_soc_step_length(
            step, soc_res, s, zL, zU, B, use_shifted, tau_shift, bound_shift);

        if (alpha_soc < 1e-6) return false;

        // Apply correction: d_full = d_base + alpha_soc * d_soc
        step.dx += alpha_soc * soc_res.dx;
        if (soc_res.dy.size() > 0 && step.dnu.size() == soc_res.dy.size()) {
            step.dnu += alpha_soc * soc_res.dy;
        }

        // Recompute ds, dlam, dzL, dzU to stay consistent with new dx
        recompute_slack_and_duals(step, JI, s, lam, zL, zU, B,
                                  mu_target, use_shifted, tau_shift, bound_shift);

        return true;
    }

   private:
    Config config_;

    template <typename StepData, typename Bounds, typename Sigmas>
    dvec build_soc_rhs(const dvec& s, const dvec& lam,
                       const dvec& zL, const dvec& zU,
                       const StepData& step, const Bounds& B,
                       double alpha_aff, double mu_target,
                       bool use_shifted, double tau_shift,
                       double bound_shift, const Sigmas& Sg,
                       const std::optional<spmat>& JI) const {
        const int n = step.dx.size();
        dvec rhs_x = dvec::Zero(n);

        // Inequality complementarity corrections
        if (s.size() > 0 && JI && JI->nonZeros()) {
            const int mI = static_cast<int>(s.size());
            for (int i = 0; i < mI; ++i) {
                const double s_aff = std::max(
                    s[i] + alpha_aff * step.ds[i] +
                        (use_shifted ? tau_shift : 0.0),
                    1e-14);
                const double lam_aff = std::max(
                    lam[i] + alpha_aff * step.dlam[i], 1e-14);
                const double soc_term = mu_target - s_aff * lam_aff;

                if (std::abs(soc_term) > 1e-14 * mu_target &&
                    i < static_cast<int>(Sg.Sigma_s.size())) {
                    rhs_x += (*JI).row(i).transpose() *
                             (Sg.Sigma_s[i] * soc_term);
                }
            }
        }

        // Bound complementarity corrections
        for (int i = 0; i < n; ++i) {
            if (B.hasL[i]) {
                const double sL_aff = std::max(
                    B.sL[i] + alpha_aff * step.dx[i] +
                        (use_shifted ? bound_shift : 0.0),
                    1e-14);
                const double zL_aff = std::max(
                    zL[i] + alpha_aff * step.dzL[i], 1e-14);
                const double soc_term =
                    (mu_target - sL_aff * zL_aff) / sL_aff;
                if (std::abs(soc_term) > 1e-14 * mu_target) {
                    rhs_x[i] += soc_term;
                }
            }
            if (B.hasU[i]) {
                const double sU_aff = std::max(
                    B.sU[i] - alpha_aff * step.dx[i] +
                        (use_shifted ? bound_shift : 0.0),
                    1e-14);
                const double zU_aff = std::max(
                    zU[i] + alpha_aff * step.dzU[i], 1e-14);
                const double soc_term =
                    (mu_target - sU_aff * zU_aff) / sU_aff;
                if (std::abs(soc_term) > 1e-14 * mu_target) {
                    rhs_x[i] -= soc_term;
                }
            }
        }

        return rhs_x;
    }

    template <typename StepData>
    double compute_soc_step_length(const StepData& base_step,
                                   const KKTResult& soc_res,
                                   const dvec& s,
                                   const dvec& zL, const dvec& zU,
                                   const Bounds& B,
                                   bool use_shifted,
                                   double tau_shift, double bound_shift) const {
        double alpha = 1.0;
        constexpr double eps = 1e-14;

        // Primal feasibility: x and s must stay feasible
        const int n = static_cast<int>(base_step.dx.size());

        // x-step: x_new = x + (dx + alpha_soc * soc_dx)
        // For lower bounds: x - lb >= 0  =>  sL + dx + alpha_soc * soc_dx > 0
        //   alpha_soc > -(sL + dx) / soc_dx  when soc_dx < 0
        // For upper bounds: ub - x >= 0  =>  sU - dx - alpha_soc * soc_dx > 0
        //   alpha_soc > (dx - sU) / soc_dx  when soc_dx > 0
        for (int i = 0; i < n; ++i) {
            const double total_dx = base_step.dx[i] + soc_res.dx[i];
            if (B.hasL[i] && soc_res.dx[i] < -eps) {
                const double sL = use_shifted ? (B.sL[i] + bound_shift) : B.sL[i];
                const double denom = -(sL + total_dx);
                if (denom > 0) alpha = std::min(alpha, denom / (-soc_res.dx[i]));
            }
            if (B.hasU[i] && soc_res.dx[i] > eps) {
                const double sU = use_shifted ? (B.sU[i] + bound_shift) : B.sU[i];
                const double denom = -(sU - total_dx);
                if (denom > 0) alpha = std::min(alpha, denom / soc_res.dx[i]);
            }
        }

        // s-step: s_new = s + (ds + alpha_soc * soc_ds)
        if (s.size() > 0 && s.size() == static_cast<int>(soc_res.dx.size())) {
            for (int i = 0; i < static_cast<int>(s.size()); ++i) {
                if (soc_res.dx[i] < -eps && base_step.ds.size() > i) {
                    const double total_ds = base_step.ds[i] + soc_res.dx[i];
                    const double denom = -(s[i] + total_ds +
                                           (use_shifted ? tau_shift : 0.0));
                    if (denom > 0) alpha = std::min(alpha, denom / (-soc_res.dx[i]));
                }
            }
        }

        return std::min(config_.tau, alpha);
    }

    template <typename StepData>
    void recompute_slack_and_duals(StepData& step,
                                   const std::optional<spmat>& JI,
                                   const dvec& s, const dvec& lam,
                                   const dvec& zL, const dvec& zU,
                                   const Bounds& B,
                                   double mu_target,
                                   bool use_shifted,
                                   double tau_shift,
                                   double bound_shift) const {
        const int n = static_cast<int>(step.dx.size());

        // ds = -JI * dx  (equality with slack: s + JI*x = 0 => ds + JI*dx = 0)
        if (s.size() > 0 && JI && JI->nonZeros()) {
            step.ds = -(*JI) * step.dx;
            step.dlam.resize(s.size());
            for (int i = 0; i < static_cast<int>(s.size()); ++i) {
                const double d = std::max(
                    use_shifted ? (s[i] + tau_shift) : s[i], 1e-14);
                step.dlam[i] = (mu_target - d * lam[i] - lam[i] * step.ds[i]) / d;
            }
        }

        // Bound dual steps
        step.dzL.resize(n);
        step.dzU.resize(n);
        for (int i = 0; i < n; ++i) {
            if (B.hasL[i]) {
                const double d = std::max(
                    use_shifted ? (B.sL[i] + bound_shift) : B.sL[i], 1e-14);
                step.dzL[i] = (mu_target - d * zL[i] - zL[i] * step.dx[i]) / d;
            }
            if (B.hasU[i]) {
                const double d = std::max(
                    use_shifted ? (B.sU[i] + bound_shift) : B.sU[i], 1e-14);
                step.dzU[i] = (mu_target - d * zU[i] + zU[i] * step.dx[i]) / d;
            }
        }
    }
};
