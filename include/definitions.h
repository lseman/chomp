#pragma once
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cstring>  // std::memcpy
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>

#include "model.h"
using dvec = Eigen::VectorXd;
using dmat = Eigen::MatrixXd;
using spmat = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;

// The value type your map stores (adjust if yours differs)
using Val = std::variant<double, dvec, dmat, spmat>;
using Dict = std::unordered_map<std::string, Val>;

// ---------- small helpers ----------

namespace nb = nanobind;
using namespace nb::literals;

// Get attribute with fallback; works with any castable T.
template <class T>
static inline T get_attr_or(const nb::handle& obj, const char* name,
                            const T& fallback) {
    if (!obj || !nb::hasattr(obj, name)) return fallback;
    try {
        return nb::cast<T>(obj.attr(name));
    } catch (const nb::cast_error&) {
        return fallback;
    }
}

// ---------- Python attribute helpers ----------
namespace pyu {
[[nodiscard]] inline bool has_attr(const nb::object& o,
                                   const char* name) noexcept {
    return o.is_valid() && PyObject_HasAttrString(o.ptr(), name);
}

}  // namespace pyu

struct KKT {
    double stat{0.0}, eq{0.0}, ineq{0.0}, comp{0.0};
};

namespace kkt {
// ---------------------------- reusable API ----------------------------

// ------------------------------ config -------------------------------
struct KKTConfig {
    double cg_tol = 1e-4;
    int cg_maxit = 100;
    double ip_hess_reg0 = 1e-8;
    double schur_dense_cutoff = 0.25;
    std::string prec_type = "ssor";  // "jacobi" | "ssor" | "none"
    double ssor_omega = 1.0;
    std::string sym_ordering = "amd_custom";  // "amd" | "none"
    bool use_simd = true;
    int block_size = 256;
    bool adaptive_gamma = false;
    bool assemble_schur_if_m_small = true;
    bool use_prec = true;
    double delta_min =
        1e-12;  // gentle starting floor for δ₁ when 0 is passed in
    double delta_max = 1e+6;  // hard cap to avoid runaway δ₁

    // δ₂ shift for Schur system (used when LLT/CG struggles on S = G K^{-1} Gᵀ)
    double schur_delta2_min = 1e-12;  // initial δ₂ for S + δ₂ I
    double schur_delta2_max = 1e-2;   // cap for δ₂ growth (×10 per retry)

    // Second CG budget for the δ₂-shifted solve (first try uses cg_maxit)
    int cg_maxit2 = 2 * 200;  // e.g., if cg_maxit==200, set cg_maxit2==400

    // define amd_dense_cutoff
    double amd_dense_cutoff = 0.1;  // switch to dense AMD for n < m / cutoff
    bool amd_dense_cutoff_has_value =
        false;  // whether user set amd_dense_cutoff
    bool use_hvp = false;
    std::optional<double> hvp_smw_threshold =
        1e-3;  // Threshold for using SMW vs iterative
    std::optional<double> hvp_iterative_tol =
        1e-10;  // Tolerance for HVP iterative solver
    std::optional<int> hvp_iterative_maxiter =
        std::nullopt;  // Max iterations for HVP solver (auto if nullopt)
};

struct KKTReusable {
    virtual ~KKTReusable() = default;
    virtual std::pair<dvec, dvec> solve(const dvec& r1,
                                        const std::optional<dvec>& r2,
                                        double cg_tol = 1e-8,
                                        int cg_maxit = 200) = 0;
};

// ---------------------------- strategy base ---------------------------
struct KKTStrategy {
    virtual ~KKTStrategy() = default;
    virtual std::tuple<dvec, dvec, std::shared_ptr<KKTReusable>>
    factor_and_solve(ModelC* model_in, const spmat& W,
                     const std::optional<spmat>& G, const dvec& r1,
                     const std::optional<dvec>& r2, const KKTConfig& cfg,
                     std::optional<double> regularizer,
                     std::unordered_map<std::string, dvec>& cache,
                     double delta = 0.0,
                     std::optional<double> gamma = std::nullopt,
                     bool assemble_schur_if_m_small = true,
                     bool use_prec = true) = 0;
    std::string name;
};

}  // namespace kkt

struct ChompConfig : public kkt::KKTConfig {
    ChompConfig() {
        cg_tol = 1e-12;
        cg_maxit = cg_maxiter;
    }

    // Core toggles
    std::string mode = "auto";
    bool verbose = true;
    bool use_filter = true;
    bool use_line_search = true;
    bool use_trust_region = true;
    bool use_soc = true;
    bool use_funnel = false;
    bool use_watchdog = false;
    bool use_nonmonotone_ls = false;
    std::string hessian_mode = "exact";

    // Tolerances
    double tol_feas = 1e-5;
    double tol_stat = 1e-5;
    double tol_comp = 1e-5;
    bool adaptive_tol = true;

    // Filter
    double filter_gamma_theta = 1e-5;
    double filter_gamma_f = 1e-5;
    double filter_theta_min = 1e-8;
    double filter_margin_min = 1e-5;
    int filter_max_size = 100;
    double iter_scale_factor = 50.0;
    double switch_theta = 0.9;
    double switch_f = 1.0;

    // Line search
    double ls_armijo_f = 1e-4;
    double ls_armijo_theta = 1e-4;
    double ls_backtrack = 0.5;
    int ls_max_iter = 25;
    int ls_nonmonotone_M = 5;
    double ls_min_alpha = 1e-12;
    bool ls_use_wolfe = false;
    double ls_wolfe_c = 0.9;
    double ls_theta_restoration = 1e2;

    // SOC
    double soc_kappa = 0.8;
    double soc_tol = 1e-8;
    int soc_max_corrections = 2;

    // QP backend
    double piqp_eps_abs = 1e-8;
    double piqp_eps_rel = 1e-8;
    int piqp_max_iter = 500;
    bool piqp_verbose = false;

    // Hessian options
    int lbfgs_memory = 10;
    double bfgs_min_curv = 1e-8;
    bool bfgs_powell_damp = true;
    double lbfgs_sparse_threshold = 1e-12;

    // Regularization
    std::string reg_mode = "AUTO";
    double reg_sigma = 1e-8;
    double reg_target_cond = 1e12;
    double reg_adapt_factor = 2.0;
    std::string shift_invert_mode = "buckling";

    // Funnel
    double funnel_initial_tau = 1.0;
    double funnel_delta = 0.1;
    double funnel_sigma = 1e-4;
    double funnel_beta = 0.99;
    double funnel_kappa = 0.1;
    double funnel_min_tau = 1e-8;
    int funnel_max_history = 100;
    double funnel_kappa_initial = 0.25;
    double funnel_sigma_rho_f = 0.10;
    double funnel_theta_curv_scale = 1.0;
    double funnel_phi_alpha = 0.10;

    // Trust region / globalization
    double delta0 = 1.0;
    double eta1 = 0.1;
    double eta2 = 0.9;
    double gamma1 = 0.5;
    double gamma2 = 2.0;
    double zeta = 0.8;
    double cg_tol_rel = 0.1;
    int cg_maxiter = 200;
    double neg_curv_tol = 1e-14;
    double constraint_tol = 1e-10;
    int max_active_set_iter = 100;
    bool adaptive_zeta = true;
    bool curvature_aware = true;
    bool feasibility_emphasis = true;
    double rcond = 1e-12;
    double reg_floor = 1e-10;
    double reg_max = 1e6;
    bool cache_nullspace = true;
    std::string prec_kind = "auto_jacobi";
    bool criticality_enabled = true;
    double kappa_g = 1e-2;
    double theta_crit = 0.5;
    int max_crit_shrinks = 1;
    std::string norm_type = "2";
    double metric_shift = 1e-8;
    double tau_ftb = 0.995;
    int history_length = 10;
    bool non_monotone = false;
    int non_monotone_window = 5;
    int max_iter = 100;
    std::string box_mode = "alpha";

    // D-opt / QP stabilization
    std::string dopt_scaling = "ruiz";
    double dopt_sigma_E = 1e-2;
    double dopt_sigma_I = 1e-2;
    double dopt_mu_target = 1e-4;
    double dopt_active_tol = 1e-8;
    double dopt_max_shift = 1e2;
    double dopt_k_theta = 0.5;
    double dopt_k_delta = 0.5;
    double dopt_delta_ref = 1.0;
    double dopt_tol_eq_off = 1e-8;
    double dopt_alpha_comp = 0.5;
    double dopt_rel_cap = 0.5;
    bool qp_aplusb_le0 = true;

    // IP / barrier
    bool ip_exact_hessian = true;
    double ip_eq_reg = 1e-4;
    bool ip_use_shifted_barrier = true;
    double ip_shift_tau = 0.01;
    double ip_shift_bounds = 0.1;
    bool ip_shift_adaptive = true;
    double ip_mu_init = 1e-2;
    double ip_mu_min = 1e-12;
    double ip_sigma_power = 3.0;
    double ip_tau = 0.995;
    double ip_tau_pri = 0.995;
    double ip_tau_dual = 0.99;
    double ip_alpha_max = 1.0;
    double ip_alpha_min = 1e-10;
    double ip_alpha_backtrack = 0.5;
    double ip_armijo_coeff = 1e-4;
    int ip_ls_max = 5;
    double ip_dx_max = 1e3;
    double ip_theta_clip = 1e-2;
    double ip_slack_init_min = 1e-6;
    double ip_slack_init_pad = 1e-3;
    double sigma_eps_abs = 1e-8;
    double sigma_cap = 1e8;
    std::string ip_kkt_method = "hykkt";
    double ip_tr_init = 1e3;
    double ip_penalty_rho_init = 1.0;
    double ip_penalty_sigma_init = 0.1;
    double ip_penalty_rho_min = 1e-6;
    double ip_penalty_rho_max = 1e6;
    double ip_penalty_sigma_min = 1e-8;
    double ip_penalty_sigma_max = 1e3;
    double ip_theta_cubic_threshold = 1e-2;
    double ip_penalty_rho_relative = 10.0;
    double ip_min_step_norm = 1e-15;
    double ip_pred_threshold = 1e-16;
    double ip_tr_eta_excellent = 0.9;
    double ip_tr_eta_good = 0.7;
    double ip_tr_eta_poor = 0.15;
    double ip_tr_gamma_expand = 2.0;
    double ip_tr_gamma_shrink = 0.6;
    double ip_tr_beta = 0.25;
    double ip_tr_min = 1e-8;
    double ip_tr_max = 1e6;
    double ip_s_max = 100.0;
    double ip_shift_beta = 0.25;
    double ip_shift_tau_k_mu = 0.50;
    double ip_shift_tau_k_active = 0.50;
    double ip_shift_tau_k_stall = 0.25;
    double ip_active_tol = 1e-3;
    double ip_mu_target = 1e-4;
    double ip_shift_tau_max = 1.0;
    double ip_shift_tau_min = 0.0;
    double ip_bound_near_tol = 1e-6;
    double ip_shift_bounds_k_mu = 0.30;
    double ip_shift_bounds_k_frac = 0.75;
    double ip_shift_bounds_k_gap = 0.50;
    double ip_shift_bounds_max = 1.0;
    double ip_shift_bounds_min = 0.0;
    double ip_sigma_E = 0.10;
    double ip_sigma_E_min = 0.02;
    double ip_sigma_E_max = 0.50;
    double ip_sigmaE_k_prog = 0.60;
    double ip_sigmaE_k_cond = 0.80;
    double ip_max_shift = 1e-2;
    double ip_max_shift_rel = 0.10;
    double ip_fraction_to_boundary_tau = 0.995;

    // Gondzio / Mehrotra
    int gondzio_max_corrections = 3;
    double gondzio_gamma_a = 0.1;
    double gondzio_gamma_b = 10.0;
    double gondzio_beta_min = 0.1;
    double gondzio_beta_max = 10.0;
    double gondzio_tau_min = 0.005;
    bool gondzio_adaptive_gamma = true;
    double gondzio_progress_threshold = 0.1;
    double soc_threshold = 0.1;
    int max_soc_steps = 3;
    double mehrotra_safeguard_threshold = 0.1;
    bool adaptive_correction_count = true;

    // Auto mode switching
    double ip_switch_theta = 1e-3;
    double auto_ip2sqp_theta_cut = 5e-5;
    double auto_ip2sqp_mu_cut = 1e-6;
    int auto_ip_min_iters = 3;
    double auto_sqp2ip_theta_blowup = 1e-2;
    int auto_sqp2ip_stall_iters = 3;
    int auto_sqp2ip_reject_streak = 2;
    int auto_sqp2ip_small_alpha_streak = 3;
    int auto_sqp_min_iters = 3;
    double auto_hysteresis_factor = 2.0;
    int auto_min_iter_between_switches = 2;
    double auto_small_alpha = 1e-6;

    // Termination / misc
    double tol_obj_change = 1e-10;
    double tol_step_abs = 1e-12;
    double tol_step_rel = 1e-9;
    double tol_theta_abs = 1e-10;
    double tol_kkt_abs = 1e-8;
    bool tiny_delta_stop = true;
    int max_stall_iters = 5;
};

using CHOMPConfig = ChompConfig;
using CHOMPConf = ChompConfig;

struct SolverInfo {
    std::string mode{"ip"};  // algorithmic mode (ip, sqp, etc.)
    double step_norm{0.0};
    bool accepted{true};
    bool converged{true};

    double f{0.0};      // objective value
    double theta{0.0};  // merit or filter measure
    double stat{0.0};   // gradient norm (safe_inf_norm(g))
    double ineq{0.0};   // infinity norm of violated inequalities
    double eq{0.0};     // infinity norm of equality violations
    double comp{0.0};   // complementarity measure

    int ls_iters{0};    // line search iterations
    double alpha{0.0};  // step size
    double rho{0.0};    // trust region ratio
    double tr_radius{0.0};
    double delta{0.0};  // trust region radius
    double mu{0.0};     // barrier parameter

    bool shifted_barrier{false};  // whether the barrier was shifted
    double tau_shift{0.0};        // amount of barrier shift
    double bound_shift{0.0};      // amount of bound shift

    double penalty_rho{0.0};    // cubic penalty ρ parameter
    double penalty_sigma{0.0};  // cubic penalty σ parameter
    double step_quality_ratio{
        0.0};  // ratio of actual to predicted reduction in cubic penalty
    bool was_clipped{
        false};          // whether the step was clipped by the trust region
    int clip_streak{0};  // consecutive clipped steps
    int good_streak{0};  // consecutive good steps
    double theta_reduction{0.0};  // relative reduction in constraint violation
};
