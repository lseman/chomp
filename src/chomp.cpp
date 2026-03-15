// nlp_hybrid_cpp_nb.cpp
// C++23 + nanobind hybrid NLP solver wrapper (IP + SQP)
// Build target name suggestion: `chomp`
//
// Requires:
//   - nanobind (with Eigen support)
//   - Eigen
//   - fmt
//   - Your C++ IP/SQP headers: ../include/ip/stepper.h,
//     ../include/sqp/stepper.h
//
// Notes:
//   * We accept/return Eigen vectors (nanobind auto-converts to/from NumPy).
//   * Bounds (lb/ub) can be None or 1-D arrays; we forward as Eigen vectors.

#include <fmt/color.h>
#include <fmt/core.h>
#include <fmt/format.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <Eigen/Core>
#include <chrono>
#include <cmath>
#include <iostream>
#include <optional>
#include <string>

#include "../include/ip/stepper.h"
#include "../include/model.h"
#include "../include/sqp/stepper.h"

namespace nb = nanobind;
using nb::arg;
using dvec = Eigen::VectorXd;

// -------------------- small helpers --------------------

inline bool has_attr(const nb::object &o, const char *name) { return nb::hasattr(o, name); }

template <typename T>
inline void ensure_attr(nb::object &cfg, const char *name, T value) {
    if (!nb::hasattr(cfg, name)) { cfg.attr(name) = nb::cast(std::move(value)); }
}

template <typename T>
inline void ensure_attr_from_alias(nb::object &cfg, const char *name, const char *alias, const T &fallback) {
    if (!nb::hasattr(cfg, name)) { cfg.attr(name) = nb::cast(get_attr_or<T>(cfg, alias, fallback)); }
}

inline double get_float_attr_or(const nb::object &o, const char *name, double defval) {
    if (!nb::hasattr(o, name)) return defval;
    try {
        return nb::cast<double>(o.attr(name));
    } catch (...) { return defval; }
}

inline std::string get_str_attr_or(const nb::object &o, const char *name, const std::string &defval) {
    if (!nb::hasattr(o, name)) return defval;
    try {
        return nb::cast<std::string>(o.attr(name));
    } catch (...) { return defval; }
}

inline bool get_bool_attr_or(const nb::object &o, const char *name, bool defval) {
    if (!nb::hasattr(o, name)) return defval;
    try {
        return nb::cast<bool>(o.attr(name));
    } catch (...) { return defval; }
}

inline nb::object make_default_config_object() { return nb::cast(ChompConfig{}); }

inline void ensure_config_aliases(nb::object &cfg) {
    ensure_attr_from_alias(cfg, "tr_delta0", "delta0", 1.0);
    ensure_attr_from_alias(cfg, "delta0", "tr_delta0", 1.0);
    ensure_attr_from_alias(cfg, "tr_norm_type", "norm_type", std::string("2"));
    ensure_attr_from_alias(cfg, "norm_type", "tr_norm_type", std::string("2"));
}

// -------------------- Printer --------------------
static void print_iteration_row(int k, const SolverInfo &info, const std::string &mode, int last_header_ref,
                                bool force_header = false) {
    using fmt::color;
    using fmt::emphasis;
    using fmt::fg;

    static bool                  banner_printed = false;
    static std::optional<double> f_prev;
    static std::optional<double> theta_prev;

    if (!banner_printed) {
        fmt::print(fg(color::cyan) | emphasis::bold, "\nCHOMP");
        fmt::print(fg(color::light_gray) | emphasis::bold, " — made by ");
        fmt::print(fg(color::white) | emphasis::bold, "L. O. Seman\n");
        banner_printed = true;
    }

    if (force_header || k == 0 || (k - last_header_ref) >= 20) {
        fmt::print(fg(color::light_gray) | emphasis::bold, " {:>3s} {:>3s} {:>12s} {:>13s} {:>11s} {:>9s} {:>9s}\n",
                   "k", "st", "step", "f", "theta", "alpha", "Δ");
    }

    auto trend_color = [](std::optional<double> prev, double val) -> fmt::color {
        if (!prev.has_value() || std::isnan(val)) return color::white;
        return color::white; // (simple neutral trend color for now)
    };
    const auto f_col     = trend_color(f_prev, info.f);
    const auto theta_col = trend_color(theta_prev, info.theta);
    const auto st_col    = info.accepted ? color::green : color::red;

    fmt::print(" {:>3d} ", k);
    fmt::print(fg(st_col) | emphasis::bold, "{:>3s} ", info.accepted ? "A" : "R");

    fmt::print("{:>12.3e} ", info.step_norm);
    fmt::print(fg(f_col), "{:>13.6e} ", info.f);
    fmt::print(fg(theta_col), "{:>11.3e} ", info.theta);
    fmt::print("{:>9.2e} {:>9.2e}\n", info.alpha, info.tr_radius);

    if (!std::isnan(info.f)) f_prev = info.f;
    if (!std::isnan(info.theta)) theta_prev = info.theta;
}

// ==================== NLPSolver (nanobind) ====================

class chomp {
public:
    chomp(nb::object  f,
          nb::object  c_ineq_list, // list[Callable] or None
          nb::object  c_eq_list,   // list[Callable] or None
          nb::object  lb_or_none,  // None or 1-D array-like
          nb::object  ub_or_none,  // None or 1-D array-like
          const dvec &x0,          // 1-D
          nb::object  cfg_or_none)  // None or config-like
    {
        // --- config ---
        cfg_ = cfg_or_none.is_none() ? make_default_config_object() : cfg_or_none;
        ensure_config_aliases(cfg_);
        ensure_auto_defaults_(cfg_);

        // --- x0 (state) ---
        x_ = x0;
        n_ = static_cast<int>(x_.size());

        // Default empty bounds
        nb::object lb_py = nb::cast(dvec()); // size 0
        nb::object ub_py = nb::cast(dvec()); // size 0

        // lb
        if (!lb_or_none.is_none()) {
            try {
                dvec lb_vec = nb::cast<dvec>(lb_or_none);
                if (lb_vec.size() > 0) lb_py = nb::cast(std::move(lb_vec));
            } catch (...) {
                // keep empty if cast fails
            }
        }
        // ub
        if (!ub_or_none.is_none()) {
            try {
                dvec ub_vec = nb::cast<dvec>(ub_or_none);
                if (ub_vec.size() > 0) ub_py = nb::cast(std::move(ub_vec));
            } catch (...) {
                // keep empty if cast fails
            }
        }
        // Equalities/inequalities sizes
        mI_ = c_ineq_list.is_none() ? 0 : static_cast<int>(nb::len(c_ineq_list));
        mE_ = c_eq_list.is_none() ? 0 : static_cast<int>(nb::len(c_eq_list));

        // Multipliers
        lam_ = dvec::Zero(mI_);
        nu_  = dvec::Zero(mE_);

        // Mode selection
        std::string mode = get_str_attr_or(cfg_, "mode", "auto");
        if (mode != "ip" && mode != "sqp" && mode != "auto" && mode != "dfo") mode = "auto";
        mode_ = mode;

        ModelC *m = nullptr;
        try {
            m = new ModelC(f, c_ineq_list, c_eq_list, n_, lb_or_none, ub_or_none);
        } catch (const std::exception &e) {
            throw std::runtime_error(std::string("chomp: ModelC construction failed: ") + e.what());
        }

        try {
            m->run_convexity_analysis();
        } catch (const std::exception &e) {
            throw std::runtime_error(std::string("chomp: convexity analysis failed: ") + e.what());
        }

        // IP stepper + state
        try {
            ip_state_   = IPState(); // default-init
            ip_stepper_ = new InteriorPointStepper(cfg_, m);
        } catch (const std::exception &e) {
            throw std::runtime_error(std::string("chomp: IP stepper construction failed: ") + e.what());
        }

        // SQP stepper
        try {
            sqp_stepper_ = new SQPStepper(cfg_, nb::none(), m);
        } catch (const std::exception &e) {
            throw std::runtime_error(std::string("chomp: SQP stepper construction failed: ") + e.what());
        }

        // Trackers
        last_header_row_  = -1;
        last_switch_iter_ = -1000000000;
        prev_theta_.reset();
        reject_streak_ = small_alpha_streak_ = no_progress_streak_ = 0;
    }

    // Solve; returns the final x as an Eigen vector (NumPy array in Python)
    dvec solve(int max_iter = 100, double tol = 1e-8, bool verbose = true) {
        const double tol_stat = get_float_attr_or(cfg_, "tol_stat", 1e-6);
        const double tol_feas = get_float_attr_or(cfg_, "tol_feas", 1e-6);
        const double tol_comp = get_float_attr_or(cfg_, "tol_comp", 1e-6);

        for (int k = 0; k < max_iter; ++k) {
            SolverInfo info;
            if (mode_ == "ip") {
                info = ip_step_(k);
            } else if (mode_ == "sqp") {
                info = sqp_step_(k);
            } else if (mode_ == "dfo") {
                // DFO not wired here; keep placeholder behavior
                info.accepted = false;
                info.mode     = "dfo";
            } else {
                throw std::runtime_error("Unknown mode: " + mode_);
            }

            if (verbose) {
                print_iteration_row(k, info, mode_, last_header_row_, (k == 0 || (k - last_header_row_) >= 20));
                if (k == 0 || (k - last_header_row_) >= 20) last_header_row_ = k;
            }

            const double k_stat = info.stat;
            const double k_ineq = info.ineq;
            const double k_eq   = info.eq;
            const double k_comp = info.comp;

            if (k_stat <= tol_stat && k_ineq <= tol_feas && k_eq <= tol_feas && k_comp <= tol_comp) {
                if (verbose) fmt::print("✓ Converged at iteration {}\n", k);
                break;
            }
        }
        return x_;
    }

private:
    // ---- steps --------------------------------------------------------------
    SolverInfo ip_step_(int it) {
        auto [x_out, lam_out, nu_out, info] = ip_stepper_->step(x_, lam_, nu_, it, ip_state_);
        if (info.accepted) {
            x_   = std::move(x_out);
            lam_ = std::move(lam_out);
            nu_  = std::move(nu_out);
        }
        return info;
    }

    SolverInfo sqp_step_(int it) {
        auto [x_out, lam_out, nu_out, info] = sqp_stepper_->step(x_, lam_, nu_, it);

        if (info.accepted) {
            if (get_bool_attr_or(cfg_, "use_watchdog", false)) {
                watchdog_update_(x_out); // stub (no-op)
            }
            x_   = std::move(x_out);
            lam_ = std::move(lam_out);
            nu_  = std::move(nu_out);
        }
        return info;
    }

    // ---- config defaults ----------------------------------------------------
    void ensure_auto_defaults_(nb::object &cfg) {
        // Initial mode decision
        ensure_attr(cfg, "ip_switch_theta", 1e-3);

        // IP → SQP switch
        ensure_attr(cfg, "auto_ip2sqp_theta_cut", 5e-5);
        ensure_attr(cfg, "auto_ip2sqp_mu_cut", 1e-6);
        ensure_attr(cfg, "auto_ip_min_iters", 3);

        // SQP → IP switch
        ensure_attr(cfg, "auto_sqp2ip_theta_blowup", 1e-2);
        ensure_attr(cfg, "auto_sqp2ip_stall_iters", 3);
        ensure_attr(cfg, "auto_sqp2ip_reject_streak", 2);
        ensure_attr(cfg, "auto_sqp2ip_small_alpha_streak", 3);
        ensure_attr(cfg, "auto_sqp_min_iters", 3);

        // Hysteresis and spacing
        ensure_attr(cfg, "auto_hysteresis_factor", 2.0);
        ensure_attr(cfg, "auto_min_iter_between_switches", 2);

        // Small step detection
        ensure_attr(cfg, "auto_small_alpha", 1e-6);
    }

    void watchdog_update_(const dvec &x_cand) {
        (void)x_cand; // stub no-op (port your Python watchdog if/when needed)
    }

private:
    // Core problem bits
    nb::object cfg_;
    nb::object model_;
    nb::object hess_;
    nb::object rest_;
    nb::object reg_;
    nb::object qp_;

    // Steppers
    InteriorPointStepper *ip_stepper_ = nullptr;
    IPState               ip_state_;
    SQPStepper           *sqp_stepper_ = nullptr;

    // State
    dvec x_;
    dvec lam_;
    dvec nu_;
    int  n_{0}, mI_{0}, mE_{0};

    // Mode + trackers
    std::string           mode_;
    int                   last_header_row_{-1};
    int                   last_switch_iter_{-1000000000};
    std::optional<double> prev_theta_;
    int                   reject_streak_{0};
    int                   small_alpha_streak_{0};
    int                   no_progress_streak_{0};
};

// -------------------- nanobind module --------------------

NB_MODULE(chomp, m) {
    m.doc() = "Hybrid NLP Solver (IP + SQP) — nanobind wrapper";

    nb::class_<ChompConfig>(m, "ChompConfig", nb::dynamic_attr())
        .def(nb::init<>())
        .def_rw("mode", &ChompConfig::mode)
        .def_rw("verbose", &ChompConfig::verbose)
        .def_rw("use_filter", &ChompConfig::use_filter)
        .def_rw("use_line_search", &ChompConfig::use_line_search)
        .def_rw("use_trust_region", &ChompConfig::use_trust_region)
        .def_rw("use_soc", &ChompConfig::use_soc)
        .def_rw("use_funnel", &ChompConfig::use_funnel)
        .def_rw("use_watchdog", &ChompConfig::use_watchdog)
        .def_rw("use_nonmonotone_ls", &ChompConfig::use_nonmonotone_ls)
        .def_rw("hessian_mode", &ChompConfig::hessian_mode)
        .def_rw("tol_feas", &ChompConfig::tol_feas)
        .def_rw("tol_stat", &ChompConfig::tol_stat)
        .def_rw("tol_comp", &ChompConfig::tol_comp)
        .def_rw("adaptive_tol", &ChompConfig::adaptive_tol)
        .def_rw("filter_gamma_theta", &ChompConfig::filter_gamma_theta)
        .def_rw("filter_gamma_f", &ChompConfig::filter_gamma_f)
        .def_rw("filter_theta_min", &ChompConfig::filter_theta_min)
        .def_rw("filter_margin_min", &ChompConfig::filter_margin_min)
        .def_rw("filter_max_size", &ChompConfig::filter_max_size)
        .def_rw("iter_scale_factor", &ChompConfig::iter_scale_factor)
        .def_rw("switch_theta", &ChompConfig::switch_theta)
        .def_rw("switch_f", &ChompConfig::switch_f)
        .def_rw("ls_armijo_f", &ChompConfig::ls_armijo_f)
        .def_rw("ls_armijo_theta", &ChompConfig::ls_armijo_theta)
        .def_rw("ls_backtrack", &ChompConfig::ls_backtrack)
        .def_rw("ls_max_iter", &ChompConfig::ls_max_iter)
        .def_rw("ls_nonmonotone_M", &ChompConfig::ls_nonmonotone_M)
        .def_rw("ls_min_alpha", &ChompConfig::ls_min_alpha)
        .def_rw("ls_use_wolfe", &ChompConfig::ls_use_wolfe)
        .def_rw("ls_wolfe_c", &ChompConfig::ls_wolfe_c)
        .def_rw("soc_kappa", &ChompConfig::soc_kappa)
        .def_rw("soc_tol", &ChompConfig::soc_tol)
        .def_rw("soc_max_corrections", &ChompConfig::soc_max_corrections)
        .def_rw("piqp_eps_abs", &ChompConfig::piqp_eps_abs)
        .def_rw("piqp_eps_rel", &ChompConfig::piqp_eps_rel)
        .def_rw("piqp_max_iter", &ChompConfig::piqp_max_iter)
        .def_rw("piqp_verbose", &ChompConfig::piqp_verbose)
        .def_rw("lbfgs_memory", &ChompConfig::lbfgs_memory)
        .def_rw("bfgs_min_curv", &ChompConfig::bfgs_min_curv)
        .def_rw("bfgs_powell_damp", &ChompConfig::bfgs_powell_damp)
        .def_rw("lbfgs_sparse_threshold", &ChompConfig::lbfgs_sparse_threshold)
        .def_rw("reg_mode", &ChompConfig::reg_mode)
        .def_rw("reg_sigma", &ChompConfig::reg_sigma)
        .def_rw("reg_target_cond", &ChompConfig::reg_target_cond)
        .def_rw("reg_adapt_factor", &ChompConfig::reg_adapt_factor)
        .def_rw("shift_invert_mode", &ChompConfig::shift_invert_mode)
        .def_rw("funnel_initial_tau", &ChompConfig::funnel_initial_tau)
        .def_rw("funnel_delta", &ChompConfig::funnel_delta)
        .def_rw("funnel_sigma", &ChompConfig::funnel_sigma)
        .def_rw("funnel_beta", &ChompConfig::funnel_beta)
        .def_rw("funnel_kappa", &ChompConfig::funnel_kappa)
        .def_rw("funnel_min_tau", &ChompConfig::funnel_min_tau)
        .def_rw("funnel_max_history", &ChompConfig::funnel_max_history)
        .def_rw("delta0", &ChompConfig::delta0)
        .def_prop_rw(
            "tr_delta0", [](const ChompConfig &cfg) { return cfg.delta0; },
            [](ChompConfig &cfg, double value) { cfg.delta0 = value; })
        .def_rw("delta_min", &ChompConfig::delta_min)
        .def_rw("delta_max", &ChompConfig::delta_max)
        .def_rw("eta1", &ChompConfig::eta1)
        .def_rw("eta2", &ChompConfig::eta2)
        .def_rw("gamma1", &ChompConfig::gamma1)
        .def_rw("gamma2", &ChompConfig::gamma2)
        .def_rw("zeta", &ChompConfig::zeta)
        .def_rw("cg_tol", &ChompConfig::cg_tol)
        .def_rw("cg_tol_rel", &ChompConfig::cg_tol_rel)
        .def_prop_rw(
            "cg_maxiter", [](const ChompConfig &cfg) { return cfg.cg_maxiter; },
            [](ChompConfig &cfg, int value) {
                cfg.cg_maxiter = value;
                cfg.cg_maxit = value;
            })
        .def_rw("neg_curv_tol", &ChompConfig::neg_curv_tol)
        .def_rw("constraint_tol", &ChompConfig::constraint_tol)
        .def_rw("max_active_set_iter", &ChompConfig::max_active_set_iter)
        .def_rw("curvature_aware", &ChompConfig::curvature_aware)
        .def_rw("use_prec", &ChompConfig::use_prec)
        .def_rw("prec_kind", &ChompConfig::prec_kind)
        .def_rw("box_mode", &ChompConfig::box_mode)
        .def_rw("norm_type", &ChompConfig::norm_type)
        .def_prop_rw(
            "tr_norm_type", [](const ChompConfig &cfg) { return cfg.norm_type; },
            [](ChompConfig &cfg, const std::string &value) { cfg.norm_type = value; })
        .def_rw("metric_shift", &ChompConfig::metric_shift)
        .def_rw("criticality_enabled", &ChompConfig::criticality_enabled)
        .def_rw("kappa_g", &ChompConfig::kappa_g)
        .def_rw("theta_crit", &ChompConfig::theta_crit)
        .def_rw("max_crit_shrinks", &ChompConfig::max_crit_shrinks)
        .def_rw("max_iter", &ChompConfig::max_iter)
        .def_rw("dopt_scaling", &ChompConfig::dopt_scaling)
        .def_rw("dopt_sigma_E", &ChompConfig::dopt_sigma_E)
        .def_rw("dopt_sigma_I", &ChompConfig::dopt_sigma_I)
        .def_rw("dopt_mu_target", &ChompConfig::dopt_mu_target)
        .def_rw("dopt_active_tol", &ChompConfig::dopt_active_tol)
        .def_rw("dopt_max_shift", &ChompConfig::dopt_max_shift)
        .def_rw("dopt_k_theta", &ChompConfig::dopt_k_theta)
        .def_rw("dopt_k_delta", &ChompConfig::dopt_k_delta)
        .def_rw("dopt_delta_ref", &ChompConfig::dopt_delta_ref)
        .def_rw("dopt_tol_eq_off", &ChompConfig::dopt_tol_eq_off)
        .def_rw("dopt_alpha_comp", &ChompConfig::dopt_alpha_comp)
        .def_rw("dopt_rel_cap", &ChompConfig::dopt_rel_cap)
        .def_rw("qp_aplusb_le0", &ChompConfig::qp_aplusb_le0)
        .def_rw("ip_mu_init", &ChompConfig::ip_mu_init)
        .def_rw("ip_mu_min", &ChompConfig::ip_mu_min)
        .def_rw("ip_tau", &ChompConfig::ip_tau)
        .def_rw("ip_tau_pri", &ChompConfig::ip_tau_pri)
        .def_rw("ip_tau_dual", &ChompConfig::ip_tau_dual)
        .def_rw("ip_sigma_power", &ChompConfig::ip_sigma_power)
        .def_rw("ip_kkt_method", &ChompConfig::ip_kkt_method)
        .def_rw("ip_use_shifted_barrier", &ChompConfig::ip_use_shifted_barrier)
        .def_rw("ip_shift_tau", &ChompConfig::ip_shift_tau)
        .def_rw("ip_shift_bounds", &ChompConfig::ip_shift_bounds)
        .def_rw("ip_eq_reg", &ChompConfig::ip_eq_reg)
        .def_rw("sigma_eps_abs", &ChompConfig::sigma_eps_abs)
        .def_rw("sigma_cap", &ChompConfig::sigma_cap)
        .def_rw("ip_switch_theta", &ChompConfig::ip_switch_theta)
        .def_rw("auto_ip2sqp_theta_cut", &ChompConfig::auto_ip2sqp_theta_cut)
        .def_rw("auto_ip2sqp_mu_cut", &ChompConfig::auto_ip2sqp_mu_cut)
        .def_rw("auto_ip_min_iters", &ChompConfig::auto_ip_min_iters)
        .def_rw("auto_sqp2ip_theta_blowup", &ChompConfig::auto_sqp2ip_theta_blowup)
        .def_rw("auto_sqp2ip_stall_iters", &ChompConfig::auto_sqp2ip_stall_iters)
        .def_rw("auto_sqp2ip_reject_streak", &ChompConfig::auto_sqp2ip_reject_streak)
        .def_rw("auto_sqp2ip_small_alpha_streak", &ChompConfig::auto_sqp2ip_small_alpha_streak)
        .def_rw("auto_sqp_min_iters", &ChompConfig::auto_sqp_min_iters)
        .def_rw("auto_hysteresis_factor", &ChompConfig::auto_hysteresis_factor)
        .def_rw("auto_min_iter_between_switches", &ChompConfig::auto_min_iter_between_switches)
        .def_rw("auto_small_alpha", &ChompConfig::auto_small_alpha)
        .def_rw("tol_obj_change", &ChompConfig::tol_obj_change)
        .def_rw("tol_step_abs", &ChompConfig::tol_step_abs)
        .def_rw("tol_step_rel", &ChompConfig::tol_step_rel)
        .def_rw("tol_theta_abs", &ChompConfig::tol_theta_abs)
        .def_rw("tol_kkt_abs", &ChompConfig::tol_kkt_abs)
        .def_rw("tiny_delta_stop", &ChompConfig::tiny_delta_stop)
        .def_rw("max_stall_iters", &ChompConfig::max_stall_iters)
        .def_rw("schur_dense_cutoff", &ChompConfig::schur_dense_cutoff)
        .def_rw("prec_type", &ChompConfig::prec_type)
        .def_rw("ssor_omega", &ChompConfig::ssor_omega)
        .def_rw("sym_ordering", &ChompConfig::sym_ordering)
        .def_rw("use_simd", &ChompConfig::use_simd)
        .def_rw("block_size", &ChompConfig::block_size)
        .def_rw("adaptive_gamma", &ChompConfig::adaptive_gamma)
        .def_rw("assemble_schur_if_m_small", &ChompConfig::assemble_schur_if_m_small)
        .def_rw("amd_dense_cutoff", &ChompConfig::amd_dense_cutoff)
        .def_rw("use_hvp", &ChompConfig::use_hvp)
        .def_rw("hvp_smw_threshold", &ChompConfig::hvp_smw_threshold)
        .def_rw("hvp_iterative_tol", &ChompConfig::hvp_iterative_tol)
        .def_rw("hvp_iterative_maxiter", &ChompConfig::hvp_iterative_maxiter)
        .def("__repr__", [](const ChompConfig &cfg) {
            return "<chomp.ChompConfig mode='" + cfg.mode + "' delta0=" + std::to_string(cfg.delta0) + ">";
        });

    m.attr("CHOMPConfig") = m.attr("ChompConfig");
    m.attr("CHOMPConf") = m.attr("ChompConfig");

    nb::class_<chomp>(m, "chomp")
        .def(nb::init<nb::object, nb::object, nb::object, nb::object, nb::object, const dvec &, nb::object>(), arg("f"),
             arg("c_ineq") = nb::none(), arg("c_eq") = nb::none(), arg("lb") = nb::none(), arg("ub") = nb::none(),
             arg("x0"), arg("config") = nb::none())
        .def("solve", &chomp::solve, arg("max_iter") = 100, arg("tol") = 1e-8, arg("verbose") = true,
             "Run hybrid solve; returns the final x (Eigen/NumPy).");
}
