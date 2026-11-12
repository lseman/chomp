// Python interface and interpreter-shutdown resource cleanup for CHOMP.
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nlp/solver.h>
#include <nlp/iteration_output.h>
#include <mutex>
#include <vector>

namespace nb = nanobind;
using nb::arg;

class chomp : public nlp::Solver {
public:
    chomp(nb::object f, nb::object cI, nb::object cE, nb::object lb,
          nb::object ub, const dvec &x0, nb::object cfg)
        : nlp::Solver(f, cI, cE, lb, ub, x0, cfg) {
        register_instance_(this);
    }
    ~chomp() { unregister_instance_(this); }

    nb::dict stats() const {
        nb::dict out;
        out["status"] = status();
        out["iterations"] = iterations();
        out["mode"] = last_info().mode;
        out["objective"] = last_info().f;
        out["stationarity"] = last_info().stat;
        out["inequality_violation"] = last_info().ineq;
        out["equality_violation"] = last_info().eq;
        out["complementarity"] = last_info().comp;
        out["mu"] = last_info().mu;
        out["alpha"] = last_info().alpha;
        out["trust_radius"] = last_info().tr_radius;
        out["accepted"] = last_info().accepted;
        return out;
    }

    dvec solve(int max_iter = 100, double tol = 1e-8, bool verbose = true) {
        // Retain the legacy argument; convergence uses config KKT tolerances.
        (void)tol;
        int last_header = -1;
        auto result = nlp::Solver::solve(max_iter, [&](int k, const SolverInfo &info) {
            if (verbose) {
                nlp::print_iteration_row(k, info, last_header);
                if (k == 0 || k - last_header >= 20) last_header = k;
            }
        });
        if (verbose && status() == "converged")
            fmt::print("✓ Converged at iteration {}\n", iterations() - 1);
        return result;
    }

    static void cleanup_live_instances() {
        std::vector<chomp *> snapshot;
        {
            std::lock_guard<std::mutex> lock(live_instances_mutex_());
            snapshot = live_instances_();
        }
        for (chomp *instance : snapshot) {
            if (instance) {
                try {
                    instance->close();
                } catch (...) {}
            }
        }
    }

private:
    static std::vector<chomp *> &live_instances_() {
        static std::vector<chomp *> instances;
        return instances;
    }

    static std::mutex &live_instances_mutex_() {
        static std::mutex m;
        return m;
    }

    static void register_instance_(chomp *instance) {
        std::lock_guard<std::mutex> lock(live_instances_mutex_());
        live_instances_().push_back(instance);
    }

    static void unregister_instance_(chomp *instance) {
        std::lock_guard<std::mutex> lock(live_instances_mutex_());
        auto                       &instances = live_instances_();
        instances.erase(std::remove(instances.begin(), instances.end(), instance), instances.end());
    }

};

// -------------------- nanobind module --------------------

NB_MODULE(chomp, m) {
    m.doc() = "Hybrid NLP Solver (IP + SQP) — nanobind wrapper";
    nb::module_::import_("ad");

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
                cfg.cg_maxit   = value;
            })
        .def_rw("neg_curv_tol", &ChompConfig::neg_curv_tol)
        .def_rw("constraint_tol", &ChompConfig::constraint_tol)
        .def_rw("max_active_set_iter", &ChompConfig::max_active_set_iter)
        .def_rw("curvature_aware", &ChompConfig::curvature_aware)
        .def_rw("feasibility_emphasis", &ChompConfig::feasibility_emphasis)
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
    m.attr("CHOMPConf")   = m.attr("ChompConfig");

    nb::class_<chomp>(m, "chomp")
        .def(nb::init<nb::object, nb::object, nb::object, nb::object, nb::object, const dvec &, nb::object>(), arg("f"),
             arg("c_ineq") = nb::none(), arg("c_eq") = nb::none(), arg("lb") = nb::none(), arg("ub") = nb::none(),
             arg("x0"), arg("config") = nb::none())
        .def("close", &chomp::close, "Release solver-owned Python/C++ resources early.")
        .def_prop_ro("stats", &chomp::stats, "Diagnostics from the most recent solve call.")
        .def(
            "__enter__", [](chomp &self) -> chomp & { return self; }, nb::rv_policy::reference_internal)
        .def("__exit__",
             [](chomp &self, nb::args) {
                 self.close();
                 return false;
             })
        .def("solve", &chomp::solve, arg("max_iter") = 100, arg("tol") = 1e-8, arg("verbose") = true,
             "Run hybrid solve; returns the final x (Eigen/NumPy).");

    m.def("_cleanup_live_solvers", &chomp::cleanup_live_instances);

    nb::module_ atexit = nb::module_::import_("atexit");
    atexit.attr("register")(m.attr("_cleanup_live_solvers"));
}
