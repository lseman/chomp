#pragma once

#include <algorithm>
#include <functional>
#include <ip/stepper.h>
#include <memory>
#include <nlp/config.h>
#include <sqp/stepper.h>
#include <string>
#include <utility>

namespace nlp {

// Owns the compiled model, selected algorithm, primal/dual state and solve loop.
// Model construction/configuration currently require Python and the GIL.
class Solver {
public:
    virtual ~Solver() { close(); }

    Solver(const Solver &)            = delete;
    Solver &operator=(const Solver &) = delete;

    Solver(nb::object  f,
           nb::object  c_ineq_list, // list[Callable] or None
           nb::object  c_eq_list,   // list[Callable] or None
           nb::object  lb_or_none,  // None or 1-D array-like
           nb::object  ub_or_none,  // None or 1-D array-like
           const dvec &x0,          // 1-D
           nb::object  cfg_or_none) // None or config-like
    {
        // --- config ---
        cfg_ = cfg_or_none.is_none() ? make_default_config_object() : cfg_or_none;
        ensure_config_aliases(cfg_);

        // --- x0 (state) ---
        x_ = x0;
        n_ = static_cast<int>(x_.size());

        // Equalities/inequalities sizes
        mI_ = c_ineq_list.is_none() ? 0 : static_cast<int>(nb::len(c_ineq_list));
        mE_ = c_eq_list.is_none() ? 0 : static_cast<int>(nb::len(c_eq_list));

        // Multipliers
        lam_ = dvec::Zero(mI_);
        nu_  = dvec::Zero(mE_);

        // Mode selection
        std::string mode = get_str_attr_or(cfg_, "mode", "auto");
        if (mode != "ip" && mode != "sqp" && mode != "auto" && mode != "dfo") mode = "auto";
        // Automatic selection is based on problem structure. Keep one backend
        // for the lifetime of a solve; switching requires explicit state transfer.
        if (mode == "auto") {
            const bool bounded = (!lb_or_none.is_none() && nb::len(lb_or_none) > 0) ||
                                 (!ub_or_none.is_none() && nb::len(ub_or_none) > 0);
            mode               = (mI_ > 0 || bounded) ? "ip" : "sqp";
        }
        if (mode == "dfo") throw std::invalid_argument("chomp: DFO mode is not implemented");
        mode_           = mode;
        last_info_.mode = mode_;

        try {
            model_owner_ = std::make_unique<ModelC>(f, c_ineq_list, c_eq_list, n_, lb_or_none, ub_or_none);
        } catch (const std::exception &e) {
            throw std::runtime_error(std::string("chomp: ModelC construction failed: ") + e.what());
        }
        ModelC *m = model_owner_.get();

        try {
            const bool print_convexity_analysis = get_bool_attr_or(cfg_, "print_convexity_analysis", false);
            m->run_convexity_analysis(print_convexity_analysis);
        } catch (const std::exception &e) {
            throw std::runtime_error(std::string("chomp: convexity analysis failed: ") + e.what());
        }

        // IP stepper + state
        try {
            if (mode_ == "ip") ip_stepper_ = std::make_unique<InteriorPointStepper>(cfg_, m);
        } catch (const std::exception &e) {
            throw std::runtime_error(std::string("chomp: IP stepper construction failed: ") + e.what());
        }

        // SQP stepper
        try {
            if (mode_ == "sqp") sqp_stepper_ = std::make_unique<SQPStepper>(cfg_, nb::none(), m);
        } catch (const std::exception &e) {
            throw std::runtime_error(std::string("chomp: SQP stepper construction failed: ") + e.what());
        }
    }

    void close() {
        if (closed_) return;
        closed_ = true;

        sqp_stepper_.reset();
        ip_stepper_.reset();
        model_owner_.reset();

        cfg_ = nb::none();
        x_.resize(0);
        lam_.resize(0);
        nu_.resize(0);
        n_ = mI_ = mE_ = 0;
        mode_          = "closed";
    }

    const SolverInfo  &last_info() const { return last_info_; }
    int                iterations() const { return iterations_; }
    const std::string &status() const { return status_; }

    using Observer = std::function<void(int, const SolverInfo &)>;

    dvec solve(int max_iter = 100, const Observer &observe = {}) {
        if (closed_) return x_;
        const double tol_stat = get_float_attr_or(cfg_, "tol_stat", 1e-6);
        const double tol_feas = get_float_attr_or(cfg_, "tol_feas", 1e-6);
        const double tol_comp = get_float_attr_or(cfg_, "tol_comp", 1e-6);

        iterations_ = 0;
        status_     = "iteration_limit";
        for (int k = 0; k < max_iter; ++k) {
            SolverInfo info = step_(k);

            last_info_  = info;
            iterations_ = k + 1;
            if (observe) observe(k, info);

            const double k_stat = info.stat;
            const double k_ineq = info.ineq;
            const double k_eq   = info.eq;
            const double k_comp = info.comp;

            if (k_stat <= tol_stat && k_ineq <= tol_feas && k_eq <= tol_feas && k_comp <= tol_comp) {
                status_ = "converged";
                break;
            }
        }
        return x_;
    }

private:
    SolverInfo step_(int it) {
        auto [x_out, lam_out, nu_out, info] =
            mode_ == "ip" ? ip_stepper_->step(x_, lam_, nu_, it, ip_state_) : sqp_stepper_->step(x_, lam_, nu_, it);
        if (info.accepted) {
            x_   = std::move(x_out);
            lam_ = std::move(lam_out);
            nu_  = std::move(nu_out);
        }
        return info;
    }

private:
    // Core problem bits
    SolverInfo              last_info_;
    int                     iterations_{0};
    std::string             status_{"not_started"};
    nb::object              cfg_;
    std::unique_ptr<ModelC> model_owner_;

    // Steppers
    std::unique_ptr<InteriorPointStepper> ip_stepper_;
    IPState                               ip_state_;
    std::unique_ptr<SQPStepper>           sqp_stepper_;

    // State
    dvec x_;
    dvec lam_;
    dvec nu_;
    int  n_{0}, mI_{0}, mE_{0};
    bool closed_{false};

    std::string mode_;
};

} // namespace nlp
