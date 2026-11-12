// model.h — Precompiled-from-Python AD model (hot path is pure C++)
#pragma once
#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/QR>
#include <Eigen/SparseCore>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <list>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "ad/ad_bindings.h" // GradFn, LagHessFn, compile_*
#include "ad/suspect.h"
#include "core/definitions.h"
#include "trustregion/definitions.h"
#include "model/hessian_cache.h"
#include "model/lbfgs.h"

struct ModelSuspectReport {
    using AnalyzerT = suspect::Analyzer<suspect_adapter::ModelView, ADNodePtr>;
    AnalyzerT::Summary                                 summary;
    std::unordered_map<std::size_t, suspect::NodeInfo> node_info; // optional
};

// ============================================================================
// Native AD functors (compiled once via Python `ad` module)
struct ADCompiled {
    std::function<double(const dvec &)> val;
    std::function<dvec(const dvec &)>   grad;
    std::function<spmat(const spmat &)> hess;
};

// ============================================================================
// Cached results for a given x (keep original structure)
struct EvalEntry {
    dvec        x;
    std::size_t hash{0};

    std::optional<double> f;
    std::optional<dvec>   g, cI, cE;
    std::optional<spmat>  JI, JE;
    LagrangianHessianCache H;

    mutable int access_order{0};

    EvalEntry()                                      = default;
    EvalEntry(EvalEntry &&other) noexcept            = default;
    EvalEntry &operator=(EvalEntry &&other) noexcept = default;
};

// ============================================================================
// Improved LRU cache with better performance
class EvalCache {
public:
    explicit EvalCache(std::size_t capacity = 16) : capacity_(capacity), current_access_(0) {
        entries_.reserve(capacity);
    }

    EvalEntry *find(const Eigen::Ref<const dvec> &x) {
        std::size_t h = hash_vec(x);
        for (auto &entry : entries_) {
            if (entry.hash == h && entry.x.size() == x.size() && entry.x.isApprox(x, 1e-14)) {
                entry.access_order = ++current_access_;
                return &entry;
            }
        }
        return nullptr;
    }

    EvalEntry &insert(const Eigen::Ref<const dvec> &x) {
        std::size_t h = hash_vec(x);
        if (entries_.size() < capacity_) {
            entries_.emplace_back();
            EvalEntry &entry   = entries_.back();
            entry.x            = x;
            entry.hash         = h;
            entry.access_order = ++current_access_;
            return entry;
        }
        auto lru_it  = std::min_element(entries_.begin(), entries_.end(), [](const EvalEntry &a, const EvalEntry &b) {
            return a.access_order < b.access_order;
        });
        lru_it->x    = x;
        lru_it->hash = h;
        lru_it->access_order = ++current_access_;

        lru_it->f.reset();
        lru_it->g.reset();
        lru_it->cI.reset();
        lru_it->cE.reset();
        lru_it->JI.reset();
        lru_it->JE.reset();
        lru_it->H.reset();
        return *lru_it;
    }

    static std::size_t hash_vec(const Eigen::Ref<const dvec> &v) {
        constexpr std::size_t FNV_OFFSET_BASIS = 14695981039346656037ULL;
        constexpr std::size_t FNV_PRIME        = 1099511628211ULL;
        std::size_t           hash             = FNV_OFFSET_BASIS;
        const double         *data             = v.data();
        const std::size_t     size             = v.size() * sizeof(double);
        const uint8_t        *bytes            = reinterpret_cast<const uint8_t *>(data);
        for (std::size_t i = 0; i < size; ++i) {
            hash ^= bytes[i];
            hash *= FNV_PRIME;
        }
        return hash;
    }

    std::size_t capacity() const { return capacity_; }
    void        set_capacity(std::size_t new_capacity) {
        capacity_ = new_capacity;
        if (entries_.size() > capacity_) {
            std::sort(entries_.begin(), entries_.end(),
                      [](const EvalEntry &a, const EvalEntry &b) { return a.access_order > b.access_order; });
            entries_.resize(capacity_);
        }
        entries_.reserve(capacity_);
    }

private:
    std::size_t            capacity_;
    std::vector<EvalEntry> entries_;
    int                    current_access_;
};

// ============================================================================
// ModelC with LRU multi-point cache + optional L-BFGS Lagrangian Hessian
// ============================================================================
class ModelC {
public:
    EvalEntry current_vals = EvalEntry();

    std::optional<double> get_f() const { return current_vals.f; }
    std::optional<dvec>   get_g() const { return current_vals.g; }
    std::optional<dvec>   get_cI() const { return current_vals.cI; }
    std::optional<dvec>   get_cE() const { return current_vals.cE; }
    std::optional<spmat>  get_JI() const { return current_vals.JI; }
    std::optional<spmat>  get_JE() const { return current_vals.JE; }
    std::optional<dvec>   get_lb() const { return lb_; }
    std::optional<dvec>   get_ub() const { return ub_; }

    std::shared_ptr<GradFn>              f_grad_;
    std::shared_ptr<LagHessFn>           f_hess_;
    std::vector<std::shared_ptr<GradFn>> cI_compiled_, cE_compiled_;

    dvec   Sigma_x;
    dvec   Sigma_s;
    double mu;
    double theta;
    int    it = 0;

    void set_params(const std::optional<dvec> &Sigma_x_in, const std::optional<dvec> &Sigma_s_in, const double mu_in,
                    const double theta_in, int it_in) {
        Sigma_x = Sigma_x_in ? Sigma_x_in.value() : dvec();
        Sigma_s = Sigma_s_in ? Sigma_s_in.value() : dvec();
        mu      = mu_in;
        theta   = theta_in;
        it      = it_in;
    }

    auto get_parms() { return std::make_tuple(Sigma_x, Sigma_s, mu, theta, it); }

    int get_mI() const { return mI_; }
    int get_mE() const { return mE_; }
    int get_n() const { return n_; }

    // New toggles
    void set_use_lbfgs_hess(bool v) { use_lbfgs_hess_ = v; }
    void set_lbfgs_mem(int m) { lbfgs_.m_mem = std::max(1, m); }
    void set_lbfgs_sparse_threshold(double t) { lbfgs_.densify_threshold = std::max(0.0, t); }

    // Materialize current LBFGS Hessian without updating memory
    spmat lbfgs_matrix(double threshold = 1e-12) const { return lbfgs_.build_sparse_matrix(n_, threshold); }

    // Return a LinOp for LBFGS Hessian-vector product (Hessian-free TR)
    // Uses the compact BFGS formula to apply B*v without forming the matrix
    LinOp lbfgs_hess_linop() const {
        LinOp op;
        op.n = n_;
        op.mv = [this](const dvec& v, dvec& out) {
            out = lbfgs_.apply_B(v);
        };
        return op;
    }

    ModelC(const ModelC &)            = default;
    ModelC(ModelC &&)                 = default;
    ModelC &operator=(const ModelC &) = default;
    ModelC &operator=(ModelC &&)      = default;

    explicit ModelC(nb::object f, std::optional<nb::object> c_ineq, std::optional<nb::object> c_eq, int n,
                    std::optional<nb::object> lb, std::optional<nb::object> ub, bool use_sparse = false,
                    bool use_lbfgs_hess = false, int lbfgs_mem = 20, double lbfgs_sparse_threshold = 1e-12)
        : n_(n), use_sparse_(use_sparse), cache_(16), use_lbfgs_hess_(use_lbfgs_hess) {
        f_grad_ = compile_objective_(f);
        f_hess_ = compile_lag_hess_(f, c_ineq, c_eq);

        if (c_ineq && !c_ineq->is_none()) {
            auto ineq_list = nb::cast<std::vector<nb::object>>(*c_ineq);
            for (auto &obj : ineq_list) cI_compiled_.push_back(compile_constraint_(obj));
        }
        if (c_eq && !c_eq->is_none()) {
            auto eq_list = nb::cast<std::vector<nb::object>>(*c_eq);
            for (auto &obj : eq_list) cE_compiled_.push_back(compile_constraint_(obj));
        }

        if (lb && !lb->is_none()) {
            lb_ = nb::cast<dvec>(*lb);
            if (lb_.size() != n_) throw std::runtime_error("ModelC: lb size mismatch");
        } else {
            lb_ = dvec::Constant(n_, -std::numeric_limits<double>::infinity());
        }
        if (ub && !ub->is_none()) {
            ub_ = nb::cast<dvec>(*ub);
            if (ub_.size() != n_) throw std::runtime_error("ModelC: ub size mismatch");
        } else {
            ub_ = dvec::Constant(n_, std::numeric_limits<double>::infinity());
        }

        mI_ = (int)cI_compiled_.size();
        mE_ = (int)cE_compiled_.size();

        // L-BFGS init
        lbfgs_.reset(n_);
        lbfgs_.m_mem             = std::max(1, lbfgs_mem);
        lbfgs_.densify_threshold = std::max(0.0, lbfgs_sparse_threshold);
        last_x_.resize(0);
        last_gradL_.resize(0);
    }

    // -------------------------------------------------------------------------
    // Evaluate (exact or LBFGS) Lagrangian Hessian
    spmat hess(const Eigen::Ref<const dvec> &x, const Eigen::Ref<const dvec> &lam,
               const Eigen::Ref<const dvec> &nu) const {
        if (!use_lbfgs_hess_) {
            // Exact AD path (unchanged)
            EvalEntry *e = cache_.find(x);
            if (!e) e = &cache_.insert(x);
            return e->H.get(lam, nu, [&] { return f_hess_->hess_sparse(x, lam, nu); });
        }
        std::cout << "Warning: using L-BFGS Hessian approximation; ensure "
                     "lbfgs_update(...) is called per accepted step.\n";
        // LBFGS path: DO NOT update here; just materialize from current memory.
        // Any (s,y) updates should be driven externally via lbfgs_update(...)
        return lbfgs_.build_sparse_matrix(n_, lbfgs_.densify_threshold);
    }

    CompiledWOp hvp;
    CompiledWOp getCompiledWOp() { return hvp; }

    void compileWOp(dvec                &Sigma_x, // size n OR empty
                    std::optional<spmat> JI,      // may be std::nullopt
                    std::optional<dvec>  Sigma_s, // may be std::nullopt
                    double               sigma_isotropic = 0.0) {
        auto compiled_hpv = CompiledWOp(f_hess_, Sigma_x, JI, Sigma_s, sigma_isotropic);
        hvp               = compiled_hpv;
    }

    // -------------------------------------------------------------------------
    void eval_all(const Eigen::Ref<const dvec> &x, std::optional<std::vector<std::string>> components = std::nullopt) {
        EvalEntry *e = cache_.find(x);
        if (!e) e = &cache_.insert(x);

        const std::vector<std::string> want = (components && !components->empty())
                                                  ? *components
                                                  : std::vector<std::string>{"f", "g", "cI", "JI", "cE", "JE"};

        auto wants = [&](const char *k) { return std::find(want.begin(), want.end(), k) != want.end(); };

        // f and g
        if ((wants("f") || wants("g")) && (!e->f || !e->g)) {
            auto [fv, fg] = f_grad_->value_grad_eigen(x);
            e->f          = fv;
            e->g          = fg;
        }

        // inequalities
        if ((wants("cI") || wants("JI")) && (!e->cI || !e->JI) && mI_ > 0) {
            auto [vals, J] = batch_value_grad_from_gradfns_sparse(cI_compiled_, x);
            e->cI          = vals;
            e->JI          = J;
        }

        // equalities
        if ((wants("cE") || wants("JE")) && (!e->cE || !e->JE) && mE_ > 0) {
            auto [vals, J] = batch_value_grad_from_gradfns_sparse(cE_compiled_, x);
            e->cE          = vals;
            e->JE          = J;
        }

        current_vals.x            = x;
        current_vals.hash         = e->hash;
        current_vals.access_order = e->access_order;
        if ((wants("f") || wants("g")) && e->f && e->g) {
            current_vals.f = e->f;
            current_vals.g = e->g;
        }
        if ((wants("cI") || wants("JI")) && mI_ > 0 && e->cI && e->JI) {
            current_vals.cI = e->cI;
            current_vals.JI = e->JI;
        }
        if ((wants("cE") || wants("JE")) && mE_ > 0 && e->cE && e->JE) {
            current_vals.cE = e->cE;
            current_vals.JE = e->JE;
        }
    }

    // -------------------------------------------------------------------------
    double constraint_violation(const Eigen::Ref<const dvec> &x) {
        eval_all(x, std::vector<std::string>{"cI", "cE"});
        dvec cI_v, cE_v;
        if (mI_) cI_v = current_vals.cI.value();
        if (mE_) cE_v = current_vals.cE.value();

        const double scale = std::max<double>(1.0, std::max<double>(n_, mI_ + mE_));
        double       theta = 0.0;
        if (mI_) theta += (cI_v.array().max(0.0)).sum() / scale;
        if (mE_) theta += cE_v.array().abs().sum() / scale;

        return std::isfinite(theta) ? theta : std::numeric_limits<double>::infinity();
    }

    // Call once per accepted step to update L-BFGS memory (no materialization)
    void lbfgs_update(const Eigen::Ref<const dvec> &x, const Eigen::Ref<const dvec> &lam,
                      const Eigen::Ref<const dvec> &nu) {
        // Ensure gradient of Lagrangian is available
        eval_all(x, std::vector<std::string>{"g", (mI_ ? "JI" : "none"), (mE_ ? "JE" : "none")});

        dvec gradL = current_vals.g.value();
        if (mI_) gradL.noalias() += current_vals.JI.value().transpose() * lam;
        if (mE_) gradL.noalias() += current_vals.JE.value().transpose() * nu;

        if (last_x_.size() == x.size() && last_gradL_.size() == gradL.size()) {
            lbfgs_.update(last_x_, last_gradL_, x, gradL);
        }
        last_x_     = x;
        last_gradL_ = std::move(gradL);
    }

    // Optional: clear memory between phases
    void lbfgs_reset() {
        lbfgs_.reset(n_);
        last_x_.resize(0);
        last_gradL_.resize(0);
    }

private:
    int  n_{0}, mI_{0}, mE_{0};
    bool use_sparse_{false};
    dvec lb_, ub_;

    mutable EvalCache cache_; // global per-model multipoint cache

    // L-BFGS state
    bool                         use_lbfgs_hess_{false};
    mutable detail::LBFGSLagHess lbfgs_;
    mutable dvec                 last_x_;
    mutable dvec                 last_gradL_;

    // Compilation helpers
    std::shared_ptr<LagHessFn> compile_lag_hess_(const nb::object                &f,
                                                 const std::optional<nb::object> &c_ineq = nb::none(),
                                                 const std::optional<nb::object> &c_eq   = nb::none()) {
        auto ineq_list =
            (c_ineq && !c_ineq->is_none()) ? nb::cast<std::vector<nb::object>>(*c_ineq) : std::vector<nb::object>{};
        auto eq_list =
            (c_eq && !c_eq->is_none()) ? nb::cast<std::vector<nb::object>>(*c_eq) : std::vector<nb::object>{};
        return std::make_shared<LagHessFn>(f, ineq_list, eq_list, (size_t)n_, true);
    }

    std::shared_ptr<GradFn> compile_objective_(const nb::object f) {
        return std::make_shared<GradFn>(f, (size_t)n_, true);
    }

    std::shared_ptr<GradFn> compile_constraint_(const nb::object &c, bool vector_input = true) {
        return std::make_shared<GradFn>(c, (size_t)n_, vector_input);
    }

public:
    ModelSuspectReport run_convexity_analysis(bool print_summary = false) const;
};

inline ModelSuspectReport ModelC::run_convexity_analysis(bool print_summary) const {
    using namespace suspect_adapter;
    using Traits = suspect::SuspectTraits<ModelView, ADNodePtr>;

    ModelView MV;

    // Bounds and binary flags from ModelC
    const int n = n_;
    MV.lb.resize(n);
    MV.ub.resize(n);
    MV.is_binary.assign(n, false); // set to true where applicable in your model
    for (int i = 0; i < n; ++i) {
        MV.lb[i] = lb_(i);
        MV.ub[i] = ub_(i);
    }
    MV.obj_sense = +1; // or set -1 if this model is maximize

    // Objective (if present)
    if (f_grad_) {
        ExprView ev = make_view(*f_grad_);
        if (ev.ok()) {
            // Stamp var metadata into that expression's var nodes
            stamp_var_metadata(ev.vars, lb_, ub_, MV.is_binary);
            MV.objective = ev;
            collect_closure(ev.root, MV.all_nodes);
        }
    }

    // Inequalities: each cI_compiled_[i] is g_i(x) ≤ 0
    for (auto &gi : cI_compiled_) {
        ExprView ev = make_view(*gi);
        if (!ev.ok()) continue;
        stamp_var_metadata(ev.vars, lb_, ub_, MV.is_binary);
        MV.constraints.emplace_back(ev, '<');
        collect_closure(ev.root, MV.all_nodes);
    }

    // Equalities: each cE_compiled_[i] is h_i(x) = 0
    for (auto &hi : cE_compiled_) {
        ExprView ev = make_view(*hi);
        if (!ev.ok()) continue;
        stamp_var_metadata(ev.vars, lb_, ub_, MV.is_binary);
        MV.constraints.emplace_back(ev, '=');
        collect_closure(ev.root, MV.all_nodes);
    }

    // DEBUG: Check what operators are in the constraint expressions
    // std::cout << "\n=== DEBUG: Constraint Analysis ===\n";
    // for (size_t i = 0; i < MV.constraints.size(); ++i) {
    //     const auto &con = MV.constraints[i];
    //     std::cout << "Constraint " << i << " (sense '" << con.second << "'):\n";
    //     std::cout << "  Root node type: " << static_cast<int>(con.first.root->type) << "\n";
    //     std::cout << "  Root is const: " << con.first.root->isConstant() << "\n";
    //     std::cout << "  Root is var: " << con.first.root->isVariable() << "\n";
    //     std::cout << "  # inputs: " << con.first.root->inputs.size() << "\n";

    //     // Check if root is in all_nodes
    //     bool found = false;
    //     for (const auto &n : MV.all_nodes) {
    //         if (n.get() == con.first.root.get()) {
    //             found = true;
    //             break;
    //         }
    //     }
    //     std::cout << "  Root in all_nodes: " << found << "\n";
    // }
    // std::cout << "=== END DEBUG ===\n\n";

    // Deduplicate MV.all_nodes while preserving order
    // {
    //     std::vector<ADNodePtr> uniq;
    //     uniq.reserve(MV.all_nodes.size());
    //     std::unordered_set<const ADNode *> seen;
    //     for (auto &nd : MV.all_nodes)
    //         if (seen.insert(nd.get()).second) uniq.push_back(nd);
    //     MV.all_nodes.swap(uniq);
    // }

    // Run the analyzer
    suspect::Analyzer<ModelView, ADNodePtr> A(MV);
    const auto                             &info = A.analyze(/*max_iter=*/10, /*tol=*/1e-12);

    ModelSuspectReport R;
    R.summary   = A.buildSummary(print_summary);
    R.node_info = info; // keep if you want node-level diagnostics
    return R;
}
