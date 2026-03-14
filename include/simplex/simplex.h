#pragma once

// -----------------------------------------------------------------------------
// Revised Simplex (header-only, drop-in compatible) — now with Dual Simplex
// Public API preserved: LPSolution, to_string, RevisedSimplexOptions,
//                       RevisedSimplex{ ctor, solve(...) }.
// Internals tidied without behavioral changes, plus a dual simplex phase:
//   - Options::mode = {Auto, Primal, Dual}
//   - Auto tries primal, and if primal reports negative basic variables,
//     falls back to dual before Phase I.
//   - You can force Dual by setting options.mode = SimplexMode::Dual.
// -----------------------------------------------------------------------------

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/Sparse>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "presolver.h"    // presolve::LP, Presolver
#include "simplex_aux.h"  // FTBasis helpers declared by your project
#include "simplex_lu.h"  // FTBasis implementation (solve_B, solve_BT, replace_column, refactor)

// ============================================================================
// Public result container
// ============================================================================
struct LPSolution {
    enum class Status {
        Optimal,
        Unbounded,
        Infeasible,
        IterLimit,
        Singular,
        NeedPhase1
    };

    Status status{};
    Eigen::VectorXd x;  // primal solution (original space)
    double obj = std::numeric_limits<double>::quiet_NaN();
    std::vector<int> basis;  // basis indices in original problem
    int iters = 0;           // total iterations (Phase I + II)
    std::unordered_map<std::string, std::string> info;  // telemetry
    Eigen::VectorXd farkas_y;  // Farkas certificate of infeasibility (if any)
    bool farkas_has_cert = false;  // whether farkas_y is valid
};

inline const char* to_string(LPSolution::Status s) {
    switch (s) {
        case LPSolution::Status::Optimal:
            return "optimal";
        case LPSolution::Status::Unbounded:
            return "unbounded";
        case LPSolution::Status::Infeasible:
            return "infeasible";
        case LPSolution::Status::IterLimit:
            return "iterlimit";
        case LPSolution::Status::Singular:
            return "singular";
        case LPSolution::Status::NeedPhase1:
            return "need_phase1";
    }
    return "unknown";
}

// ============================================================================
// Options
// ============================================================================
enum class SimplexMode { Auto, Primal, Dual };

struct RevisedSimplexOptions {
    // Global
    int max_iters = 50'000;
    double tol = 1e-9;
    bool bland = false;
    double svd_tol = 1e-8;
    double ratio_delta = 1e-12;
    double ratio_eta = 1e-7;
    double deg_step_tol = 1e-12;
    double epsilon_cost = 1e-10;
    int rng_seed = 13;

    // Basis / LU
    int refactor_every = 128;  // FT hard cap
    int compress_every = 64;   // FT soft cap
    double lu_pivot_rel = 1e-12;
    double lu_abs_floor = 1e-16;
    double alpha_tol = 1e-10;
    double z_inf_guard = 1e6;
    std::string basis_update = "forrest_tomlin";  // or "eta"
    int ft_bandwidth_cap = 16;

    // Pricing
    int devex_reset = 200;
    std::string pricing_rule = "adaptive";  // or "devex" / "most_negative"
    int adaptive_reset_freq = 1000;

    // Recovery
    int max_basis_rebuilds = 3;

    // Algorithm selection/tuning (append to your options)
    bool dual_allow_bound_flip = true;   // enable Beale bound-flipping
    double dual_flip_pivot_tol = 1e-10;  // |pN(e)| below this ⇒ consider flip
    double dual_flip_rc_tol = 1e-10;     // |rN(e)| “near dual-feasible”
    int dual_flip_max_per_iter = 2;      // avoid pathological flip storms

    // Algorithm selection
    SimplexMode mode = SimplexMode::Auto;  // Auto | Primal | Dual
};

// Forward decls from degeneracy/pricer header (kept external)
static inline std::unordered_map<std::string, std::string> dm_stats_to_map(
    const DegeneracyManager::Stats& s) {
    std::unordered_map<std::string, std::string> info;
    info["deg_streak"] = std::to_string(s.degeneracy_streak);
    info["deg_total"] = std::to_string(s.degeneracy_total);
    info["cycle_len"] = std::to_string(s.suspected_cycling);
    info["cond_est"] = std::to_string(s.cond_est);
    info["deg_thresh"] = std::to_string(s.adaptive_deg_threshold);
    info["deg_epoch"] = std::to_string(s.epoch);
    return info;
}

// ============================================================================
// RevisedSimplex
// ============================================================================
class RevisedSimplex {
   public:
    explicit RevisedSimplex(RevisedSimplexOptions opt = {})
        : opt_(std::move(opt)),
          rng_(opt_.rng_seed),
          degen_(opt_.rng_seed),
          adaptive_pricer_(1)  // initialized to a dummy size; rebuilt per solve
    {}

    // Main entry (drop-in compatible)
    LPSolution solve(const Eigen::MatrixXd& A_in, const Eigen::VectorXd& b_in,
                     const Eigen::VectorXd& c_in,
                     std::optional<std::vector<int>> basis_opt = std::nullopt) {
        const int n = static_cast<int>(A_in.cols());
        return solve(
            A_in, b_in, c_in, Eigen::VectorXd::Zero(n),
            Eigen::VectorXd::Constant(n, presolve::inf()), basis_opt);
    }

    LPSolution solve(const Eigen::MatrixXd& A_in, const Eigen::VectorXd& b_in,
                     const Eigen::VectorXd& c_in, const Eigen::VectorXd& l_in,
                     const Eigen::VectorXd& u_in,
                     std::optional<std::vector<int>> basis_opt = std::nullopt) {
        const int n = static_cast<int>(A_in.cols());
        if (b_in.size() != A_in.rows()) {
            throw std::invalid_argument("simplex: b size mismatch with rows(A)");
        }
        if (c_in.size() != n || l_in.size() != n || u_in.size() != n) {
            throw std::invalid_argument(
                "simplex: c/l/u sizes must equal cols(A)");
        }

        Eigen::MatrixXd A_model = A_in;
        Eigen::VectorXd b_model = b_in;
        Eigen::VectorXd c_model = c_in;
        Eigen::VectorXd l_model = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd u_model = Eigen::VectorXd::Constant(n, presolve::inf());
        Eigen::VectorXd anchor = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd sign = Eigen::VectorXd::Ones(n);

        for (int j = 0; j < n; ++j) {
            const bool has_l = std::isfinite(l_in(j));
            const bool has_u = std::isfinite(u_in(j));
            if (!has_l && !has_u) {
                throw std::invalid_argument(
                    "simplex: free variables are unsupported in solve(A,b,c,l,u)");
            }

            if (has_l) {
                anchor(j) = l_in(j);
                l_model(j) = 0.0;
                u_model(j) = has_u ? (u_in(j) - l_in(j)) : presolve::inf();
            } else {
                anchor(j) = u_in(j);
                sign(j) = -1.0;
                l_model(j) = 0.0;
                u_model(j) = presolve::inf();
                A_model.col(j) = -A_model.col(j);
                c_model(j) = -c_model(j);
            }

            if (anchor(j) != 0.0) b_model.noalias() -= A_model.col(j) * anchor(j);
        }

        // ---- (0) Wrap into presolve LP: Ax=b, default bounds, costs=c ----
        presolve::LP lp;
        lp.A = A_model;
        lp.b = b_model;
        lp.sense.assign(static_cast<int>(A_in.rows()), presolve::RowSense::EQ);
        lp.c = c_model;
        lp.l = l_model;
        lp.u = u_model;
        lp.c0 = c_in.dot(anchor);

        // ---- (1) Presolve ----
        presolve::Presolver::Options popt;
        popt.enable_rowreduce = true;
        popt.enable_scaling = true;
        popt.max_passes = 10;
        if (A_in.cols() > static_cast<int>(A_in.rows() * 1.2)) {
            popt.conservative_mode = true;
        }

        presolve::Presolver P(popt);
        const auto pres = P.run(lp);

        if (pres.proven_infeasible) {
            return make_solution_(LPSolution::Status::Infeasible,
                                  Eigen::VectorXd::Zero(n),
                                  std::numeric_limits<double>::infinity(), {},
                                  0, {{"presolve", "infeasible"}});
        }
        if (pres.proven_unbounded) {
            Eigen::VectorXd xnan = Eigen::VectorXd::Constant(
                n, std::numeric_limits<double>::quiet_NaN());
            return make_solution_(LPSolution::Status::Unbounded, xnan,
                                  -std::numeric_limits<double>::infinity(), {},
                                  0, {{"presolve", "unbounded"}});
        }

        const Eigen::MatrixXd& Atil = pres.reduced.A;
        const Eigen::VectorXd& btil = pres.reduced.b;
        const Eigen::VectorXd& ctil = pres.reduced.c;
        const Eigen::VectorXd& lred = pres.reduced.l;
        const Eigen::VectorXd& ured = pres.reduced.u;

        // ---- (2) m==0 fast path: optimize over bounds only ----
        if (Atil.rows() == 0) {
            Eigen::VectorXd vred =
                Eigen::VectorXd::Zero(static_cast<int>(ctil.size()));
            bool is_bounded = true;
            for (int j = 0; j < static_cast<int>(ctil.size()); ++j) {
                if (ctil(j) > opt_.tol) {
                    vred(j) = std::isfinite(lred(j)) ? lred(j) : 0.0;
                } else if (ctil(j) < -opt_.tol) {
                    if (std::isfinite(ured(j)))
                        vred(j) = ured(j);
                    else {
                        is_bounded = false;
                        break;
                    }
                } else {
                    vred(j) = std::isfinite(lred(j)) ? lred(j) : 0.0;
                }
            }
            if (!is_bounded) {
                Eigen::VectorXd xnan = Eigen::VectorXd::Constant(
                    n, std::numeric_limits<double>::quiet_NaN());
                return make_solution_(
                    LPSolution::Status::Unbounded, xnan,
                    -std::numeric_limits<double>::infinity(), {}, 0,
                    {{"presolve", "m=0 neg cost & +inf upper"}});
            }
            auto [z_full, obj_corr] = P.postsolve(vred);
            Eigen::VectorXd x_full = anchor + sign.cwiseProduct(z_full);
            const double total_obj = c_in.dot(x_full) + obj_corr;
            return make_solution_(LPSolution::Status::Optimal, std::move(x_full),
                                  total_obj, {}, 0,
                                  {{"presolve", "m=0 optimized over bounds"}});
        }

        // ---- (3) Solve reduced problem directly with explicit bounds ----
        Eigen::MatrixXd Ared = Atil;
        Eigen::VectorXd bred = btil;
        Eigen::VectorXd cred = ctil;
        std::vector<int> col_orig_map = pres.orig_col_index;

        const int m_eff = static_cast<int>(Ared.rows());
        const int n_eff = static_cast<int>(Ared.cols());

        // Effective bounds (reduced space)
        Eigen::VectorXd l_eff = lred;
        Eigen::VectorXd u_eff = ured;

        // ---- (4) Map incoming basis into reduced space (optional) ----
        std::optional<std::vector<int>> red_basis_opt = std::nullopt;
        if (basis_opt && !basis_opt->empty()) {
            std::unordered_map<int, int> orig2red;
            orig2red.reserve(n_eff);
            for (int jr = 0; jr < n_eff; ++jr) {
                const int jorig = col_orig_map[jr];
                if (jorig >= 0) orig2red[jorig] = jr;
            }
            std::vector<int> cand;
            cand.reserve(std::min(m_eff, (int)basis_opt->size()));
            for (int jorig : *basis_opt) {
                auto it = orig2red.find(jorig);
                if (it != orig2red.end()) {
                    cand.push_back(it->second);
                    if ((int)cand.size() == m_eff) break;
                }
            }
            if ((int)cand.size() == m_eff) red_basis_opt = std::move(cand);
        }

        // ---- (5) Try Phase II directly on reduced problem (Primal/Dual per
        // mode) ----
        std::vector<int> basis_guess;
        bool basis_guess_from_crash = false;
        if (red_basis_opt && (int)red_basis_opt->size() == m_eff) {
            basis_guess = *red_basis_opt;
        } else {
            if (auto maybe = find_initial_basis_(Ared, bred, cred)) {
                basis_guess = *maybe;
                basis_guess_from_crash = true;
            }
        }

        const auto add_info =
            [&](std::unordered_map<std::string, std::string> info) {
                info["presolve_actions"] = std::to_string(pres.stack.size());
                info["reduced_m"] = std::to_string(m_eff);
                info["reduced_n"] = std::to_string(n_eff);
                info["obj_shift"] = std::to_string(pres.obj_shift);
                return info;
            };

        const bool crash_primal_feasible =
            basis_guess_from_crash &&
            basis_is_primal_feasible_(Ared, bred, basis_guess, opt_.tol);
        const bool allow_direct_from_guess =
            ((int)basis_guess.size() == m_eff) &&
            (!basis_guess_from_crash || crash_primal_feasible);

        if (allow_direct_from_guess) {
            LPSolution::Status st;
            Eigen::VectorXd v2;
            std::vector<int> red_basis2;
            int it2;
            std::unordered_map<std::string, std::string> info2;

            auto run_primal = [&] {
                return phase_(Ared, bred, cred, basis_guess, l_eff, u_eff);
            };
            auto run_dual = [&] {
                return dual_phase_(Ared, bred, cred, basis_guess, l_eff, u_eff);
            };

            if (opt_.mode == SimplexMode::Dual) {
                if (basis_guess_from_crash) {
                    st = LPSolution::Status::NeedPhase1;
                    info2["reason"] = "crash_basis_dual_disabled";
                } else {
                    std::tie(st, v2, red_basis2, it2, info2) = run_dual();
                }
            } else if (opt_.mode == SimplexMode::Primal) {
                std::tie(st, v2, red_basis2, it2, info2) = run_primal();
            } else {
                // Auto: primal first; if primal reports negative basics → dual
                std::tie(st, v2, red_basis2, it2, info2) = run_primal();
                if (!basis_guess_from_crash &&
                    st == LPSolution::Status::NeedPhase1 &&
                    info2.count("reason") &&
                    info2.at("reason") == std::string("negative_basic_vars")) {
                    std::tie(st, v2, red_basis2, it2, info2) = run_dual();
                }
            }

            if (st == LPSolution::Status::Optimal ||
                st == LPSolution::Status::Unbounded ||
                st == LPSolution::Status::IterLimit) {
                auto [z_full, obj_corr] = P.postsolve(v2);
                Eigen::VectorXd x_full = anchor + sign.cwiseProduct(z_full);
                const double total_obj = c_in.dot(x_full) + obj_corr;

                std::vector<int> basis_full;
                basis_full.reserve(red_basis2.size());
                for (int jr : red_basis2) {
                    if (jr >= 0 && jr < (int)col_orig_map.size()) {
                        const int jorig = col_orig_map[jr];
                        if (jorig >= 0) basis_full.push_back(jorig);
                    }
                }
                auto info = add_info(std::move(info2));
                return make_solution_(st, std::move(x_full), total_obj,
                                      basis_full, it2, std::move(info));
            }
            if (st == LPSolution::Status::Singular) {
                auto info = add_info({});
                return make_solution_(LPSolution::Status::Singular,
                                      Eigen::VectorXd::Zero(n),
                                      std::numeric_limits<double>::quiet_NaN(),
                                      {}, 0, std::move(info));
            }
        }

        // ---- (6) Phase I on reduced problem ----
        auto [A1, b1, c1, basis1, n_orig_eff, m_rows] =
            make_phase1_(Ared, bred);
        auto [status1, v1, basis1_out, it1, info1] =
            phase_(A1, b1, c1, basis1, Eigen::VectorXd::Zero(A1.cols()),
                   Eigen::VectorXd::Constant(A1.cols(), presolve::inf()));

        // If phase I fails or artificial cost > tol ⇒ infeasible
        if (status1 != LPSolution::Status::Optimal || c1.dot(v1) > opt_.tol) {
            auto info = add_info({{"phase1_status", to_string(status1)}});
            const auto s = degen_.get_stats();
            auto more = dm_stats_to_map(s);
            info.insert(more.begin(), more.end());
            return make_solution_(LPSolution::Status::Infeasible,
                                  Eigen::VectorXd::Zero(n),
                                  std::numeric_limits<double>::infinity(), {},
                                  it1, std::move(info));
        }

        // Warm-start Phase II basis by removing artificials
        std::vector<int> red_basis2;
        red_basis2.reserve(m_rows);
        for (int j : basis1_out)
            if (j < (int)n_orig_eff) red_basis2.push_back(j);

        // Basis completion if needed
        if ((int)red_basis2.size() < m_rows) {
            for (int j = 0; j < (int)n_orig_eff; ++j) {
                if ((int)red_basis2.size() == m_rows) break;
                if (std::find(red_basis2.begin(), red_basis2.end(), j) !=
                    red_basis2.end())
                    continue;
                std::vector<int> cand = red_basis2;
                cand.push_back(j);
                if ((int)cand.size() > m_rows) continue;
                const Eigen::MatrixXd Btest =
                    Ared(Eigen::all,
                         Eigen::VectorXi::Map(cand.data(), (int)cand.size()));
                Eigen::FullPivLU<Eigen::MatrixXd> lu(Btest);
                if (lu.rank() == (int)cand.size() && lu.isInvertible())
                    red_basis2 = std::move(cand);
            }
        }

        // Final Phase II on reduced problem (respect mode)
        LPSolution::Status status2;
        Eigen::VectorXd v2;
        std::vector<int> red_basis_out;
        int it2 = 0;
        std::unordered_map<std::string, std::string> info2;

        if ((int)red_basis2.size() == m_rows) {
            if (opt_.mode == SimplexMode::Dual) {
                std::tie(status2, v2, red_basis_out, it2, info2) =
                    dual_phase_(Ared, bred, cred, red_basis2, l_eff, u_eff);
                if (status2 == LPSolution::Status::Infeasible) {
                    auto it = info2.find("farkas_has_cert");
                    if (it != info2.end() && it->second == "1") {
                        // parse CSV into a vector
                        Eigen::VectorXd yF(m_eff);
                        {
                            std::vector<double> vals;
                            vals.reserve(m_eff);
                            std::stringstream ss(info2["farkas_y"]);
                            std::string tok;
                            while (std::getline(ss, tok, ','))
                                vals.push_back(std::stod(tok));
                            yF = Eigen::Map<const Eigen::VectorXd>(
                                vals.data(), (int)vals.size());
                        }
                        return make_solution_(
                            LPSolution::Status::Infeasible,
                            Eigen::VectorXd::Zero(n),
                            std::numeric_limits<double>::infinity(), {}, it2,
                            add_info(std::move(info2)), yF, true);
                    }
                }

            } else if (opt_.mode == SimplexMode::Primal) {
                std::tie(status2, v2, red_basis_out, it2, info2) =
                    phase_(Ared, bred, cred, red_basis2, l_eff, u_eff);
            } else {
                // Auto: primal first; if negative basics → dual
                std::tie(status2, v2, red_basis_out, it2, info2) =
                    phase_(Ared, bred, cred, red_basis2, l_eff, u_eff);
                if (status2 == LPSolution::Status::NeedPhase1 &&
                    info2.count("reason") &&
                    info2.at("reason") == std::string("negative_basic_vars")) {
                    std::tie(status2, v2, red_basis_out, it2, info2) =
                        dual_phase_(Ared, bred, cred, red_basis2, l_eff, u_eff);
                }
            }
        } else {
            // Fall back to find a basis internally
            std::tie(status2, v2, red_basis_out, it2, info2) =
                phase_(Ared, bred, cred, std::nullopt, l_eff, u_eff);
            if (status2 == LPSolution::Status::NeedPhase1) {
                status2 = LPSolution::Status::Singular;
                info2["note"] = "reduced matrix cannot form a proper basis";
            }
        }

        const int total_iters = it1 + it2;
        auto merged_info = add_info(std::move(info2));
        merged_info.insert({"phase1_iters", std::to_string(it1)});

        auto [z_full, obj_correction] = P.postsolve(v2);
        Eigen::VectorXd x_full = anchor + sign.cwiseProduct(z_full);
        const double total_obj = c_in.dot(x_full) + obj_correction;

        std::vector<int> basis_full;
        basis_full.reserve(red_basis_out.size());
        for (int jr : red_basis_out) {
            if (jr >= 0 && jr < (int)col_orig_map.size()) {
                const int jorig = col_orig_map[jr];
                if (jorig >= 0) basis_full.push_back(jorig);
            }
        }

        if (status2 == LPSolution::Status::Optimal) {
            return make_solution_(LPSolution::Status::Optimal, x_full,
                                  total_obj, basis_full, total_iters,
                                  std::move(merged_info));
        }
        if (status2 == LPSolution::Status::Unbounded) {
            return make_solution_(LPSolution::Status::Unbounded, x_full,
                                  -std::numeric_limits<double>::infinity(),
                                  basis_full, total_iters,
                                  std::move(merged_info));
        }

        const double obj_fallback =
            x_full.array().isFinite().all()
                ? total_obj
                : std::numeric_limits<double>::quiet_NaN();
        return make_solution_(status2, x_full, obj_fallback, basis_full,
                              total_iters, std::move(merged_info));
    }

   private:
    // =========================================================================
    // Helpers (private; signatures preserved where externally referenced)
    // =========================================================================

    static Eigen::VectorXd clip_small_(Eigen::VectorXd x, double tol = 1e-12) {
        for (int i = 0; i < x.size(); ++i)
            if (std::abs(x(i)) < tol) x(i) = 0.0;
        return x;
    }

    FTBasis::Options make_basis_options_() const {
        FTBasis::Options bopt;
        bopt.refactor_every = opt_.refactor_every;
        bopt.compress_every = opt_.compress_every;
        bopt.pivot_rel = opt_.lu_pivot_rel;
        bopt.abs_floor = opt_.lu_abs_floor;
        bopt.alpha_tol = opt_.alpha_tol;
        bopt.z_inf_guard = opt_.z_inf_guard;
        bopt.ft_bandwidth_cap = opt_.ft_bandwidth_cap;

        std::string mode = opt_.basis_update;
        std::transform(mode.begin(), mode.end(), mode.begin(),
                       [](unsigned char ch) {
                           return static_cast<char>(std::tolower(ch));
                       });
        if (mode == "eta" || mode == "eta_stack") {
            bopt.update_mode = FTBasis::Options::UpdateMode::EtaStack;
        } else {
            bopt.update_mode = FTBasis::Options::UpdateMode::ForrestTomlin;
        }

        return bopt;
    }

    static Eigen::VectorXd assemble_primal_(int n,
                                            const std::vector<int>& basis,
                                            const Eigen::VectorXd& xB,
                                            const Eigen::VectorXd& l,
                                            const Eigen::VectorXd& u,
                                            const std::vector<int>* sigma =
                                                nullptr) {
        Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
        std::vector<char> inB(n, 0);
        for (int i = 0; i < (int)basis.size(); ++i) {
            const int j = basis[i];
            if (j >= 0 && j < n) {
                inB[j] = 1;
                if (i < xB.size()) x(j) = xB(i);
            }
        }

        for (int j = 0; j < n; ++j) {
            if (inB[j]) continue;

            const bool upper_view =
                sigma && j < (int)sigma->size() && (*sigma)[j] < 0;
            if (upper_view && j < u.size() && std::isfinite(u(j))) {
                x(j) = u(j);
            } else if (j < l.size() && std::isfinite(l(j))) {
                x(j) = l(j);
            } else {
                x(j) = 0.0;
            }
        }

        return clip_small_(x);
    }

    enum class BoundView { Lower, Upper, Fixed };

    static BoundView default_bound_view_(int j, const Eigen::VectorXd& l,
                                         const Eigen::VectorXd& u) {
        const bool has_l = (j < l.size()) && std::isfinite(l(j));
        const bool has_u = (j < u.size()) && std::isfinite(u(j));
        if (has_l && has_u && std::abs(u(j) - l(j)) <= 1e-12) {
            return BoundView::Fixed;
        }
        if (has_u && !has_l) return BoundView::Upper;
        return BoundView::Lower;
    }

    static double bound_anchor_(BoundView view, int j, const Eigen::VectorXd& l,
                                const Eigen::VectorXd& u) {
        switch (view) {
            case BoundView::Upper:
                return (j < u.size() && std::isfinite(u(j))) ? u(j) : 0.0;
            case BoundView::Fixed:
            case BoundView::Lower:
            default:
                return (j < l.size() && std::isfinite(l(j))) ? l(j) : 0.0;
        }
    }

    static int view_sign_(BoundView view) {
        return (view == BoundView::Upper) ? -1 : 1;
    }

    static double bound_range_(int j, const Eigen::VectorXd& l,
                               const Eigen::VectorXd& u) {
        if (j >= l.size() || j >= u.size() || !std::isfinite(l(j)) ||
            !std::isfinite(u(j))) {
            return std::numeric_limits<double>::infinity();
        }
        return std::max(0.0, u(j) - l(j));
    }

    static Eigen::VectorXd transformed_rhs_(const Eigen::MatrixXd& A,
                                            const std::vector<BoundView>& view,
                                            const Eigen::VectorXd& l,
                                            const Eigen::VectorXd& u) {
        Eigen::VectorXd rhs = A.rows() ? Eigen::VectorXd::Zero(A.rows())
                                       : Eigen::VectorXd{};
        for (int j = 0; j < A.cols(); ++j) {
            const double anchor = bound_anchor_(view[j], j, l, u);
            if (anchor != 0.0) rhs.noalias() += A.col(j) * anchor;
        }
        return rhs;
    }

    static Eigen::VectorXd effective_costs_(const Eigen::VectorXd& c,
                                            const std::vector<BoundView>& view) {
        Eigen::VectorXd chat = c;
        for (int j = 0; j < chat.size() && j < (int)view.size(); ++j) {
            chat(j) *= static_cast<double>(view_sign_(view[j]));
        }
        return chat;
    }

    static Eigen::VectorXd assemble_transformed_primal_(
        int n, const std::vector<int>& basis, const Eigen::VectorXd& yB,
        const Eigen::VectorXd& l, const Eigen::VectorXd& u,
        const std::vector<BoundView>& view) {
        Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
        std::vector<char> inB(n, 0);
        for (int i = 0; i < (int)basis.size(); ++i) {
            const int j = basis[i];
            if (j < 0 || j >= n) continue;
            inB[j] = 1;
            const double anchor = bound_anchor_(view[j], j, l, u);
            if (view_sign_(view[j]) > 0) {
                x(j) = anchor + ((i < yB.size()) ? yB(i) : 0.0);
            } else {
                x(j) = anchor - ((i < yB.size()) ? yB(i) : 0.0);
            }
        }

        for (int j = 0; j < n; ++j) {
            if (!inB[j]) x(j) = bound_anchor_(view[j], j, l, u);
        }
        return clip_small_(x);
    }

    struct CrashCandidate {
        int col = -1;
        int pivot_row = -1;
        double score = -std::numeric_limits<double>::infinity();
    };

    static CrashCandidate choose_slack_like_column_(
        const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
        const Eigen::VectorXd& c, const std::vector<char>& used_row,
        const std::vector<char>& used_col) {
        CrashCandidate best;
        const int m = static_cast<int>(A.rows());
        const int n = static_cast<int>(A.cols());
        double c_scale = 1.0;
        if (c.size() > 0) c_scale = std::max(1.0, c.cwiseAbs().maxCoeff());

        for (int j = 0; j < n; ++j) {
            if (used_col[j]) continue;
            int pivot_row = -1;
            int nnz = 0;
            double pivot = 0.0;
            double off_sum = 0.0;
            for (int i = 0; i < m; ++i) {
                const double aij = A(i, j);
                if (std::abs(aij) <= 1e-12) continue;
                ++nnz;
                if (used_row[i]) {
                    off_sum += std::abs(aij);
                    continue;
                }
                if (std::abs(aij) > std::abs(pivot)) {
                    if (pivot_row >= 0) off_sum += std::abs(pivot);
                    pivot_row = i;
                    pivot = aij;
                } else {
                    off_sum += std::abs(aij);
                }
            }
            if (pivot_row < 0 || std::abs(pivot) <= 1e-12) continue;

            const bool exact_unit =
                (nnz == 1 && std::abs(std::abs(pivot) - 1.0) <= 1e-10);
            const bool slack_like = (nnz == 1) || (off_sum <= 1e-10);
            if (!slack_like) continue;

            double score = exact_unit ? 1e6 : 1e5;
            score += 1e3 / (1.0 + off_sum);
            score += 10.0 / (1.0 + std::abs(std::abs(pivot) - 1.0));
            if (b(pivot_row) >= -1e-10) score += 50.0;
            score -= std::abs(c(j)) / c_scale;
            score -= 0.01 * static_cast<double>(j);

            if (score > best.score) {
                best = {j, pivot_row, score};
            }
        }
        return best;
    }

    static CrashCandidate choose_triangular_column_(
        const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
        const Eigen::VectorXd& c, const std::vector<char>& used_row,
        const std::vector<char>& used_col) {
        CrashCandidate best;
        const int m = static_cast<int>(A.rows());
        const int n = static_cast<int>(A.cols());
        double c_scale = 1.0;
        if (c.size() > 0) c_scale = std::max(1.0, c.cwiseAbs().maxCoeff());

        for (int j = 0; j < n; ++j) {
            if (used_col[j]) continue;

            int pivot_row = -1;
            double pivot_abs = 0.0;
            double uncovered_sum = 0.0;
            double covered_sum = 0.0;
            int uncovered_nnz = 0;
            int total_nnz = 0;
            for (int i = 0; i < m; ++i) {
                const double aij = A(i, j);
                const double aa = std::abs(aij);
                if (aa <= 1e-12) continue;
                ++total_nnz;
                if (used_row[i]) {
                    covered_sum += aa;
                    continue;
                }
                ++uncovered_nnz;
                uncovered_sum += aa;
                if (aa > pivot_abs) {
                    pivot_abs = aa;
                    pivot_row = i;
                }
            }
            if (pivot_row < 0 || pivot_abs <= 1e-12) continue;

            const double dominance = pivot_abs / std::max(1e-12, uncovered_sum);
            const double triangularity =
                pivot_abs / std::max(1e-12, covered_sum + uncovered_sum);
            const double sparsity_bonus = 1.0 / (1.0 + total_nnz);
            const double rhs_bonus = (b(pivot_row) >= -1e-10) ? 0.25 : 0.0;
            const double cost_penalty = 0.05 * (std::abs(c(j)) / c_scale);

            const double score = 100.0 * dominance + 25.0 * triangularity +
                                 10.0 * sparsity_bonus + rhs_bonus -
                                 cost_penalty - 0.001 * static_cast<double>(j);
            if (score > best.score) {
                best = {j, pivot_row, score};
            }
        }
        return best;
    }

    static std::optional<std::vector<int>> find_initial_basis_(
        const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
        const Eigen::VectorXd& c) {
        const int m = static_cast<int>(A.rows());
        const int n = static_cast<int>(A.cols());
        if (m == 0) return std::vector<int>{};
        if (n < m) return std::nullopt;

        std::vector<int> basis;
        basis.reserve(m);
        std::vector<char> used_row(m, 0), used_col(n, 0);

        while ((int)basis.size() < m) {
            const CrashCandidate cand =
                choose_slack_like_column_(A, b, c, used_row, used_col);
            if (cand.col < 0) break;
            basis.push_back(cand.col);
            used_row[cand.pivot_row] = 1;
            used_col[cand.col] = 1;
        }

        while ((int)basis.size() < m) {
            const CrashCandidate cand =
                choose_triangular_column_(A, b, c, used_row, used_col);
            if (cand.col < 0) break;
            basis.push_back(cand.col);
            used_row[cand.pivot_row] = 1;
            used_col[cand.col] = 1;
        }

        if ((int)basis.size() < m) {
            std::vector<int> ranked;
            ranked.reserve(n);
            for (int j = 0; j < n; ++j)
                if (!used_col[j]) ranked.push_back(j);
            std::sort(ranked.begin(), ranked.end(), [&](int a, int b_idx) {
                const double nnz_a =
                    (A.col(a).array().abs() > 1e-12).cast<double>().sum();
                const double nnz_b =
                    (A.col(b_idx).array().abs() > 1e-12).cast<double>().sum();
                const double score_a = nnz_a + 0.1 * std::abs(c(a));
                const double score_b = nnz_b + 0.1 * std::abs(c(b_idx));
                if (std::abs(score_a - score_b) > 1e-12)
                    return score_a < score_b;
                return a < b_idx;
            });

            Eigen::FullPivLU<Eigen::MatrixXd> current_lu;
            int current_rank = 0;
            if (!basis.empty()) {
                const Eigen::MatrixXd B0 =
                    A(Eigen::all, Eigen::VectorXi::Map(basis.data(),
                                                       (int)basis.size()));
                current_lu.compute(B0);
                current_rank = current_lu.rank();
            }

            for (int j : ranked) {
                if ((int)basis.size() == m) break;
                std::vector<int> cand = basis;
                cand.push_back(j);
                const Eigen::MatrixXd Bcand =
                    A(Eigen::all,
                      Eigen::VectorXi::Map(cand.data(), (int)cand.size()));
                Eigen::FullPivLU<Eigen::MatrixXd> lu(Bcand);
                const int rank = lu.rank();
                if (rank > current_rank) {
                    basis.push_back(j);
                    used_col[j] = 1;
                    current_rank = rank;
                }
            }
        }

        if ((int)basis.size() != m) return std::nullopt;

        const Eigen::MatrixXd B =
            A(Eigen::all, Eigen::VectorXi::Map(basis.data(), (int)basis.size()));
        Eigen::FullPivLU<Eigen::MatrixXd> lu(B);
        if (lu.rank() != m || !lu.isInvertible()) return std::nullopt;
        return basis;
    }

    static bool basis_is_primal_feasible_(const Eigen::MatrixXd& A,
                                          const Eigen::VectorXd& b,
                                          const std::vector<int>& basis,
                                          double tol) {
        const int m = static_cast<int>(A.rows());
        if ((int)basis.size() != m) return false;
        if (m == 0) return true;
        const Eigen::MatrixXd B =
            A(Eigen::all, Eigen::VectorXi::Map(basis.data(), m));
        Eigen::FullPivLU<Eigen::MatrixXd> lu(B);
        if (lu.rank() != m || !lu.isInvertible()) return false;
        const Eigen::VectorXd xB = lu.solve(b);
        return xB.allFinite() && (xB.array() >= -tol).all();
    }

    static std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd,
                      std::vector<int>, std::size_t, int>
    make_phase1_(const Eigen::MatrixXd& A, const Eigen::VectorXd& b) {
        const int m = static_cast<int>(A.rows());
        const int n = static_cast<int>(A.cols());

        Eigen::MatrixXd A1 = A;
        Eigen::VectorXd b1 = b;
        for (int i = 0; i < m; ++i)
            if (b1(i) < 0) {
                A1.row(i) *= -1.0;
                b1(i) *= -1.0;
            }

        Eigen::MatrixXd A_aux(m, n + m);
        A_aux.leftCols(n) = A1;
        A_aux.rightCols(m) = Eigen::MatrixXd::Identity(m, m);

        Eigen::VectorXd c_aux(n + m);
        c_aux.setZero();
        c_aux.tail(m).setOnes();

        std::vector<int> basis(m);
        std::iota(basis.begin(), basis.end(), n);

        return {A_aux, b1, c_aux, basis, static_cast<std::size_t>(n), m};
    }

    // Harris ratio (improved): returns (leaving_row, theta_B) for primal
    std::pair<std::optional<int>, double> harris_ratio_(
        const Eigen::VectorXd& xB, const Eigen::VectorXd& dB, double delta,
        double eta) const {
        std::vector<int> pos;
        pos.reserve(dB.size());
        for (int i = 0; i < dB.size(); ++i)
            if (dB(i) > delta) pos.push_back(i);
        if (pos.empty())
            return {std::nullopt, std::numeric_limits<double>::infinity()};

        double theta_star = std::numeric_limits<double>::infinity();
        for (int idx : pos)
            theta_star = std::min(theta_star, xB(idx) / dB(idx));

        double max_resid = 0.0;
        std::vector<int> candidates;
        for (int idx : pos) {
            const double ratio = xB(idx) / dB(idx);
            if (std::abs(ratio - theta_star) <= 1e-10)
                candidates.push_back(idx);
            const double resid = xB(idx) - theta_star * dB(idx);
            max_resid = std::max(max_resid, std::max(0.0, resid));
        }
        if (!candidates.empty()) {
            int best = candidates.front();
            for (int idx : candidates)
                if (idx < best) best = idx;
            return {best, theta_star};
        }

        const double kappa = std::max(eta, eta * max_resid);
        std::vector<int> eligible;
        for (int idx : pos) {
            const double resid = xB(idx) - theta_star * dB(idx);
            if (resid <= kappa) eligible.push_back(idx);
        }
        if (!eligible.empty()) {
            int best = eligible.front();
            for (int idx : eligible)
                if (idx < best) best = idx;
            return {best, theta_star};
        }

        int best = pos.front();
        double best_ratio = xB(best) / dB(best);
        for (int i = 1; i < (int)pos.size(); ++i) {
            const int idx = pos[i];
            const double r = xB(idx) / dB(idx);
            if (r < best_ratio) {
                best_ratio = r;
                best = idx;
            }
        }
        return {best, best_ratio};
    }

    // BFRT: permissible step for entering variable to nearest active bound
    // (primal)
    struct BFRTStep {
        double theta_e = std::numeric_limits<double>::infinity();
        bool to_upper = false;
    };

    BFRTStep entering_bound_step_(double x_e, double l_e, double u_e,
                                  double rc_e, double tol) const {
        BFRTStep out;
        // Primal simplex (minimization): rc_e < 0 ⇒ objective improves as x_e
        // increases
        if (rc_e < -tol) {
            if (std::isfinite(u_e)) {
                out.theta_e = std::max(0.0, u_e - x_e);
                out.to_upper = true;
            }
        } else if (rc_e > tol) {  // move downward
            if (std::isfinite(l_e)) {
                out.theta_e = std::max(0.0, x_e - l_e);
                out.to_upper = false;
            }
        }
        return out;
    }

    // --------------------------- PRIMAL PHASE ---------------------------
    std::tuple<LPSolution::Status, Eigen::VectorXd, std::vector<int>, int,
               std::unordered_map<std::string, std::string>>
    phase_(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
           const Eigen::VectorXd& c, std::optional<std::vector<int>> basis_opt,
           const Eigen::VectorXd& l, const Eigen::VectorXd& u) {
        const int m = static_cast<int>(A.rows());
        const int n = static_cast<int>(A.cols());
        int iters = 0;

        // --- Basis initialization ---
        std::vector<int> basis;
        if (basis_opt) {
            basis = *basis_opt;
            if ((int)basis.size() != m)
                return {LPSolution::Status::NeedPhase1,
                        Eigen::VectorXd::Zero(n),
                        {},
                        0,
                        {{"reason", "basis size != m"}}};
        } else {
            auto maybe = find_initial_basis_(A, b, c);
            if (!maybe)
                return {LPSolution::Status::NeedPhase1,
                        Eigen::VectorXd::Zero(n),
                        {},
                        0,
                        {{"reason", "no_crash_basis"}}};
            basis = *maybe;
        }

        // Nonbasic list N
        std::vector<int> N;
        N.reserve(n - m);
        {
            std::vector<char> inB(n, 0);
            for (int j : basis) {
                if (j < 0 || j >= n)
                    return {LPSolution::Status::Singular,
                            Eigen::VectorXd::Zero(n),
                            basis,
                            0,
                            {{"where", "initial basis index out of range"}}};
                inB[j] = 1;
            }
            for (int j = 0; j < n; ++j)
                if (!inB[j]) N.push_back(j);
        }

        // Basis factor
        FTBasis B(A, basis, make_basis_options_());

        // Adaptive pricing setup (if enabled)
        if (opt_.pricing_rule == "adaptive") {
            AdaptivePricer::PricingOptions popts;
            popts.steepest_pool_max = 0;
            popts.steepest_reset_freq = opt_.adaptive_reset_freq;
            popts.devex_reset_freq = opt_.devex_reset;
            popts.dual_steepest_reset_freq = opt_.adaptive_reset_freq;
            adaptive_pricer_ = AdaptivePricer(n, popts);
            adaptive_pricer_.build_pools(B, A, N);
            bridge_ = std::make_unique<DegeneracyPricerBridge<AdaptivePricer>>(
                degen_, adaptive_pricer_);
        }

        int rebuild_attempts = 0;

        // --- Main loop ---
        while (iters < opt_.max_iters) {
            ++iters;

            // xB = B^{-1} b
            Eigen::VectorXd xB;
            try {
                xB = B.solve_B(b);
            } catch (...) {
                if (rebuild_attempts < opt_.max_basis_rebuilds) {
                    ++rebuild_attempts;
                    B.refactor();
                    if (opt_.pricing_rule == "adaptive") {
                        adaptive_pricer_.build_pools(B, A, N);
                        adaptive_pricer_.clear_rebuild_flag();
                    }
                    continue;
                }
                return {LPSolution::Status::Singular,
                        Eigen::VectorXd::Zero(n),
                        basis,
                        iters,
                        {{"where", "solve(B,b) repair failed"}}};
            }

            // Basic feasibility
            if ((xB.array() < -opt_.tol).any()) {
                // Signal to caller that primal is infeasible for this basis
                return {LPSolution::Status::NeedPhase1,
                        Eigen::VectorXd::Zero(n),
                        basis,
                        iters,
                        {{"reason", "negative_basic_vars"}}};
            }
            xB = xB.cwiseMax(0.0);

            // y = B^{-T} c_B
            Eigen::VectorXd cB(m);
            for (int i = 0; i < m; ++i) cB(i) = c(basis[i]);

            Eigen::VectorXd y;
            try {
                y = B.solve_BT(cB);
            } catch (...) {
                B.refactor();
                y = B.solve_BT(cB);
                if (opt_.pricing_rule == "adaptive") {
                    adaptive_pricer_.build_pools(B, A, N);
                    adaptive_pricer_.clear_rebuild_flag();
                }
            }

            // Reduced costs rN on nonbasics
            Eigen::VectorXd rN(N.size());
            for (int k = 0; k < (int)N.size(); ++k) {
                const int j = N[k];
                rN(k) = c(j) - A.col(j).dot(y);
            }

            // Choose entering
            std::optional<int> e_rel;

            if (opt_.bland) {
                int idx = -1;
                for (int k = 0; k < (int)N.size(); ++k)
                    if (rN(k) < -opt_.tol) {
                        idx = k;
                        break;
                    }
                if (idx < 0) {
                    Eigen::VectorXd x =
                        assemble_primal_(n, basis, xB, l, u);
                    return {LPSolution::Status::Optimal, clip_small_(x), basis,
                            iters, dm_stats_to_map(degen_.get_stats())};
                }
                e_rel = idx;
            } else {
                if (opt_.pricing_rule == "adaptive") {
                    // Current objective for adaptive signals
                    Eigen::VectorXd xcur =
                        assemble_primal_(n, basis, xB, l, u);
                    const double current_obj = c.dot(xcur);

                    e_rel = bridge_->choose_entering(rN, N, opt_.tol, iters,
                                                     current_obj, B, A);
                } else {
                    // Most negative rc
                    int idx = -1;
                    double best = 0.0;
                    for (int k = 0; k < (int)N.size(); ++k)
                        if (rN(k) < -opt_.tol) {
                            if (idx < 0 || rN(k) < best) {
                                best = rN(k);
                                idx = k;
                            }
                        }
                    if (idx >= 0) e_rel = idx;
                }

                if (!e_rel) {
                    Eigen::VectorXd x =
                        assemble_primal_(n, basis, xB, l, u);
                    return {LPSolution::Status::Optimal, clip_small_(x), basis,
                            iters, dm_stats_to_map(degen_.get_stats())};
                }
            }

            const int e = N[*e_rel];
            const auto aE = A.col(e);

            // dB = B^{-1} a_e
            Eigen::VectorXd dB;
            try {
                dB = B.solve_B(aE);
            } catch (...) {
                B.refactor();
                dB = B.solve_B(aE);
                if (opt_.pricing_rule == "adaptive") {
                    adaptive_pricer_.build_pools(B, A, N);
                    adaptive_pricer_.clear_rebuild_flag();
                }
            }

            // Harris ratio (leaving) from basics
            auto [leave_rel_opt, theta_B] =
                harris_ratio_(xB, dB, opt_.ratio_delta, opt_.ratio_eta);

            // BFRT step for entering bound
            const int idxN = *e_rel;
            const double rc_e = rN(idxN);
            const double l_e = (e >= 0 && e < l.size()) ? l(e) : 0.0;
            const double u_e =
                (e >= 0 && e < u.size()) ? u(e) : presolve::inf();
            const double x_e =
                std::isfinite(l_e) ? l_e : 0.0;  // nonbasic at bound
            const BFRTStep bfrt =
                entering_bound_step_(x_e, l_e, u_e, rc_e, opt_.tol);

            double step = std::min(theta_B, bfrt.theta_e);
            if (!std::isfinite(step)) {
                Eigen::VectorXd x = Eigen::VectorXd::Constant(
                    n, std::numeric_limits<double>::quiet_NaN());
                return {LPSolution::Status::Unbounded, x, basis, iters,
                        dm_stats_to_map(degen_.get_stats())};
            }

            // If BFRT wins strictly, flip enter direction locally
            const bool flip_entering = (bfrt.theta_e + 1e-14 < theta_B);
            if (flip_entering) {
                dB = -dB;
                const_cast<Eigen::VectorXd&>(rN)(idxN) = -rc_e;
            }

            if (!leave_rel_opt) {
                Eigen::VectorXd x = Eigen::VectorXd::Constant(
                    n, std::numeric_limits<double>::quiet_NaN());
                return {LPSolution::Status::Unbounded, x, basis, iters,
                        dm_stats_to_map(degen_.get_stats())};
            }

            const int r = *leave_rel_opt;
            const double alpha = dB(r);
            const int oldAbs = basis[r];
            const int eAbs = e;

            // Degeneracy signals
            const bool is_degenerate =
                degen_.detect_degeneracy(step, opt_.deg_step_tol);
            if (is_degenerate && degen_.should_apply_perturbation()) {
                auto [Ap, bp, cp] =
                    degen_.apply_perturbation(A, b, c, basis, iters);
                (void)Ap;
                (void)bp;
                (void)cp;  // no-op by default, preserves API
            } else {
                (void)degen_.reset_perturbation();
            }

            // Pricer updates
            if (opt_.pricing_rule == "adaptive") {
                const double rc_impr = -rN(idxN);
                bridge_->after_pivot(r, eAbs, oldAbs, dB, alpha, step, A, N,
                                     rc_impr);
            }

            // Pivot indices
            basis[r] = eAbs;
            N[idxN] = oldAbs;

            // Update basis matrix column
            try {
                B.replace_column(r, aE);
            } catch (...) {
                B.refactor();
                if (opt_.pricing_rule == "adaptive") {
                    adaptive_pricer_.build_pools(B, A, N);
                    adaptive_pricer_.clear_rebuild_flag();
                }
            }

            // Rebuild pricing pools if requested
            if (opt_.pricing_rule == "adaptive" &&
                adaptive_pricer_.needs_rebuild()) {
                adaptive_pricer_.build_pools(B, A, N);
                adaptive_pricer_.clear_rebuild_flag();
            }
        }

        return {LPSolution::Status::IterLimit, Eigen::VectorXd::Zero(n), basis,
                iters, dm_stats_to_map(degen_.get_stats())};
    }

    // --------------------------- DUAL PHASE ---------------------------
    // Dual Harris two-pass ratio test: entering e for leaving row r.
    // Inputs:
    //   r: leaving basic row (xB(r) < 0)
    //   rN: reduced costs on N (dual-feasible target rN >= -tol)
    //   pN: row r of B^{-1}A_N
    // Eligibility: pN(k) < -delta (so increasing τ ≥ 0 increases x_r by -τ
    // pN(k))
    struct DualChoose {
        std::optional<int> e_rel;
        double tau = std::numeric_limits<double>::infinity();
    };

    struct DualBFRTDecision {
        std::optional<int> pivot_rel;
        double tau = std::numeric_limits<double>::infinity();
        std::vector<int> flip_rels;
    };

    DualChoose dual_harris_choose_(const Eigen::VectorXd& rN,
                                   const Eigen::VectorXd& pN, double delta,
                                   double eta) const {
        std::vector<int> E;
        E.reserve((int)pN.size());
        for (int k = 0; k < pN.size(); ++k)
            if (pN(k) < -delta) E.push_back(k);
        if (E.empty()) return {};

        // First pass τ* = min_k rN_k / -pN_k
        double tau_star = std::numeric_limits<double>::infinity();
        for (int k : E) tau_star = std::min(tau_star, rN(k) / (-pN(k)));

        // Second pass: allow slight relaxation
        const double kappa = std::max(eta, eta * std::abs(tau_star));
        std::vector<int> candidates;
        for (int k : E) {
            if ((rN(k) / (-pN(k))) <= tau_star + kappa) candidates.push_back(k);
        }
        if (!candidates.empty()) {
            // Bland-ish tie-break
            int best = candidates.front();
            double best_ratio = rN(best) / (-pN(best));
            for (int kk : candidates) {
                const double val = rN(kk) / (-pN(kk));
                if ((val < best_ratio - 1e-16) ||
                    (std::abs(val - best_ratio) <= 1e-16 && kk < best)) {
                    best = kk;
                    best_ratio = val;
                }
            }
            return {best, std::max(0.0, best_ratio)};
        }

        // Fallback to strict minimum
        int best = E.front();
        double best_ratio = rN(best) / (-pN(best));
        for (int i = 1; i < (int)E.size(); ++i) {
            const int k = E[i];
            const double val = rN(k) / (-pN(k));
            if (val < best_ratio) {
                best_ratio = val;
                best = k;
            }
        }
        return {best, std::max(0.0, best_ratio)};
    }

    DualBFRTDecision dual_bfrt_decide_(const Eigen::VectorXd& rN,
                                       const Eigen::VectorXd& pN,
                                       const std::vector<int>& N,
                                       const std::vector<BoundView>& view,
                                       const Eigen::VectorXd& l,
                                       const Eigen::VectorXd& u,
                                       int max_flips) const {
        DualBFRTDecision out;
        DualChoose dc =
            dual_harris_choose_(rN, pN, opt_.ratio_delta, opt_.ratio_eta);
        out.pivot_rel = dc.e_rel;
        out.tau = dc.tau;
        if (!dc.e_rel || !std::isfinite(dc.tau) || max_flips <= 0) return out;

        struct Event {
            double tau;
            int rel;
        };
        std::vector<Event> events;
        events.reserve(N.size());
        const double tau_cap =
            dc.tau + std::max(opt_.ratio_eta, 1e-12 * (1.0 + dc.tau));

        for (int k = 0; k < (int)N.size(); ++k) {
            if (k == *dc.e_rel) continue;
            if (!(pN(k) < -opt_.ratio_delta)) continue;

            const int j = N[k];
            const double range = bound_range_(j, l, u);
            if (!std::isfinite(range) || range <= opt_.tol) continue;
            if (view[j] == BoundView::Fixed) continue;

            const double tau_k = rN(k) / (-pN(k));
            if (!std::isfinite(tau_k) || tau_k < 0.0 || tau_k > tau_cap) {
                continue;
            }
            events.push_back({tau_k, k});
        }

        std::sort(events.begin(), events.end(), [](const Event& a,
                                                   const Event& b) {
            if (std::abs(a.tau - b.tau) > 1e-16) return a.tau < b.tau;
            return a.rel < b.rel;
        });

        for (int i = 0; i < (int)events.size() && i < max_flips; ++i) {
            out.flip_rels.push_back(events[i].rel);
        }
        return out;
    }

    std::tuple<LPSolution::Status, Eigen::VectorXd, std::vector<int>, int,
               std::unordered_map<std::string, std::string>>
    dual_phase_(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
                const Eigen::VectorXd& c,
                std::optional<std::vector<int>> basis_opt,
                const Eigen::VectorXd& l, const Eigen::VectorXd& u) {
        const int m = static_cast<int>(A.rows());
        const int n = static_cast<int>(A.cols());
        int iters = 0;

        // --- Basis initialization ---
        std::vector<int> basis;
        if (basis_opt) {
            basis = *basis_opt;
            if ((int)basis.size() != m)
                return {LPSolution::Status::NeedPhase1,
                        Eigen::VectorXd::Zero(n),
                        {},
                        0,
                        {{"reason", "basis size != m"}}};
        } else {
            auto maybe = find_initial_basis_(A, b, c);
            if (!maybe)
                return {LPSolution::Status::NeedPhase1,
                        Eigen::VectorXd::Zero(n),
                        {},
                        0,
                        {{"reason", "no_crash_basis"}}};
            basis = *maybe;
        }

        // Nonbasic list N
        std::vector<int> N;
        N.reserve(n - m);
        {
            std::vector<char> inB(n, 0);
            for (int j : basis) {
                if (j < 0 || j >= n)
                    return {LPSolution::Status::Singular,
                            Eigen::VectorXd::Zero(n),
                            basis,
                            0,
                            {{"where", "initial basis index out of range"}}};
                inB[j] = 1;
            }
            for (int j = 0; j < n; ++j)
                if (!inB[j]) N.push_back(j);
        }

        std::vector<BoundView> view(n, BoundView::Lower);
        for (int j = 0; j < n; ++j) view[j] = default_bound_view_(j, l, u);
        bridge_.reset();
        DualSteepestEdgePricer dual_dse(/*pool_max=*/0,
                                        /*reset_frequency=*/
                                        opt_.adaptive_reset_freq);

        Eigen::MatrixXd Ahat = A;
        Eigen::VectorXd chat = c;
        for (int j : basis) {
            if (j >= 0 && j < n) view[j] = BoundView::Lower;
        }

        FTBasis B(Ahat, basis, make_basis_options_());

        auto apply_views_to_nonbasics = [&](const Eigen::VectorXd& ydual) {
            bool changed = false;
            std::vector<char> inB(n, 0);
            for (int j : basis)
                if (j >= 0 && j < n) inB[j] = 1;

            for (int j = 0; j < n; ++j) {
                if (inB[j]) continue;

                const double raw_rc = c(j) - A.col(j).dot(ydual);
                const bool has_l = (j < l.size()) && std::isfinite(l(j));
                const bool has_u = (j < u.size()) && std::isfinite(u(j));
                BoundView next = view[j];

                if (has_l && has_u) {
                    if (std::abs(u(j) - l(j)) <= opt_.tol) {
                        next = BoundView::Fixed;
                    } else {
                        next = (raw_rc < 0.0) ? BoundView::Upper
                                              : BoundView::Lower;
                    }
                } else if (has_u && !has_l) {
                    next = BoundView::Upper;
                } else {
                    next = BoundView::Lower;
                }

                if (next != view[j]) {
                    view[j] = next;
                    changed = true;
                }
            }

            if (changed) {
                for (int j = 0; j < n; ++j) {
                    const double sign = static_cast<double>(view_sign_(view[j]));
                    if (sign > 0.0) {
                        Ahat.col(j) = A.col(j);
                        chat(j) = c(j);
                    } else {
                        Ahat.col(j) = -A.col(j);
                        chat(j) = -c(j);
                    }
                }
            }
            return changed;
        };

        // Make the initial basis dual-feasible with respect to nonbasic views.
        {
            Eigen::VectorXd cB(m);
            for (int i = 0; i < m; ++i) cB(i) = chat(basis[i]);
            Eigen::VectorXd ydual = B.solve_BT(cB);
            apply_views_to_nonbasics(ydual);
        }

        // Rebuild factor if any basis column orientation ever changes later.
        for (int j : basis) {
            if (view_sign_(view[j]) < 0) {
                Ahat.col(j) = -A.col(j);
                chat(j) = -c(j);
            }
        }
        B.refactor();
        dual_dse.build_pool(B, Ahat, N);

        int rebuild_attempts = 0;
        int total_flips = 0;

        // helper: serialize a vector to CSV for info map
        auto serialize_vec = [](const Eigen::VectorXd& v) {
            std::ostringstream oss;
            oss.setf(std::ios::scientific);
            oss << std::setprecision(17);
            for (int i = 0; i < v.size(); ++i) {
                if (i) oss << ",";
                oss << v(i);
            }
            return oss.str();
        };

        // --- Main dual loop ---
        while (iters < opt_.max_iters) {
            ++iters;
            int flips_this_iter = 0;
            Eigen::VectorXd rhs_eff = b - transformed_rhs_(A, view, l, u);
            Eigen::VectorXd yB;
            Eigen::VectorXd cB(m);
            Eigen::VectorXd ydual;
            Eigen::VectorXd pN;
            Eigen::VectorXd rN;
            int r_leave = -1;
            Eigen::VectorXd w;
            int e_rel = -1;
            int eAbs = -1;
            Eigen::VectorXd s_enter;
            double tau = std::numeric_limits<double>::infinity();

            while (true) {
                try {
                    yB = B.solve_B(rhs_eff);
                } catch (...) {
                    if (rebuild_attempts < opt_.max_basis_rebuilds) {
                        ++rebuild_attempts;
                        B.refactor();
                        dual_dse.build_pool(B, Ahat, N);
                        continue;
                    }
                    return {LPSolution::Status::Singular,
                            Eigen::VectorXd::Zero(n),
                            basis,
                            iters,
                            {{"where", "dual: solve(Bhat,rhs) repair failed"}}};
                }

                for (int i = 0; i < m; ++i) cB(i) = chat(basis[i]);
                try {
                    ydual = B.solve_BT(cB);
                } catch (...) {
                    B.refactor();
                    dual_dse.build_pool(B, Ahat, N);
                    ydual = B.solve_BT(cB);
                }

                if (apply_views_to_nonbasics(ydual)) {
                    rhs_eff = b - transformed_rhs_(A, view, l, u);
                    dual_dse.build_pool(B, Ahat, N);
                    continue;
                }

                const auto leaving = dual_dse.choose_leaving(B, yB, opt_.tol);
                r_leave = leaving.row;
                if (r_leave < 0) {
                    rN.resize(N.size());
                    bool dual_feasible = true;
                    for (int k = 0; k < (int)N.size(); ++k) {
                        const int j = N[k];
                        rN(k) = chat(j) - Ahat.col(j).dot(ydual);
                        if (rN(k) < -opt_.tol) dual_feasible = false;
                    }
                    if (dual_feasible) {
                        Eigen::VectorXd x = assemble_transformed_primal_(
                            n, basis, yB.cwiseMax(0.0), l, u, view);
                        auto info_map = dm_stats_to_map(degen_.get_stats());
                        info_map["dual_pricing"] = "dual_steepest";
                        info_map["dual_bfrt_flips"] =
                            std::to_string(total_flips);
                        return {LPSolution::Status::Optimal, std::move(x), basis,
                                iters, std::move(info_map)};
                    }
                    return {LPSolution::Status::NeedPhase1,
                            Eigen::VectorXd::Zero(n),
                            basis,
                            iters,
                            {{"reason", "dual_infeasible_at_primal_feasible"}}};
                }

                w = leaving.dual_row;
                pN.resize(N.size());
                rN.resize(N.size());
                for (int k = 0; k < (int)N.size(); ++k) {
                    const int j = N[k];
                    pN(k) = w.dot(Ahat.col(j));
                    rN(k) = chat(j) - Ahat.col(j).dot(ydual);
                }

                const DualBFRTDecision bfrt = dual_bfrt_decide_(
                    rN, pN, N, view, l, u,
                    opt_.dual_allow_bound_flip
                        ? (opt_.dual_flip_max_per_iter - flips_this_iter)
                        : 0);
                if (!bfrt.pivot_rel) {
                    if (rebuild_attempts < opt_.max_basis_rebuilds) {
                        ++rebuild_attempts;
                        B.refactor();
                        dual_dse.build_pool(B, Ahat, N);
                        continue;
                    }
                    return {LPSolution::Status::Singular,
                            Eigen::VectorXd::Zero(n),
                            basis,
                            iters,
                            {{"where", "dual: no eligible entering"}}};
                }

                if (!bfrt.flip_rels.empty()) {
                    for (int rel_k : bfrt.flip_rels) {
                        const int j = N[rel_k];
                        const double old_anchor = bound_anchor_(view[j], j, l, u);
                        view[j] = (view[j] == BoundView::Upper)
                                      ? BoundView::Lower
                                      : BoundView::Upper;
                        const double new_anchor = bound_anchor_(view[j], j, l, u);
                        const double delta_anchor = new_anchor - old_anchor;
                        if (delta_anchor != 0.0) {
                            rhs_eff.noalias() -= A.col(j) * delta_anchor;
                        }
                        Ahat.col(j) = -Ahat.col(j);
                        chat(j) = -chat(j);
                        ++flips_this_iter;
                        ++total_flips;
                    }
                    dual_dse.build_pool(B, Ahat, N);
                    continue;
                }

                e_rel = *bfrt.pivot_rel;
                eAbs = N[e_rel];
                tau = bfrt.tau;
                try {
                    s_enter = B.solve_B(Ahat.col(eAbs));
                } catch (...) {
                    if (rebuild_attempts < opt_.max_basis_rebuilds) {
                        ++rebuild_attempts;
                        B.refactor();
                        dual_dse.build_pool(B, Ahat, N);
                        continue;
                    }
                    return {LPSolution::Status::Singular,
                            Eigen::VectorXd::Zero(n),
                            basis,
                            iters,
                            {{"where", "dual: solve(Bhat,a_e) repair failed"}}};
                }
                break;
            }

            if (!std::isfinite(tau)) {
                Eigen::VectorXd yF = w;
                if (yF.dot(rhs_eff) >= 0) yF = -yF;

                auto info_map = dm_stats_to_map(degen_.get_stats());
                info_map["where"] = "dual: infinite step";
                info_map["dual_pricing"] = "dual_steepest";
                info_map["dual_bfrt_flips"] = std::to_string(total_flips);
                info_map["farkas_has_cert"] = "1";
                info_map["farkas_dim"] = std::to_string(m);
                info_map["farkas_y"] = serialize_vec(yF);

                return {LPSolution::Status::Infeasible,
                        Eigen::VectorXd::Zero(n), basis, iters,
                        std::move(info_map)};
            }

            const bool is_degenerate =
                degen_.detect_degeneracy(tau, opt_.deg_step_tol);
            if (is_degenerate && degen_.should_apply_perturbation()) {
                auto [Ap, bp, cp] =
                    degen_.apply_perturbation(A, b, c, basis, iters);
                (void)Ap;
                (void)bp;
                (void)cp;
            } else {
                (void)degen_.reset_perturbation();
            }

            const int oldAbs = basis[r_leave];
            basis[r_leave] = eAbs;
            N[e_rel] = oldAbs;

            try {
                B.replace_column(r_leave, Ahat.col(eAbs));
            } catch (...) {
                B.refactor();
                dual_dse.build_pool(B, Ahat, N);
            }

            dual_dse.update_after_pivot(r_leave, eAbs, oldAbs, s_enter,
                                        s_enter(r_leave), Ahat, N, &w, true);
            if (dual_dse.needs_rebuild()) {
                dual_dse.build_pool(B, Ahat, N);
                dual_dse.clear_rebuild_flag();
            }
        }

        auto info_map = dm_stats_to_map(degen_.get_stats());
        info_map["dual_pricing"] = "dual_steepest";
        info_map["dual_bfrt_flips"] = std::to_string(total_flips);
        return {LPSolution::Status::IterLimit, Eigen::VectorXd::Zero(n), basis,
                iters, std::move(info_map)};
    }

    // --------------------------- Utilities ---------------------------
    static LPSolution make_solution_(
        LPSolution::Status st, Eigen::VectorXd x, double obj,
        std::vector<int> basis, int iters,
        std::unordered_map<std::string, std::string> info,
        std::optional<Eigen::VectorXd> farkas_y = std::nullopt,
        std::optional<bool> farkas_has_cert = std::nullopt) {
        LPSolution sol;
        sol.status = st;
        sol.x = std::move(x);
        sol.obj = obj;
        sol.basis = std::move(basis);
        sol.iters = iters;
        sol.info = std::move(info);
        sol.farkas_y =
            farkas_y ? std::move(*farkas_y) : Eigen::VectorXd{};
        sol.farkas_has_cert = farkas_has_cert.value_or(false);
        return sol;
    }

   private:
    // Options and state
    RevisedSimplexOptions opt_;
    std::mt19937 rng_;

    // Degeneracy + pricing
    DegeneracyManager degen_;
    AdaptivePricer adaptive_pricer_{1};
    std::unique_ptr<DegeneracyPricerBridge<AdaptivePricer>> bridge_;
};
