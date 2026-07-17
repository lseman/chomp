#pragma once

// ====== Core Eigen ======
#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/SparseCore>

// ====== STL ======
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

// ====== Project ======
#include <blocks/filter.h>
#include <core/definitions.h>
#include <model/model.h>
#include "trustregion/definitions.h"

enum class TRBackend { PCG, DOGLEG, GLTR };

// ---------- NEW: Dogleg (dense SPD) ----------
struct DoglegResult {
    dvec p;
    int iters = 1;
    const char* status = "dogleg";
};

DoglegResult dogleg_step_(const dmat& Hspd, const dvec& g, const Metric& M,
                          double delta) {
    // Symmetrize (cheap) for safety
    const dmat Hs = 0.5 * (Hspd + Hspd.transpose());

    // Cauchy direction
    const dvec Hg = Hs * g;
    const double gg = g.dot(g);
    const double gHg = g.dot(Hg);
    const double alpha_sd = (gHg > 0.0) ? (gg / gHg) : 0.0;
    dvec pC = -alpha_sd * g;

    // Newton step: H pN = -g
    dvec pN;
    {
        Eigen::LLT<dmat> llt(Hs);
        if (llt.info() == Eigen::Success) {
            pN = llt.solve(-g);
        } else {
            Eigen::LDLT<dmat> ldlt(Hs);
            pN = ldlt.solve(-g);
        }
    }

    auto normM = [&](const dvec& v) -> double {
        return M.valid ? M.norm(v) : v.norm();
    };

    const double nC = normM(pC);
    const double nN = normM(pN);

    if (nN <= delta) {
        return {pN, 1, "dogleg_newton"};
    }
    if (nC >= delta) {
        const double t = std::max(1e-16, nC);
        return {(delta / t) * pC, 1, "dogleg_cauchy_boundary"};
    }

    // interpolate on dogleg: p(τ) = pC + τ (pN - pC), ||p(τ)||_M = delta
    const dvec d = pN - pC;
    const double tau = detail::boundary_intersection_metric(M, pC, d, delta);
    return {pC + std::clamp(tau, 0.0, 1.0) * d, 1, "dogleg_interpolate"};
}

// -----------------------------------------------------------------------------
// GLTR (Generalized Lanczos Trust Region) for trust region subproblems
// -----------------------------------------------------------------------------
struct GLTRResult {
    dvec p;
    TRStatus status;
    int iters;
};

[[nodiscard]] GLTRResult gltr_step_(const LinOp& H, const dvec& g, const Metric& M,
                                    double delta, double tol, int maxiter) {
    const int n = (int)g.size();
    if (n == 0 || g.norm() <= tol) {
        return {dvec::Zero(n), TRStatus::SUCCESS, 0};
    }

    bool use_metric = M.valid;
    LinOp H_op = H;
    dvec g_op = g;

    // Transform to metric space if needed
    if (use_metric) {
        g_op = M.L.triangularView<Eigen::Lower>().solve(g);
        LinOp new_H;
        new_H.n = n;
        new_H.mv = [&M, &H](const dvec& v, dvec& out) {
            dvec tmp = M.L.transpose().triangularView<Eigen::Upper>().solve(v);
            dvec Htmp;
            H_apply(H, tmp, Htmp);
            out = M.L.triangularView<Eigen::Lower>().solve(Htmp);
        };
        H_op = new_H;
    }

    const double gnorm = g_op.norm();
    if (gnorm <= tol) {
        return {dvec::Zero(n), TRStatus::SUCCESS, 0};
    }

    // Lanczos vectors and tridiagonal matrix
    std::vector<dvec> V;
    V.reserve(maxiter + 1);
    dvec v = g_op / gnorm;
    V.push_back(std::move(v));

    dvec w;
    H_apply(H_op, V.back(), w);

    double a = V.back().dot(w);
    w -= a * V.back();
    double b = w.norm();

    // Tridiagonal matrix T (stored as vectors for efficiency)
    std::vector<double> T_diag;
    std::vector<double> T_off;
    T_diag.reserve(maxiter + 1);
    T_off.reserve(maxiter);
    T_diag.push_back(a);

    double lambda = 0.0;
    bool converged = false;
    int k = 0;
    dvec s = dvec::Zero(n);
    bool interior = true;

    for (k = 0; k < maxiter; ++k) {
        // Check for negative curvature in Lanczos process
        if (b < 1e-14 && k > 0) {
            // Compute smallest eigenvalue of current T
            dmat T_curr(k + 1, k + 1);
            for (int i = 0; i <= k; ++i) T_curr(i, i) = T_diag[i];
            for (int i = 0; i < k; ++i) {
                T_curr(i, i + 1) = T_off[i];
                T_curr(i + 1, i) = T_off[i];
            }
            Eigen::SelfAdjointEigenSolver<dmat> es(T_curr);
            double lambda_min = es.eigenvalues().minCoeff();
            if (lambda_min < -1e-12) {
                return {s, TRStatus::NEG_CURV, k + 1};
            }
        }

        // Extend Lanczos process
        if (b > 1e-14) {
            v = w / b;
            V.push_back(v);
            H_apply(H_op, v, w);
            double a_new = v.dot(w);
            w -= a_new * v;
            w -= b * V[k];
            double b_new = w.norm();

            T_diag.push_back(a_new);
            T_off.push_back(b);
            a = a_new;
            b = b_new;
        } else {
            // Break if Lanczos process breaks down (invariant subspace reached)
            break;
        }

        // Solve the trust region subproblem on the tridiagonal system
        const int m = (int)T_diag.size();
        dmat T(m, m);
        for (int i = 0; i < m; ++i) T(i, i) = T_diag[i];
        for (int i = 0; i < m - 1; ++i) {
            T(i, i + 1) = T_off[i];
            T(i + 1, i) = T_off[i];
        }

        // Find the smallest eigenvalue and corresponding Lagrange multiplier
        Eigen::SelfAdjointEigenSolver<dmat> es(T);
        double lambda_min = es.eigenvalues().minCoeff();
        lambda = std::max(0.0, -lambda_min);

        // Newton iteration to solve the secular equation
        dvec h(m);
        double phi = 0.0;
        bool newton_converged = false;
        const int max_newton = 50;

        for (int newton_iter = 0; newton_iter < max_newton; ++newton_iter) {
            dmat Tp = T + lambda * dmat::Identity(m, m);
            Eigen::LDLT<dmat> ldlt(Tp);
            if (ldlt.info() != Eigen::Success) {
                lambda += std::max(1e-8, 0.1 * lambda + 1e-6);
                continue;
            }

            dvec e1 = dvec::Zero(m);
            e1(0) = gnorm;
            h = -ldlt.solve(e1);

            const double hnorm = h.norm();
            phi = hnorm - delta;

            if (lambda <= 1e-12 && hnorm <= delta * (1.0 + 1e-10)) {
                interior = true;
                newton_converged = true;
                break;
            }

            if (lambda > 1e-12 && std::abs(hnorm - delta) <= 1e-6 * delta) {
                interior = false;
                newton_converged = true;
                break;
            }

            // Secular equation derivative: dphi/dlambda = -h^T (T + lambda I)^{-2} h
            dvec rhs = ldlt.solve(h);
            double dphi_dlambda = -h.dot(rhs);

            if (std::abs(dphi_dlambda) < 1e-14) {
                lambda += 1e-6;
                continue;
            }

            const double dlambda = -phi / dphi_dlambda;
            lambda += dlambda;
            lambda = std::max(lambda, -lambda_min + 1e-10);
        }

        if (!newton_converged) {
            // Fallback: scale to boundary
            if (h.norm() > 1e-16) {
                h = h * (delta / h.norm());
            }
        }

        // Check Lanczos residual: ||b_k * h_k * v_k|| <= tol * gnorm
        double last_h = h(m - 1);
        double residual = (k < (int)T_off.size() && T_off[k] > 1e-14)
                              ? T_off[k] * std::abs(last_h)
                              : 0.0;

        if (residual <= tol * gnorm || newton_converged) {
            converged = true;
        }

        // Reconstruct solution in original space
        s = dvec::Zero(n);
        for (int i = 0; i < (int)V.size(); ++i) {
            if (i < (int)h.size()) {
                s += h(i) * V[i];
            }
        }
    }

    // Transform back to original space if metric was used
    dvec p_final = s;
    if (use_metric) {
        p_final = M.L.transpose().triangularView<Eigen::Upper>().solve(s);
    }

    TRStatus status = TRStatus::MAX_ITER;
    if (converged) status = TRStatus::SUCCESS;
    if (b < 1e-14 && k > 0) {
        // Check for negative curvature
        dmat T_curr(k + 1, k + 1);
        for (int i = 0; i <= k; ++i) T_curr(i, i) = T_diag[i];
        for (int i = 0; i < k; ++i) {
            T_curr(i, i + 1) = T_off[i];
            T_curr(i + 1, i) = T_off[i];
        }
        Eigen::SelfAdjointEigenSolver<dmat> es(T_curr);
        if (es.eigenvalues().minCoeff() < -1e-12) {
            status = TRStatus::NEG_CURV;
        }
    }
    if (!interior && converged) status = TRStatus::BOUNDARY;

    return {p_final, status, k + 1};
}

// -----------------------------------------------------------------------------
// CG (Steihaug-PCG)
// -----------------------------------------------------------------------------
struct CGResult {
    dvec p;
    TRStatus status;
    int iters;
};

[[nodiscard]] inline CGResult steihaug_pcg(const LinOp& H, const dvec& g,
                                           const Metric& M, double Delta,
                                           double tol, int maxiter,
                                           double neg_curv_tol, const Prec& P,
                                           TRWorkspace& W) {
    const int n = H.n;
    W.ensure(n);
    auto& p = W.p_try;
    auto& r = W.r;
    auto& z = W.z;
    auto& d = W.d;

    // Initialize PCG
    double rz = 0.0;
    if (W.has_warm_start) {
        // Warm-start: use previous solution as initial guess
        // Recompute residual: r = -g - H*p
        p = W.p_warm_start;
        r = -g;
        H_apply(H, p, W.Hd);
        r.noalias() -= W.Hd;
        // Precondition residual
        P.apply_into(r, z);
        d = z;
        rz = r.dot(z);
        
        // Check if warm-start already satisfies tolerance
        if (r.norm() <= tol) {
            W.clear_warm_start();
            return {p, TRStatus::SUCCESS, 0};
        }
    } else {
        // Initialize from scratch
        p.setZero();
        r = -g;
        P.apply_into(r, z);
        d = z;

        if (r.norm() <= tol) {
            return {p, TRStatus::SUCCESS, 0};
        }

        rz = r.dot(z);
    }

    // PCG iterations
    for (int k = 0; k < maxiter; ++k) {
        H_apply(H, d, W.Hd);
        const double dTHd = d.dot(W.Hd);
        if (dTHd <= neg_curv_tol * std::max(1.0, d.squaredNorm())) {
            const double tau =
                detail::boundary_intersection_metric(M, p, d, Delta);
            p.noalias() += tau * d;
            // Save warm-start state for next iteration
            W.p_warm_start = p;
            W.r_warm_start = r;
            W.has_warm_start = true;
            return {p, TRStatus::NEG_CURV, k};
        }
        const double denom =
            (std::abs(dTHd) > detail::kTinyDen)
                ? dTHd
                : (dTHd >= 0 ? detail::kTinyDen : -detail::kTinyDen);
        const double alpha = rz / denom;

        W.tmp.noalias() = p + alpha * d;
        if (M.norm(W.tmp) >= Delta) {
            const double tau =
                detail::boundary_intersection_metric(M, p, d, Delta);
            p.noalias() += tau * d;
            // Save warm-start state for next iteration
            W.p_warm_start = p;
            W.r_warm_start = r;
            W.has_warm_start = true;
            return {p, TRStatus::BOUNDARY, k};
        }
        p.swap(W.tmp);

        r.noalias() -= alpha * W.Hd;
        if (r.norm() <= tol) {
            // Save warm-start state for next iteration
            W.p_warm_start = p;
            W.r_warm_start = r;
            W.has_warm_start = true;
            return {p, TRStatus::SUCCESS, k + 1};
        }

        P.apply_into(r, W.z_next);
        const double rz_next = r.dot(W.z_next);
        const double beta = rz_next / std::max(rz, 1e-32);
        d.noalias() = W.z_next + beta * d;
        rz = rz_next;
    }
    // Save warm-start state for next iteration
    W.p_warm_start = p;
    W.r_warm_start = r;
    W.has_warm_start = true;
    return {p, TRStatus::MAX_ITER, maxiter};
}
