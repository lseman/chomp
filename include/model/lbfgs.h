#pragma once

#include <core/types.h>
#include <Eigen/Cholesky>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace detail {

// ----- Limited-memory BFGS for the Lagrangian Hessian (compact form) -----
// ----- Limited-memory BFGS for the Lagrangian Hessian (advanced) -----
struct LBFGSLagHess {
    // ===== Configuration =====
    int    m_mem             = 20;    // memory size
    double curvature_eps     = 1e-8;  // relative guard
    double densify_threshold = 1e-12; // sparseView threshold
    bool   use_damping       = true;  // Powell damping
    bool   allow_indefinite  = false; // if true, can blend SR1
    double sr1_blend         = 0.0;   // in [0,1]; 0 = pure BFGS
    double collinear_tol     = 1e-3;  // merge if angle < ~0.057 rad
    double scale_floor       = 1e-16; // floor on H0 scale gamma

    // ===== Storage =====
    int                 n = 0;
    std::vector<dvec>   S;           // s_k
    std::vector<dvec>   Y;           // y_k
    std::vector<double> rho;         // 1/(y^T s) after damping
    double              gamma = 1.0; // H0 = gamma * I

    // Cached compact factors (for materialization path of B)
    Eigen::MatrixXd              M; // 2m x 2m
    Eigen::LDLT<Eigen::MatrixXd> M_ldlt;
    Eigen::MatrixXd              STY, STS;
    bool                         need_factor = true;

    void reset(int n_new) {
        n = n_new;
        S.clear();
        Y.clear();
        rho.clear();
        gamma       = 1.0;
        need_factor = true;
        M.resize(0, 0);
        STY.resize(0, 0);
        STS.resize(0, 0);
        M_ldlt = Eigen::LDLT<Eigen::MatrixXd>();
    }
    bool has_pairs() const { return !S.empty(); }
    int  mem() const { return (int)S.size(); }

    // ---- Helper: robust H0 scale (BB + safeguards) ----
    static double robust_gamma(const dvec &s, const dvec &y, double floor_v) {
        const double sty = s.dot(y);
        const double yty = y.squaredNorm();
        const double val = (yty > floor_v) ? (sty / yty) : 1.0;
        return (std::isfinite(val) && val > floor_v) ? val : 1.0;
    }

    // ---- Helper: near-collinearity merge (replace the closest s_i) ----
    void maybe_aggregate(const dvec &s_new, const dvec &y_new) {
        if (S.empty()) return;
        // Find most collinear existing s_i
        int          best         = -1;
        double       best_abs_cos = -1.0;
        const double ns           = s_new.norm();
        if (ns <= 0) return;
        for (int i = 0; i < mem(); ++i) {
            const double c = std::abs(S[i].dot(s_new) / (S[i].norm() * ns + 1e-32));
            if (c > best_abs_cos) {
                best_abs_cos = c;
                best         = i;
            }
        }
        if (best_abs_cos >= 1.0 - 0.5 * collinear_tol * collinear_tol) {
            // Merge: simple weighted sum (keep scale similar)
            const double ws = 0.5, wy = 0.5;
            S[best]     = ws * S[best] + (1.0 - ws) * s_new;
            Y[best]     = wy * Y[best] + (1.0 - wy) * y_new;
            need_factor = true;
        } else {
            // no-op; caller will push_back
        }
    }

    // ---- Update with (x_{k+1}-x_k, gL_{k+1}-gL_k), with damping/caution ----
    void update(const dvec &x_prev, const dvec &g_prev, const dvec &x_curr, const dvec &g_curr) {
        if (x_prev.size() == 0 || g_prev.size() == 0) return;
        if (x_curr.size() != x_prev.size() || g_curr.size() != g_prev.size())
            throw std::runtime_error("LBFGSLagHess::update: size mismatch");
        if (n == 0) reset((int)x_curr.size());

        dvec s = x_curr - x_prev;
        dvec y = g_curr - g_prev;

        const double s_norm = s.norm(), y_norm = y.norm();
        if (s_norm < 1e-14 || y_norm < 1e-14) return;

        const double sty = s.dot(y);
        const double yty = y.squaredNorm();
        const double sts = s.squaredNorm();

        if (!(std::isfinite(sty) && std::isfinite(yty) && std::isfinite(sts))) return;

        // Relative curvature guard (Nocedal-Wright style)
        const double rel_guard = curvature_eps * std::sqrt(sts * yty);
        if (sty <= rel_guard) {
            // Try Powell damping: y <- theta*y + (1-theta) * (B s)
            if (use_damping) {
                // Approximate B s with simple scaled identity (pre-update)
                const double inv_gamma0 = 1.0 / std::max(robust_gamma(s, y, scale_floor), scale_floor);
                const dvec   Bs         = inv_gamma0 * s;
                const double sBs        = s.dot(Bs);
                const double theta      = 0.8 * (sBs - rel_guard) / (sBs - sty + 1e-32);
                if (theta > 0.0 && theta < 1.0) { y = theta * y + (1.0 - theta) * Bs; }
            }
        }

        // Recompute after damping
        const double sty2 = s.dot(y);
        if (sty2 <= rel_guard) {
            // still bad curvature: either skip or aggregate
            if (mem() > 0) {
                maybe_aggregate(s, y);
            } else {
                return;
            } // nothing to merge with; skip
        } else {
            // Update gamma (H0 scale) using robust BB rule
            gamma = robust_gamma(s, y, scale_floor);

            // Manage memory / collinearity
            if (mem() == m_mem) {
                S.erase(S.begin());
                Y.erase(Y.begin());
                rho.erase(rho.begin());
            }
            // Try to aggregate before pushing if highly collinear with existing
            int before = mem();
            maybe_aggregate(s, y);
            if (mem() == before) {
                S.emplace_back(std::move(s));
                Y.emplace_back(std::move(y));
            }

            // rho = 1 / (y^T s) for two-loop
            const double r = 1.0 / std::max(1e-32, Y.back().dot(S.back()));
            rho.emplace_back(r);
            need_factor = true;
        }
    }

    // ===== Two-loop recursion for H v (inverse Hessian action) =====
    dvec apply_H(const dvec &v) const {
        const int m = mem();
        if (m == 0) return gamma * v; // H0 v
        std::vector<double> alpha(m);
        dvec                q = v;
        // First loop
        for (int i = m - 1; i >= 0; --i) {
            alpha[i] = rho[i] * S[i].dot(q);
            q.noalias() -= alpha[i] * Y[i];
        }
        // Apply H0
        q *= gamma;
        // Second loop
        for (int i = 0; i < m; ++i) {
            const double beta = rho[i] * Y[i].dot(q);
            q.noalias() += (alpha[i] - beta) * S[i];
        }
        return q;
    }

    // Optional: SR1 blending on top of H action (captures indefiniteness)
    dvec apply_H_sr1(const dvec &v) const {
        if (sr1_blend <= 0.0) return apply_H(v);
        const dvec Hv = apply_H(v);
        if (!allow_indefinite) return Hv;
        // One-pass limited SR1 correction using the most recent pair
        if (mem() == 0) return Hv;
        const dvec  &s     = S.back();
        const dvec  &y     = Y.back();
        const dvec   Hy    = apply_H(y);
        const dvec   u     = (s - Hy);
        const double denom = u.dot(y);
        if (std::abs(denom) < 1e-12) return Hv;
        const double coeff = sr1_blend * (u.dot(v) / denom);
        return Hv + coeff * u;
    }

    // Convenience aliases
    dvec inv_hess_vec(const dvec &v) const { return apply_H_sr1(v); } // H v
    dvec hess_vec(const dvec &v) const {
        // Use B = H^{-1} via CG if needed; approximate with symmetric secant:
        // Here we provide a cheap fallback: B ≈ (1/gamma) I - secant low-rank.
        // For high accuracy, prefer build_sparse_matrix() or a matrix-free CG.
        const double inv_gamma = 1.0 / std::max(gamma, scale_floor);
        // Use compact formula if pairs exist:
        if (!has_pairs()) return inv_gamma * v;
        // Build W^T v and solve with M (same as your prior apply_B)
        // Use factor_if_needed() path:
        return apply_B(v); // reuses compact route below
    }

    // ===== Materialization path for B (kept from your previous version) =====
    void factor_if_needed() const {
        if (!need_factor) return;
        auto     *self = const_cast<LBFGSLagHess *>(this);
        const int m    = mem();
        self->STY.resize(m, m);
        self->STS.resize(m, m);
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < m; ++j) {
                self->STY(i, j) = S[i].dot(Y[j]);
                self->STS(i, j) = S[i].dot(S[j]);
            }
        const double inv_gamma = 1.0 / std::max(gamma, scale_floor);
        self->M.resize(2 * m, 2 * m);
        self->M.setZero();

        self->M.block(0, 0, m, m) = inv_gamma * self->STS; // S^T B0 S
        Eigen::MatrixXd L         = self->STY.triangularView<Eigen::StrictlyLower>();
        self->M.block(0, m, m, m) = L;
        self->M.block(m, 0, m, m) = L.transpose();

        Eigen::VectorXd d = self->STY.diagonal().array().abs().max(1e-12);
        self->M.block(m, m, m, m).setZero();
        self->M.block(m, m, m, m).diagonal() = -d;

        // tiny ridge
        self->M.diagonal().array() += 1e-12;
        self->M_ldlt.compute(self->M.selfadjointView<Eigen::Lower>());
        if (self->M_ldlt.info() != Eigen::Success) {
            Eigen::MatrixXd M2 = self->M;
            M2.diagonal().array() += 1e-9;
            self->M_ldlt.compute(M2.selfadjointView<Eigen::Lower>());
        }
        self->need_factor = false;
    }

    dvec apply_B(const dvec &v) const {
        const int    m         = mem();
        const double inv_gamma = 1.0 / std::max(gamma, scale_floor);
        if (m == 0) return inv_gamma * v;

        factor_if_needed();
        Eigen::VectorXd Wt_v(2 * m);
        for (int j = 0; j < m; ++j) {
            Wt_v(j)     = inv_gamma * S[j].dot(v);
            Wt_v(m + j) = Y[j].dot(v);
        }
        Eigen::VectorXd z = M_ldlt.solve(Wt_v);
        if (M_ldlt.info() != Eigen::Success) {
            Eigen::MatrixXd M2 = M;
            M2.diagonal().array() += 1e-9;
            Eigen::LDLT<Eigen::MatrixXd> bk;
            bk.compute(M2.selfadjointView<Eigen::Lower>());
            z = bk.solve(Wt_v);
        }
        dvec Wz = dvec::Zero(n);
        for (int j = 0; j < m; ++j) Wz.noalias() += (inv_gamma * z(j)) * S[j];
        for (int j = 0; j < m; ++j) Wz.noalias() += z(m + j) * Y[j];
        return inv_gamma * v - Wz;
    }

    // Build a (dense->sparse) B matrix for downstream solvers
    spmat build_sparse_matrix(int n_build, double thresh) const {
        const double inv_gamma = 1.0 / std::max(gamma, scale_floor);
        if (!has_pairs()) {
            spmat D(n_build, n_build);
            D.reserve(Eigen::ArrayXi::Constant(n_build, 1));
            for (int i = 0; i < n_build; ++i) D.insert(i, i) = inv_gamma;
            D.makeCompressed();
            return D;
        }
        Eigen::MatrixXd Bdense(n_build, n_build);
        for (int i = 0; i < n_build; ++i) {
            dvec ei       = dvec::Zero(n_build);
            ei[i]         = 1.0;
            Bdense.col(i) = apply_B(ei);
        }
        Eigen::MatrixXd Bsym = 0.5 * (Bdense + Bdense.transpose());
        auto            SPM  = Bsym.sparseView(thresh, 1.0);
        return SPM;
    }
};

} // namespace detail
