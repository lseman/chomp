#pragma once

#include <Eigen/Cholesky>
#include <Eigen/LU>

#include <algorithm>
#include <cmath>

#include <core/definitions.h>
#include <linear_system/qdldl/qdldl.h>
#include <linear_system/common/sparse_csc.h>

// -----------------------------------------------------------------------------
// AAT cached solve for projection — QDLDL version (robust; int32 guaranteed)
// -----------------------------------------------------------------------------
namespace {
inline dvec solve_sym_spd_qdldl(const dmat& N_in, const dvec& b_in) {
    using namespace qdldl23;
    using FloatT = double;
    using IntT = int32_t;

    const int m = static_cast<int>(N_in.rows());
    if (m == 0) return dvec(0);

    // Pack the (dense) upper triangle of N into CSC expected by QDLDL
    SparseD32 S;
    S.n = static_cast<IntT>(m);
    S.Ap.resize(static_cast<size_t>(m) + 1);
    S.Ap[0] = 0;
    for (IntT j = 0; j < (IntT)m; ++j)
        S.Ap[(size_t)j + 1] = S.Ap[(size_t)j] + (j + 1);  // 0..j

    const size_t nnz = (size_t)S.Ap.back();
    S.Ai.resize(nnz);
    S.Ax.resize(nnz);

    for (IntT j = 0; j < (IntT)m; ++j) {
        IntT p = S.Ap[(size_t)j];
        for (IntT i = 0; i <= j; ++i, ++p) {
            S.Ai[(size_t)p] = i;
            S.Ax[(size_t)p] = N_in(i, j);
        }
    }
    linsys::finalize_upper_inplace(S, /*require_diag=*/true);

    try {
        const Symb32 sym = analyze_fast(S);
        auto F = refactorize<FloatT, IntT>(S, sym);

        dvec x = b_in;  // will be overwritten with solution
        solve<FloatT, IntT>(F, x.data());

        // A couple of refinement sweeps (cheap for small m)
        refine<FloatT, IntT>(S, F, x.data(), b_in.data(), /*iters=*/2,
                             static_cast<const Ordering<IntT>*>(nullptr));

        return x;
    } catch (...) {
        // Dense fallbacks (should be rare)
        Eigen::LLT<dmat> llt(N_in);
        if (llt.info() == Eigen::Success) return llt.solve(b_in);
        Eigen::LDLT<dmat> ldlt(N_in);
        if (ldlt.info() == Eigen::Success) return ldlt.solve(b_in);
        return N_in.partialPivLu().solve(b_in);
    }
}
}  // namespace

[[nodiscard]] inline dvec AAT_solve(const dmat& A, const dvec& rhs,
                                    double reg_floor = 1e-12) {
    using namespace qdldl23;
    using FloatT = double;
    using IntT = int32_t;

    struct Cache {
        int rows = -1, cols = -1;
        double fnorm = NAN, dsum = NAN, reg = NAN;
        Symb32 S;
        SparseD32 pattern;  // upper CSC pattern
        bool analyzed = false;
        bool valid = false;
    };
    static thread_local Cache C;

    auto nearly = [](double a, double b, double t = 1e-10) {
        return std::abs(a - b) <=
               t * (1.0 + std::max(std::abs(a), std::abs(b)));
    };

    const int m = (int)A.rows();
    const int n = (int)A.cols();
    (void)n;

    const double reg = std::max(reg_floor, 1e-12);
    const double fnorm = A.norm();
    const double dsum = A.rowwise().squaredNorm().sum();

    const bool reuse = C.valid && C.rows == m && C.cols == (int)A.cols() &&
                       nearly(C.fnorm, fnorm) && nearly(C.dsum, dsum) &&
                       nearly(C.reg, reg, 1e-14);

    if (!reuse) {
        C.rows = m;
        C.cols = (int)A.cols();
        C.fnorm = fnorm;
        C.dsum = dsum;
        C.reg = reg;

        // Build full upper-triangular pattern (with diag)
        SparseD32 P;
        P.n = static_cast<IntT>(m);
        P.Ap.resize((size_t)m + 1);
        P.Ap[0] = 0;
        for (IntT j = 0; j < (IntT)m; ++j)
            P.Ap[(size_t)j + 1] = P.Ap[(size_t)j] + (j + 1);
        const size_t nnz = (size_t)P.Ap[(size_t)m];
        P.Ai.resize(nnz);
        P.Ax.resize(nnz);
        for (IntT j = 0; j < (IntT)m; ++j) {
            IntT p = P.Ap[(size_t)j];
            for (IntT i = 0; i <= j; ++i) P.Ai[(size_t)p++] = i;
        }
        linsys::finalize_upper_inplace(P, /*require_diag=*/true);

        C.S = analyze_fast(P);
        C.pattern = std::move(P);
        C.analyzed = true;
        C.valid = true;
    }

    // M = A*Aᵀ + reg I
    dmat M = A * A.transpose();
    M.diagonal().array() += reg;

    // Inject values into persistent CSC buffer (upper packed)
    {
        auto& Ap = C.pattern.Ap;
        auto& Ai = C.pattern.Ai;
        auto& Ax = C.pattern.Ax;
        for (int j = 0; j < m; ++j) {
            int p = Ap[(size_t)j];
            for (int i = 0; i <= j; ++i, ++p) Ax[(size_t)p] = M(i, j);
        }
    }

    auto F = qdldl23::refactorize<FloatT, IntT>(C.pattern, C.S);
    dvec x = rhs;
    qdldl23::solve<FloatT, IntT>(F, x.data());
    qdldl23::refine<FloatT, IntT>(
        C.pattern, F, x.data(), rhs.data(), /*iters=*/1,
        static_cast<const qdldl23::Ordering<IntT>*>(nullptr));
    return x;
}

inline void project_tangent_into(const dvec& v, const dmat& Aeq, dvec& out) {
    if (Aeq.size() == 0) {
        out = v;
        return;
    }
    const dvec rhs = Aeq * v;
    dvec y = AAT_solve(Aeq, rhs);
    out.noalias() = v - Aeq.transpose() * y;
}
inline dvec project_tangent(const dvec& v, const dmat& Aeq) {
    dvec out(v.size());
    project_tangent_into(v, Aeq, out);
    return out;
}

