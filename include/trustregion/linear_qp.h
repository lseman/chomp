#pragma once
#include <qp/piqp.h>
#include <optional>

namespace tr_detail {
// Solve the convex linearized QP before trust-radius truncation. Infeasible
// linearizations or failed numerical certificates fall back to normal/tangent
// restoration in the caller. Bounds are represented by sparse coordinate rows.
inline std::optional<Eigen::VectorXd> mixed_linear_qp(
    const Eigen::SparseMatrix<double>& hessian, const Eigen::VectorXd& gradient,
    const std::optional<Eigen::MatrixXd>& equality,
    const std::optional<Eigen::VectorXd>& equality_residual,
    const Eigen::MatrixXd& inequality, const Eigen::VectorXd& inequality_residual,
    const Eigen::VectorXd& lower, const Eigen::VectorXd& upper) {
    const Eigen::Index n=gradient.size();
    std::vector<Eigen::Triplet<double>> entries;
    Eigen::Index count=inequality.rows();
    for(Eigen::Index j=0;j<n;++j) {
        if(std::isfinite(lower[j])) ++count;
        if(std::isfinite(upper[j])) ++count;
    }
    Eigen::VectorXd rhs(count);
    rhs.head(inequality.rows())=-inequality_residual;
    for(Eigen::Index i=0;i<inequality.rows();++i)
        for(Eigen::Index j=0;j<n;++j)
            if(inequality(i,j)!=0) entries.emplace_back(i,j,inequality(i,j));
    Eigen::Index row=inequality.rows();
    for(Eigen::Index j=0;j<n;++j) {
        if(std::isfinite(lower[j])) {entries.emplace_back(row,j,-1);rhs[row++]=-lower[j];}
        if(std::isfinite(upper[j])) {entries.emplace_back(row,j,1);rhs[row++]=upper[j];}
    }
    Eigen::SparseMatrix<double> constraints(count,n);
    constraints.setFromTriplets(entries.begin(),entries.end());
    std::optional<Eigen::SparseMatrix<double>> eq;
    std::optional<Eigen::VectorXd> target;
    if(equality && equality->rows()) {
        eq=equality->sparseView();
        target=-equality_residual.value_or(Eigen::VectorXd::Zero(equality->rows()));
    }
    piqp::PIQPSettings settings;
    settings.eps_abs=1e-10;settings.eps_rel=1e-10;settings.max_iter=100;
    piqp::piqp_solver solver(settings);
    try {
        solver.setup(hessian,gradient,eq,target,constraints,rhs);
        auto result=solver.solve();
        if(result.status!="solved" || !result.x.allFinite() || !result.z.allFinite() || !result.y.allFinite())
            return std::nullopt;
        const auto& p=result.x;
        const Eigen::VectorXd slack=rhs-constraints*p;
        Eigen::VectorXd stationarity=hessian*p+gradient+constraints.transpose()*result.z;
        if(eq) stationarity.noalias()+=eq->transpose()*result.y;
        const double scale=std::max(1.0,gradient.lpNorm<Eigen::Infinity>());
        if(!stationarity.allFinite() || !slack.allFinite() ||
           stationarity.lpNorm<Eigen::Infinity>()>1e-7*scale ||
           (slack.array() < -1e-8).any() || (result.z.array() < 0).any() ||
           (slack.array()*result.z.array()).abs().maxCoeff()>1e-7*scale ||
           (eq && ((*eq)*p-*target).lpNorm<Eigen::Infinity>()>1e-8)) return std::nullopt;
        return p;
    } catch(const std::exception&) {return std::nullopt;}
}
}
