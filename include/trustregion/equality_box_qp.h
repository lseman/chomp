#pragma once
#include "box_qp.h"
#include <Eigen/SVD>

namespace tr_detail {
// Truncated working-set CG on A*p=0 and a box containing zero. Active
// coordinate normals join the equality projector; a bound with a negative
// face multiplier is released after convergence on that face.
template<class Apply>
BoxQPResult equality_box_truncated_cg(const Apply& apply,
    const Eigen::VectorXd& g, const Eigen::MatrixXd& a,
    const Eigen::VectorXd& lower, const Eigen::VectorXd& upper,
    double radius, double tolerance, int max_iterations,
    const Eigen::MatrixXd* metric = nullptr,
    const Eigen::VectorXd* normal_step = nullptr) {
    using V = Eigen::VectorXd;
    using M = Eigen::MatrixXd;
    const Eigen::Index n = g.size();
    if (a.cols()!=n || lower.size()!=n || upper.size()!=n ||
        !a.allFinite() || !g.allFinite() || !(radius>0) ||
        (lower.array()>0).any() || (upper.array()<0).any())
        throw std::invalid_argument("equality box QP: invalid input");
    V p=V::Zero(n), hp(n), gradient=g, r(n), d(n), hd(n);
    std::vector<int> active(n,0);
    for (Eigen::Index j=0;j<n;++j) if (lower[j]==upper[j]) active[j]=2;
    M normals, basis;
    Eigen::JacobiSVD<M> svd;
    auto project = [&](const V& v) -> V {
        return v - basis*(basis.transpose()*v);
    };
    auto restart = [&] {
        Eigen::Index count=a.rows();
        for (int side:active) if(side) ++count;
        normals.resize(n,count);
        normals.leftCols(a.rows())=a.transpose();
        Eigen::Index col=a.rows();
        for(Eigen::Index j=0;j<n;++j) if(active[j]) {
            normals.col(col).setZero();
            normals(j,col++)=active[j]==-1 ? -1.0:1.0;
        }
        if(count) {
            svd.compute(normals,Eigen::ComputeThinU|Eigen::ComputeThinV);
            svd.setThreshold(1e-12);
            basis=svd.matrixU().leftCols(svd.rank());
        } else basis.resize(n,0);
        apply(p,hp); gradient=g+hp;
        r=-project(gradient); d=r;
        return r.squaredNorm();
    };
    double rr=restart();
    for(int it=0;it<max_iterations;++it) {
        if(r.lpNorm<Eigen::Infinity>()<=tolerance) {
            rr=restart();
            if(r.lpNorm<Eigen::Infinity>()>tolerance) continue;
            V dual=normals.cols() ? V(svd.solve(-gradient)):V(0);
            Eigen::Index col=a.rows(), release=-1;
            double worst=-tolerance;
            for(Eigen::Index j=0;j<n;++j) if(active[j]) {
                if(active[j]!=2 && dual[col]<worst) {worst=dual[col];release=j;}
                ++col;
            }
            if(release<0) return {p,it,BoxQPStatus::Optimal};
            active[release]=0; rr=restart(); continue;
        }
        apply(d,hd);
        const double curvature=d.dot(hd);
        const double cg=curvature>0 ? rr/curvature:std::numeric_limits<double>::infinity();
        const V md=metric ? V(*metric*d):d;
        const V total = normal_step ? V(p + *normal_step) : p;
        const double qa=d.dot(md), qb=total.dot(md);
        const double qc=std::min(0.0,(metric ? total.dot(*metric*total):total.squaredNorm())-radius*radius);
        const double root=std::sqrt(std::max(0.0,qb*qb-qa*qc));
        const double ball=qa>0 ? (qb>0 ? -qc/(qb+root):(-qb+root)/qa)
            :std::numeric_limits<double>::infinity();
        double bound=std::numeric_limits<double>::infinity();
        Eigen::Index block=-1;int side=0;
        for(Eigen::Index j=0;j<n;++j) if(!active[j]) {
            double candidate=std::numeric_limits<double>::infinity();int s=0;
            if(d[j]<0 && std::isfinite(lower[j])) {candidate=(lower[j]-p[j])/d[j];s=-1;}
            if(d[j]>0 && std::isfinite(upper[j])) {candidate=(upper[j]-p[j])/d[j];s=1;}
            if(candidate<bound) {bound=std::max(0.0,candidate);block=j;side=s;}
        }
        const double alpha=std::min({cg,ball,bound});
        if(!std::isfinite(alpha)) throw std::runtime_error("equality box QP: nonfinite step");
        p+=alpha*d;
        if(ball<=std::min(cg,bound)) return {p,it+1,BoxQPStatus::TrustBoundary};
        if(block>=0 && bound<=cg) {
            active[block]=side;
            // Activate simultaneous hits together, especially zero-length
            // hits at the initial face. Repeated SVDs per coordinate would
            // make a block of identical active bounds unnecessarily costly.
            for (Eigen::Index j=0;j<n;++j) if(!active[j]) {
                if (d[j]<0 && p[j]==lower[j]) active[j]=-1;
                if (d[j]>0 && p[j]==upper[j]) active[j]=1;
            }
            // The step already lies on the face. Avoid coordinate clipping,
            // which would perturb the homogeneous equality residual.
            rr=restart();continue;
        }
        hp.noalias() += alpha * hd;
        r=-project((g+hp).eval());
        const double next=r.squaredNorm();
        d=project((r+(next/std::max(rr,std::numeric_limits<double>::min()))*d).eval());
        rr=next;
    }
    return {p,max_iterations,BoxQPStatus::IterationLimit};
}
}
