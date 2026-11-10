#include <trustregion/box_qp.h>
#include <trustregion/equality_box_qp.h>
#include <trustregion/linear_qp.h>
#include <Eigen/Cholesky>
#include <cassert>
#include <random>

static tr_detail::BoxQPResult solve(const Eigen::MatrixXd& h, const Eigen::VectorXd& g,
                                    const Eigen::VectorXd& lower, const Eigen::VectorXd& upper,
                                    double radius = 10.0, const Eigen::MatrixXd* metric = nullptr) {
    return tr_detail::box_truncated_cg(
        [&](const Eigen::VectorXd& x, Eigen::VectorXd& y) { y.noalias() = h * x; },
        g, lower, upper, h.diagonal(), radius, 1e-11, 1000, metric);
}

int main() {
    Eigen::Matrix2d h;
    Eigen::Vector2d g, lower, upper;
    h << 4.0, 2.0, 2.0, 2.0;
    g << -8.0, -3.0;
    lower.setConstant(-std::numeric_limits<double>::infinity());
    upper << 0.5, std::numeric_limits<double>::infinity();
    auto result = solve(h, g, lower, upper);
    assert(result.status == tr_detail::BoxQPStatus::Optimal);
    assert((result.step - Eigen::Vector2d(0.5, 1.0)).norm() < 1e-10);

    // A lower bound initially has a positive multiplier, then must be
    // released after another variable moves through a negative cross term.
    h << 2.0, -1.5, -1.5, 2.0;
    g << -1.0, 0.1;
    lower.setZero(); upper.setConstant(10.0);
    result = solve(h, g, lower, upper);
    assert(result.status == tr_detail::BoxQPStatus::Optimal);
    assert((result.step - h.ldlt().solve(-g)).norm() < 1e-10);
    const Eigen::MatrixXd metric = Eigen::Vector2d(2.0, 1.0).asDiagonal();
    result = solve(h, g, lower, upper, 0.1, &metric);
    assert(result.status == tr_detail::BoxQPStatus::TrustBoundary);
    assert(std::abs(result.step.dot(metric * result.step) - 0.01) < 1e-14);

    // Compare random convex QPs against exhaustive active-set enumeration.
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> random(-2.0, 2.0);
    for (int trial = 0; trial < 50; ++trial) {
        Eigen::Matrix3d r;
        Eigen::Vector3d gradient;
        for (int i = 0; i < 3; ++i) {
            gradient[i] = random(rng);
            for (int j = 0; j < 3; ++j) r(i, j) = random(rng);
        }
        const Eigen::Matrix3d H = r.transpose() * r + 0.25 * Eigen::Matrix3d::Identity();
        const Eigen::Vector3d lo(-0.5, 0.0, -1.0), hi(0.0, 1.0, 0.7);
        Eigen::Vector3d reference;
        double best = std::numeric_limits<double>::infinity();
        for (int state = 0; state < 27; ++state) {
            Eigen::Vector3d p = Eigen::Vector3d::Zero();
            std::vector<int> free;
            int code = state;
            for (int i = 0; i < 3; ++i, code /= 3) {
                if (code % 3 == 0) free.push_back(i);
                else p[i] = code % 3 == 1 ? lo[i] : hi[i];
            }
            const Eigen::Vector3d rhs = -gradient - H * p;
            if (!free.empty()) {
                Eigen::MatrixXd reduced(free.size(), free.size());
                Eigen::VectorXd b(free.size());
                for (size_t i = 0; i < free.size(); ++i) {
                    b[i] = rhs[free[i]];
                    for (size_t j = 0; j < free.size(); ++j) reduced(i, j) = H(free[i], free[j]);
                }
                const Eigen::VectorXd solution = reduced.ldlt().solve(b);
                for (size_t i = 0; i < free.size(); ++i) p[free[i]] = solution[i];
            }
            if ((p.array() < lo.array() - 1e-12).any() || (p.array() > hi.array() + 1e-12).any()) continue;
            const double value = gradient.dot(p) + 0.5 * p.dot(H * p);
            if (value < best) { best = value; reference = p; }
        }
        const auto got = solve(H, gradient, lo, hi);
        assert(got.status == tr_detail::BoxQPStatus::Optimal);
        assert((got.step - reference).norm() < 1e-8);
    }

    // Equality tangent with an active upper bound and an unrelated free axis.
    Eigen::MatrixXd H = Eigen::MatrixXd::Identity(3,3)*2;
    Eigen::MatrixXd A(1,3); A << 1,1,0;
    Eigen::VectorXd gradient(3), lo(3), hi(3);
    gradient << -4,2,-6;
    lo.setConstant(-10); hi << 0,10,10;
    auto tangent = tr_detail::equality_box_truncated_cg(
        [&](const Eigen::VectorXd& v,Eigen::VectorXd& out){out=H*v;},
        gradient,A,lo,hi,10,1e-11,1000);
    assert(tangent.status==tr_detail::BoxQPStatus::Optimal);
    assert((tangent.step-Eigen::Vector3d(0,0,3)).norm()<1e-9);

    // Independent exhaustive active-face KKT reference for convex QPs.
    for(int sample=0;sample<50;++sample) {
        Eigen::Matrix3d B=Eigen::Matrix3d::Random();
        H=B.transpose()*B+Eigen::Matrix3d::Identity();
        gradient=Eigen::Vector3d::Random()*3;
        lo.setConstant(-0.5);hi.setConstant(0.5);
        double best=std::numeric_limits<double>::infinity();
        for(int mask=0;mask<27;++mask) {
            int code=mask,rows=1;
            std::vector<int> face(3);
            for(int j=0;j<3;++j){face[j]=code%3;code/=3;if(face[j])++rows;}
            Eigen::MatrixXd C=Eigen::MatrixXd::Zero(rows,3);
            Eigen::VectorXd target=Eigen::VectorXd::Zero(rows);
            C.row(0)=A;int row=1;
            for(int j=0;j<3;++j)if(face[j]){C(row,j)=1;target[row++]=face[j]==1?lo[j]:hi[j];}
            Eigen::MatrixXd K=Eigen::MatrixXd::Zero(3+rows,3+rows);
            K.topLeftCorner(3,3)=H;K.topRightCorner(3,rows)=C.transpose();K.bottomLeftCorner(rows,3)=C;
            Eigen::VectorXd rhs(3+rows);rhs.head(3)=-gradient;rhs.tail(rows)=target;
            Eigen::VectorXd sol=K.jacobiSvd(Eigen::ComputeFullU|Eigen::ComputeFullV).solve(rhs);
            if((K*sol-rhs).norm()>1e-8)continue;
            Eigen::VectorXd p=sol.head(3);
            if((p.array()<lo.array()-1e-9).any()||(p.array()>hi.array()+1e-9).any())continue;
            best=std::min(best,gradient.dot(p)+0.5*p.dot(H*p));
        }
        tangent=tr_detail::equality_box_truncated_cg(
            [&](const Eigen::VectorXd& v,Eigen::VectorXd& out){out=H*v;},
            gradient,A,lo,hi,10,1e-10,1000);
        assert(tangent.status==tr_detail::BoxQPStatus::Optimal);
        assert((A*tangent.step).norm()<1e-9);
        assert((tangent.step.array()>=lo.array()-1e-9).all());
        assert((tangent.step.array()<=hi.array()+1e-9).all());
        assert(std::abs(gradient.dot(tangent.step)+0.5*tangent.step.dot(H*tangent.step)-best)<1e-8);
    }

    // The trust ball is on normal+tangent, even when they are not orthogonal.
    H=Eigen::Matrix3d::Identity();gradient=Eigen::Vector3d(0,0,-6);
    lo.setConstant(-10);hi.setConstant(10);
    Eigen::VectorXd normal=Eigen::Vector3d(0,0,0.5);
    tangent=tr_detail::equality_box_truncated_cg(
        [&](const Eigen::VectorXd& v,Eigen::VectorXd& out){out=H*v;},
        gradient,A,lo,hi,1,1e-10,1000,nullptr,&normal);
    assert(tangent.status==tr_detail::BoxQPStatus::TrustBoundary);
    assert(std::abs((normal+tangent.step).norm()-1)<1e-10);

    // Redundant equality rows must not change the tangent projector.
    Eigen::MatrixXd redundant(2,3); redundant << 1,1,0, 2,2,0;
    H=Eigen::Matrix3d::Identity()*2;gradient=Eigen::Vector3d(-4,2,-6);
    hi << 0,10,10;
    tangent=tr_detail::equality_box_truncated_cg(
        [&](const Eigen::VectorXd& v,Eigen::VectorXd& out){out=H*v;},
        gradient,redundant,lo,hi,10,1e-10,1000);
    assert(tangent.status==tr_detail::BoxQPStatus::Optimal);
    assert((tangent.step-Eigen::Vector3d(0,0,3)).norm()<1e-9);
    // Negative curvature should truncate at the ball within the nullspace.
    H(2,2)=-2;gradient=Eigen::Vector3d(0,0,-1);
    tangent=tr_detail::equality_box_truncated_cg(
        [&](const Eigen::VectorXd& v,Eigen::VectorXd& out){out=H*v;},
        gradient,redundant,lo,hi,1,1e-10,1000);
    assert(tangent.status==tr_detail::BoxQPStatus::TrustBoundary);
    assert(std::abs(tangent.step.norm()-1)<1e-10);
    assert((redundant*tangent.step).norm()<1e-10);

    Eigen::SparseMatrix<double> qp_hessian(2,2);qp_hessian.setIdentity();
    Eigen::VectorXd qp_gradient(2);qp_gradient << -2,-3;
    Eigen::MatrixXd qp_eq(1,2), qp_ineq(1,2);
    qp_eq << 1,1;qp_ineq << 0,1;
    Eigen::VectorXd qp_eq_res(1),qp_ineq_res(1),qp_lo(2),qp_hi(2);
    qp_eq_res << -1;qp_ineq_res << -1;
    qp_lo.setConstant(-10);qp_hi << 0,10;
    auto mixed=tr_detail::mixed_linear_qp(qp_hessian,qp_gradient,qp_eq,qp_eq_res,
        qp_ineq,qp_ineq_res,qp_lo,qp_hi);
    assert(mixed && (*mixed-Eigen::Vector2d(0,1)).norm()<1e-6);
    // An infeasible linearization must leave the restoration fallback available.
    qp_ineq=qp_eq;qp_ineq_res << 0;
    mixed=tr_detail::mixed_linear_qp(qp_hessian,qp_gradient,qp_eq,qp_eq_res,
        qp_ineq,qp_ineq_res,qp_lo,qp_hi);
    assert(!mixed);
}
