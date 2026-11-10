#include <trustregion/least_squares.h>
#include <cassert>
#include <random>

int main() {
    Eigen::MatrixXd a(1, 2);
    Eigen::VectorXd b(1);
    a << 1.0, 1.0;
    b << 1.0;
    auto x = tr_detail::regularized_least_squares(a, b, 1e-12);
    assert((x.array() - 1.0 / (2.0 + 1e-12)).abs().maxCoeff() < 1e-14);
    assert(std::abs(x[0] - x[1]) < 1e-15);

    a.resize(2, 2);
    b.resize(2);
    a << 1.0, 1.0, 2.0, 2.0;
    b << 1.0, 0.0;
    x = tr_detail::regularized_least_squares(a, b, 1e-8);
    assert((x.array() - 1.0 / (10.0 + 1e-8)).abs().maxCoeff() < 1e-12);

    a.resize(2, 1);
    a << 1.0, 1.0;
    b << 1.0, 1.0;
    x = tr_detail::regularized_least_squares(a, b, 0.1);
    assert(std::abs(x[0] - 2.0 / 2.1) < 1e-14);

    a.resize(1, 1);
    b.resize(1);
    a << 1e150;
    b << 1.0;
    x = tr_detail::regularized_least_squares(a, b, 1e-12);
    assert(x.allFinite());
    assert(std::abs(x[0] * 1e150 - 1.0) < 1e-14);

    a.resize(0, 3);
    b.resize(0);
    x = tr_detail::regularized_least_squares(a, b, 1e-12);
    assert(x.size() == 3 && x.norm() == 0.0);

    // Clipping the unconstrained solution leaves the second column wrong.
    a.resize(2, 2); b.resize(2);
    a << 1, 1, 0, 1; b << -1, 1;
    x = tr_detail::mixed_nonnegative_least_squares(a, b, 0);
    assert(x.norm() < 1e-10);
    // Free equality dual and positive upper-bound dual: nu=-2, zU=6.
    a << 1, 1, 1, 0; b << 4, -2;
    x = tr_detail::mixed_nonnegative_least_squares(a, b, 1);
    assert(std::abs(x[0] + 2) < 1e-9);
    assert(std::abs(x[1] - 6) < 1e-9);
    // Opposite normals for a fixed variable must recover the full dual.
    a.resize(1, 2); b.resize(1); a << -1, 1; b << 4;
    x = tr_detail::mixed_nonnegative_least_squares(a, b, 0);
    assert(x[0] == 0 && std::abs(x[1] - 4) < 1e-10);

    std::mt19937 rng(42);
    std::normal_distribution<double> normal;
    for (int sample = 0; sample < 100; ++sample) {
        a.resize(4, 4); b.resize(4);
        for (int j = 0; j < a.size(); ++j) a.data()[j] = normal(rng);
        for (int j = 0; j < b.size(); ++j) b[j] = normal(rng);
        x = tr_detail::mixed_nonnegative_least_squares(a, b, 1);
        double best = std::numeric_limits<double>::infinity();
        // Independent enumeration of all faces, one free and three NN duals.
        for (int mask = 0; mask < 8; ++mask) {
            std::vector<int> cols{0};
            for (int j = 0; j < 3; ++j) if (mask & (1 << j)) cols.push_back(j+1);
            Eigen::MatrixXd face(4, cols.size());
            for (int j = 0; j < (int)cols.size(); ++j) face.col(j) = a.col(cols[j]);
            Eigen::VectorXd y = tr_detail::regularized_least_squares(face, b, 1e-12);
            if (y.size() > 1 && y.tail(y.size()-1).minCoeff() < 0) continue;
            best = std::min(best, (face*y-b).squaredNorm() + 1e-12*y.squaredNorm());
        }
        assert(x.tail(3).minCoeff() >= 0);
        assert(std::abs((a*x-b).squaredNorm()+1e-12*x.squaredNorm()-best) < 1e-8);
    }
}
