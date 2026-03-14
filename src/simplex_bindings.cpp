// simplex_bindings.cpp
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cctype>

#include "../include/simplex/simplex.h"

namespace py = pybind11;

static std::string status_to_str(const LPSolution::Status s) {
    return to_string(s);
}

// Convert LPSolution to a Python-facing object with .status as string
struct PyLPSolution {
    std::string status;  // 'optimal', 'unbounded', ...
    Eigen::VectorXd x;
    double obj;
    std::vector<int> basis;
    int iters;
    std::unordered_map<std::string, std::string> info;
    Eigen::VectorXd farkas_y;
    bool farkas_has_cert;

    static PyLPSolution from_cpp(const LPSolution& s) {
        return PyLPSolution{status_to_str(s.status),
                            s.x,
                            s.obj,
                            s.basis,
                            s.iters,
                            s.info,
                            s.farkas_y,
                            s.farkas_has_cert};
    }
};

static std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) {
                       return static_cast<char>(std::tolower(ch));
                   });
    return value;
}

static SimplexMode parse_mode(const py::handle& obj) {
    if (py::isinstance<py::str>(obj)) {
        const std::string value = lowercase(py::cast<std::string>(obj));
        if (value == "primal") return SimplexMode::Primal;
        if (value == "dual") return SimplexMode::Dual;
        return SimplexMode::Auto;
    }
    return py::cast<SimplexMode>(obj);
}

static RevisedSimplexOptions make_opts_from_kwargs(const py::kwargs& kw) {
    RevisedSimplexOptions o;
    auto get = [&](const char* k) -> py::object {
        return kw.contains(k) ? py::reinterpret_borrow<py::object>(kw[k])
                              : py::none();
    };

    if (!get("max_iters").is_none())
        o.max_iters = py::cast<int>(kw["max_iters"]);
    if (!get("tol").is_none()) o.tol = py::cast<double>(kw["tol"]);
    if (!get("bland").is_none()) o.bland = py::cast<bool>(kw["bland"]);
    if (!get("svd_tol").is_none()) o.svd_tol = py::cast<double>(kw["svd_tol"]);
    if (!get("ratio_delta").is_none())
        o.ratio_delta = py::cast<double>(kw["ratio_delta"]);
    if (!get("ratio_eta").is_none())
        o.ratio_eta = py::cast<double>(kw["ratio_eta"]);
    if (!get("deg_step_tol").is_none())
        o.deg_step_tol = py::cast<double>(kw["deg_step_tol"]);
    if (!get("epsilon_cost").is_none())
        o.epsilon_cost = py::cast<double>(kw["epsilon_cost"]);
    if (!get("rng_seed").is_none()) o.rng_seed = py::cast<int>(kw["rng_seed"]);
    if (!get("refactor_every").is_none())
        o.refactor_every = py::cast<int>(kw["refactor_every"]);
    if (!get("compress_every").is_none())
        o.compress_every = py::cast<int>(kw["compress_every"]);
    if (!get("lu_pivot_rel").is_none())
        o.lu_pivot_rel = py::cast<double>(kw["lu_pivot_rel"]);
    if (!get("lu_abs_floor").is_none())
        o.lu_abs_floor = py::cast<double>(kw["lu_abs_floor"]);
    if (!get("alpha_tol").is_none())
        o.alpha_tol = py::cast<double>(kw["alpha_tol"]);
    if (!get("z_inf_guard").is_none())
        o.z_inf_guard = py::cast<double>(kw["z_inf_guard"]);
    if (!get("basis_update").is_none())
        o.basis_update = py::cast<std::string>(kw["basis_update"]);
    if (!get("ft_bandwidth_cap").is_none())
        o.ft_bandwidth_cap = py::cast<int>(kw["ft_bandwidth_cap"]);
    if (!get("devex_reset").is_none())
        o.devex_reset = py::cast<int>(kw["devex_reset"]);
    if (!get("pricing_rule").is_none())
        o.pricing_rule = py::cast<std::string>(kw["pricing_rule"]);
    if (!get("adaptive_reset_freq").is_none())
        o.adaptive_reset_freq = py::cast<int>(kw["adaptive_reset_freq"]);
    if (!get("max_basis_rebuilds").is_none())
        o.max_basis_rebuilds = py::cast<int>(kw["max_basis_rebuilds"]);
    if (!get("dual_allow_bound_flip").is_none())
        o.dual_allow_bound_flip =
            py::cast<bool>(kw["dual_allow_bound_flip"]);
    if (!get("dual_flip_pivot_tol").is_none())
        o.dual_flip_pivot_tol = py::cast<double>(kw["dual_flip_pivot_tol"]);
    if (!get("dual_flip_rc_tol").is_none())
        o.dual_flip_rc_tol = py::cast<double>(kw["dual_flip_rc_tol"]);
    if (!get("dual_flip_max_per_iter").is_none())
        o.dual_flip_max_per_iter =
            py::cast<int>(kw["dual_flip_max_per_iter"]);
    if (!get("mode").is_none()) o.mode = parse_mode(kw["mode"]);
    return o;
}

PYBIND11_MODULE(simplex_core, m) {
    m.doc() = "Revised Simplex (modern C++) Python bindings";

    py::enum_<SimplexMode>(m, "SimplexMode")
        .value("Auto", SimplexMode::Auto)
        .value("Primal", SimplexMode::Primal)
        .value("Dual", SimplexMode::Dual);

    // Expose options (nice for discoverability)
    py::class_<RevisedSimplexOptions>(m, "RevisedSimplexOptions")
        .def(py::init<>())
        .def_readwrite("max_iters", &RevisedSimplexOptions::max_iters)
        .def_readwrite("tol", &RevisedSimplexOptions::tol)
        .def_readwrite("bland", &RevisedSimplexOptions::bland)
        .def_readwrite("svd_tol", &RevisedSimplexOptions::svd_tol)
        .def_readwrite("ratio_delta", &RevisedSimplexOptions::ratio_delta)
        .def_readwrite("ratio_eta", &RevisedSimplexOptions::ratio_eta)
        .def_readwrite("deg_step_tol", &RevisedSimplexOptions::deg_step_tol)
        .def_readwrite("epsilon_cost", &RevisedSimplexOptions::epsilon_cost)
        .def_readwrite("rng_seed", &RevisedSimplexOptions::rng_seed)
        .def_readwrite("refactor_every", &RevisedSimplexOptions::refactor_every)
        .def_readwrite("compress_every", &RevisedSimplexOptions::compress_every)
        .def_readwrite("lu_pivot_rel", &RevisedSimplexOptions::lu_pivot_rel)
        .def_readwrite("lu_abs_floor", &RevisedSimplexOptions::lu_abs_floor)
        .def_readwrite("alpha_tol", &RevisedSimplexOptions::alpha_tol)
        .def_readwrite("z_inf_guard", &RevisedSimplexOptions::z_inf_guard)
        .def_readwrite("basis_update", &RevisedSimplexOptions::basis_update)
        .def_readwrite("ft_bandwidth_cap",
                       &RevisedSimplexOptions::ft_bandwidth_cap)
        .def_readwrite("devex_reset", &RevisedSimplexOptions::devex_reset)
        .def_readwrite("pricing_rule", &RevisedSimplexOptions::pricing_rule)
        .def_readwrite("adaptive_reset_freq",
                       &RevisedSimplexOptions::adaptive_reset_freq)
        .def_readwrite("max_basis_rebuilds",
                       &RevisedSimplexOptions::max_basis_rebuilds)
        .def_readwrite("dual_allow_bound_flip",
                       &RevisedSimplexOptions::dual_allow_bound_flip)
        .def_readwrite("dual_flip_pivot_tol",
                       &RevisedSimplexOptions::dual_flip_pivot_tol)
        .def_readwrite("dual_flip_rc_tol",
                       &RevisedSimplexOptions::dual_flip_rc_tol)
        .def_readwrite("dual_flip_max_per_iter",
                       &RevisedSimplexOptions::dual_flip_max_per_iter)
        .def_readwrite("mode", &RevisedSimplexOptions::mode);

    // Python-facing LPSolution (status is string)
    py::class_<PyLPSolution>(m, "LPSolution")
        .def_property_readonly("status",
                               [](const PyLPSolution& s) { return s.status; })
        .def_readonly("x", &PyLPSolution::x)
        .def_readonly("obj", &PyLPSolution::obj)
        .def_readonly("basis", &PyLPSolution::basis)
        .def_readonly("iters", &PyLPSolution::iters)
        .def_readonly("info", &PyLPSolution::info)
        .def_readonly("farkas_y", &PyLPSolution::farkas_y)
        .def_readonly("farkas_has_cert", &PyLPSolution::farkas_has_cert);

    // RevisedSimplex class
    py::class_<RevisedSimplex>(m, "RevisedSimplex")
        // __init__(**kwargs) optional; defaults match C++
        .def(py::init([&](const py::kwargs& kw) {
                 if (kw.size() == 0) return RevisedSimplex{};
                 return RevisedSimplex{make_opts_from_kwargs(kw)};
             }),
             R"doc(
            RevisedSimplex(**kwargs)
            Keyword options include:
              - max_iters, tol, bland, svd_tol, ratio_delta, ratio_eta,
                deg_step_tol, epsilon_cost, rng_seed, refactor_every,
                compress_every, lu_pivot_rel, lu_abs_floor, alpha_tol,
                z_inf_guard, basis_update, ft_bandwidth_cap, devex_reset,
                pricing_rule, adaptive_reset_freq, max_basis_rebuilds,
                dual_allow_bound_flip, dual_flip_pivot_tol,
                dual_flip_rc_tol, dual_flip_max_per_iter, mode
        )doc")
        // solve(A, b, c, basis=None) -> LPSolution
        .def(
            "solve",
            [](RevisedSimplex& self, const Eigen::MatrixXd& A,
               const Eigen::VectorXd& b, const Eigen::VectorXd& c,
               py::object basis_opt) -> PyLPSolution {
                std::optional<std::vector<int>> basis = std::nullopt;
                if (!basis_opt.is_none()) {
                    basis = py::cast<std::vector<int>>(basis_opt);
                }
                LPSolution raw = self.solve(A, b, c, basis);
                return PyLPSolution::from_cpp(raw);
            },
            py::arg("A"), py::arg("b"), py::arg("c"),
            py::arg("basis") = py::none(),
            R"doc(
                Solve standard-form LP:
                    min c^T x
                    s.t. A x = b, x >= 0

                Returns LPSolution with fields:
                    status (str), x (np.ndarray), obj (float),
                    basis (List[int]), iters (int), info (dict),
                    farkas_y (np.ndarray), farkas_has_cert (bool)
            )doc")
        .def(
            "solve",
            [](RevisedSimplex& self, const Eigen::MatrixXd& A,
               const Eigen::VectorXd& b, const Eigen::VectorXd& c,
               const Eigen::VectorXd& l, const Eigen::VectorXd& u,
               py::object basis_opt) -> PyLPSolution {
                std::optional<std::vector<int>> basis = std::nullopt;
                if (!basis_opt.is_none()) {
                    basis = py::cast<std::vector<int>>(basis_opt);
                }
                LPSolution raw = self.solve(A, b, c, l, u, basis);
                return PyLPSolution::from_cpp(raw);
            },
            py::arg("A"), py::arg("b"), py::arg("c"), py::arg("l"),
            py::arg("u"), py::arg("basis") = py::none(),
            R"doc(
                Solve bounded equality-form LP:
                    min c^T x
                    s.t. A x = b
                         l <= x <= u

                Variables must have at least one finite bound.
            )doc");
}
