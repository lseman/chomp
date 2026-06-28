// ad.cpp — nanobind + NumPy fast paths + GIL release (C++23, cleaned & de-duplicated)
#pragma once

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <memory_resource>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ad_graph.h"
#include "definitions.h"
#include "expression.h"
#include "operators.h" // enum Operator + math free-functions
#include "variable.h"

// If you use tsl::robin_map, include it here:
#include <robin_map.h>
#include <robin_set.h>

namespace nb = nanobind;
using namespace nb::literals;
using dvec  = Eigen::VectorXd;
using dmat  = Eigen::MatrixXd;
using spmat = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;

// ---------- Memory pool for hot allocations ----------
static std::pmr::unsynchronized_pool_resource g_pool{
    {.max_blocks_per_chunk = 256, .largest_required_pool_block = 8192}};
thread_local std::pmr::vector<double> tl_scratch{&g_pool};

// ---------- Concepts ----------
template <typename T>
concept Numeric = std::is_arithmetic_v<T>;

template <typename T>
concept PyHandle = std::same_as<T, nb::handle> || std::same_as<T, nb::object>;

// ---------- ndarray aliases ----------
using Arr1D = nb::ndarray<double, nb::numpy, nb::shape<-1>, nb::c_contig>;
using Arr2D = nb::ndarray<double, nb::numpy, nb::shape<-1, -1>, nb::c_contig>;

// ---------- small helpers ----------
[[gnu::always_inline, gnu::hot]]
static inline bool is_number(const nb::handle &h) noexcept {
    return nb::isinstance<nb::float_>(h) || nb::isinstance<nb::int_>(h);
}

[[gnu::always_inline, gnu::hot]]
static inline bool is_sequence(const nb::handle &h) noexcept {
    return nb::isinstance<nb::list>(h) || nb::isinstance<nb::tuple>(h);
}

[[gnu::always_inline, gnu::hot]]
static inline std::span<const double> as_span_1d(const Arr1D &a) {
    if (a.ndim() != 1) [[unlikely]]
        throw std::invalid_argument("expected 1D float64 array");
    return {a.data(), (size_t)a.shape(0)};
}

[[gnu::always_inline]]
static inline std::pair<ssize_t, ssize_t> shape_2d(const Arr2D &a) {
    if (a.ndim() != 2) [[unlikely]]
        throw std::invalid_argument("expected 2D float64 array");
    return {a.shape(0), a.shape(1)};
}

[[gnu::always_inline]]
static inline Arr1D create_zeros_1d(ssize_t n) {
    auto np = nb::module_::import_("numpy");
    return nb::cast<Arr1D>(np.attr("zeros")(nb::make_tuple(n), "dtype"_a = "float64"));
}

[[gnu::always_inline]]
static inline Arr2D create_zeros_2d(ssize_t m, ssize_t n) {
    auto np = nb::module_::import_("numpy");
    return nb::cast<Arr2D>(np.attr("zeros")(nb::make_tuple(m, n), "dtype"_a = "float64"));
}

// Allocate an *uninitialized* owning numpy array directly in C++ (no Python
// numpy.zeros round-trip, no wasted zero-fill). The caller is responsible for
// writing every element. A capsule frees the buffer when the array dies. The
// nb::numpy framework tag (already on Arr1D/Arr2D) makes nanobind export an
// actual numpy.ndarray. Out1D/Out2D are semantic aliases marking C++-owned
// outputs.
using Out1D = Arr1D;
using Out2D = Arr2D;

[[gnu::always_inline]]
static inline Out1D create_uninit_1d(ssize_t n) {
    double     *buf = new double[(size_t)n];
    nb::capsule owner(buf, [](void *p) noexcept { delete[] static_cast<double *>(p); });
    return Out1D(buf, {(size_t)n}, owner);
}

[[gnu::always_inline]]
static inline Out2D create_uninit_2d(ssize_t m, ssize_t n) {
    double     *buf = new double[(size_t)m * (size_t)n];
    nb::capsule owner(buf, [](void *p) noexcept { delete[] static_cast<double *>(p); });
    return Out2D(buf, {(size_t)m, (size_t)n}, owner);
}

[[gnu::always_inline]]
static inline nb::tuple to_tuple(const std::vector<nb::object> &vec) {
    nb::list temp;
    for (const auto &item : vec) temp.append(item);
    return nb::tuple(temp);
}

// ---------- Python call helpers ----------
enum class ArgPolicy : uint8_t { Tuple, List };

template <ArgPolicy P, typename Callable>
[[gnu::always_inline]]
static inline nb::object call_py_fn(Callable &&f, const std::vector<nb::object> &args) {
    if constexpr (P == ArgPolicy::Tuple) {
        nb::tuple t = to_tuple(args);
        return f(*t);
    } else {
        nb::list lst;
        for (auto &a : args) lst.append(a);
        return f(lst);
    }
}

// ---------- Input setters ----------
template <typename Vec, typename Span>
[[gnu::always_inline, gnu::hot]]
static inline void set_inputs_from_span(Vec &nodes, Span x) {
    if (x.size() != nodes.size()) [[unlikely]]
        throw std::invalid_argument("wrong input length");
    for (size_t i = 0; i < nodes.size(); ++i) nodes[i]->value = x[i];
}

template <typename Vec>
[[gnu::always_inline, gnu::hot]]
static inline void set_inputs_from_arr(Vec &nodes, const Arr1D &xin) {
    set_inputs_from_span(nodes, as_span_1d(xin));
}

template <typename Vec>
[[gnu::always_inline, gnu::hot]]
static inline void set_inputs_from_seq(Vec &nodes, const nb::object &x) {
    if (!is_sequence(x)) [[unlikely]]
        throw std::invalid_argument("expected a list/tuple");
    nb::sequence sx = nb::cast<nb::sequence>(x);
    if ((size_t)nb::len(sx) != nodes.size()) [[unlikely]]
        throw std::invalid_argument("wrong input length");
    for (size_t i = 0; i < nodes.size(); ++i) nodes[i]->value = nb::cast<double>(sx[i]);
}

// ============================================================================
// Coercion: Python handle -> Expression (delegates constants/vars to Expression)
// ============================================================================

[[gnu::hot]]
static inline std::shared_ptr<Expression> coerce_expr(const nb::handle &h, const ADGraphPtr &prefer = {}) {
    if (nb::isinstance<Expression>(h)) {
        auto e = nb::cast<std::shared_ptr<Expression>>(h);
        if (prefer) prefer->adoptSubgraph(e->node);
        return e;
    }
    if (nb::isinstance<Variable>(h)) {
        auto v = nb::cast<std::shared_ptr<Variable>>(h);
        auto g = prefer ? prefer : std::make_shared<ADGraph>();
        return std::make_shared<Expression>(v, 1.0, g);
    }
    if (is_number(h)) {
        auto g = prefer ? prefer : std::make_shared<ADGraph>();
        // Prefer an explicit constructor/helper if you expose it:
        // return Expression::from_constant(g, nb::cast<double>(h));
        // Fallback: build via Expression arithmetic (0 + s)
        auto zero = std::make_shared<Expression>(g);
        return *zero + nb::cast<double>(h);
    }
    throw std::invalid_argument("Expected Expression, Variable, or number.");
}

// ============================================================================
// Unary/binary dispatchers (Expression-only algebra)
// ============================================================================

static inline std::shared_ptr<Expression> apply_unary_expr(Operator op, const std::shared_ptr<Expression> &x) {
    switch (op) {
    case Operator::Sin: return sin(*x);
    case Operator::Cos: return cos(*x);
    case Operator::Tan: return tan(*x);
    case Operator::Exp: return exp(*x);
    case Operator::Log: return log(*x);
    case Operator::Tanh: return tanh(*x);
    case Operator::Relu: return relu(*x);
    case Operator::Silu: return silu(*x);
    case Operator::Gelu: return gelu(*x);
    default: return x;
    }
}

[[gnu::hot]]
static inline nb::object unary_dispatch(nb::object x, Operator op) {
    if (is_number(x)) [[likely]] {
        const double s = nb::cast<double>(x);
        switch (op) { // keep number→number fast-path
        case Operator::Sin: return nb::float_(std::sin(s));
        case Operator::Cos: return nb::float_(std::cos(s));
        case Operator::Tan: return nb::float_(std::tan(s));
        case Operator::Exp: return nb::float_(std::exp(s));
        case Operator::Log: return nb::float_(std::log(s));
        case Operator::Tanh: return nb::float_(std::tanh(s));
        case Operator::Relu: return nb::float_(s > 0.0 ? s : 0.0);
        case Operator::Silu: return nb::float_(s / (1.0 + std::exp(-s)));
        case Operator::Gelu: {
            constexpr double c = 0.7978845608028654;
            return nb::float_(0.5 * s * (1.0 + std::tanh(c * (s + 0.044715 * s * s * s))));
        }
        default: return nb::float_(s);
        }
    }
    auto e = coerce_expr(x);
    return nb::cast(apply_unary_expr(op, e));
}

enum class BinOp : uint8_t { Add, Sub, Mul, Div };

[[gnu::hot]]
static inline std::shared_ptr<Expression> apply_binary_expr(BinOp bop, const std::shared_ptr<Expression> &a,
                                                            const std::shared_ptr<Expression> &b) {
    switch (bop) {
    case BinOp::Add: return *a + *b;
    case BinOp::Sub: return *a - *b;
    case BinOp::Mul: return *a * *b;
    case BinOp::Div: return *a / *b;
    }
    return *a + *b; // unreachable
}

[[gnu::hot]]
static inline std::shared_ptr<Expression> binary_coerced(BinOp bop, const nb::handle &x, const nb::handle &y) {
    ADGraphPtr g;
    if (nb::isinstance<Expression>(x))
        g = nb::cast<std::shared_ptr<Expression>>(x)->graph;
    else if (nb::isinstance<Expression>(y))
        g = nb::cast<std::shared_ptr<Expression>>(y)->graph;
    if (!g) g = std::make_shared<ADGraph>();
    auto ex = coerce_expr(x, g);
    auto ey = coerce_expr(y, g);
    return apply_binary_expr(bop, ex, ey);
}

// pow dispatch
[[gnu::hot]]
static inline std::shared_ptr<Expression> pow_dispatch(const nb::handle &base, const nb::handle &expn) {
    // number ** number → number
    if (is_number(base) && is_number(expn)) {
        double b = nb::cast<double>(base), p = nb::cast<double>(expn);
        return coerce_expr(nb::float_(std::pow(b, p)));
    }
    // Expression ** number
    if (nb::isinstance<Expression>(base) || nb::isinstance<Variable>(base)) {
        auto b = coerce_expr(base);
        if (is_number(expn)) {
            double p = nb::cast<double>(expn);
            return pow(*b, p); // your Expression free-function
        }
    }
    // number ** Expression
    if (nb::isinstance<Expression>(expn) || nb::isinstance<Variable>(expn)) {
        double s;
        if (is_number(base)) {
            s = nb::cast<double>(base);
            if (s <= 0.0) throw std::domain_error("scalar ** Expression requires base > 0");
            auto e = coerce_expr(expn);
            // If you expose pow(double,const Expression&), call it here.
            // Fallback: s^x = exp(log(s) * x)
            auto g      = e->graph ? e->graph : std::make_shared<ADGraph>();
            auto s_expr = coerce_expr(nb::float_(s), g);
            // new (stepwise, type-safe)
            auto log_s = log(*s_expr); // ExpressionPtr
            auto prod  = *log_s * *e;  // ExpressionPtr
            return exp(*prod);         // exp(const Expression&) -> ExpressionPtr
        }
    }
    throw std::invalid_argument("Unsupported types for pow(base, exp).");
}

// Optional: max(a,b) if you have a free-function `max(const Expression&, const Expression&)`
[[gnu::hot]]
static inline std::shared_ptr<Expression> max_dispatch(const nb::handle &x, const nb::handle &y) {
    if (is_number(x) && is_number(y)) {
        const double a = nb::cast<double>(x), b = nb::cast<double>(y);
        return coerce_expr(nb::float_(a >= b ? a : b));
    }
    auto ex = coerce_expr(x);
    auto ey = coerce_expr(y, ex->graph);
    // Provide this in Expression.cpp:
    // ExpressionPtr max(const Expression& a, const Expression& b);
    return max(*ex, *ey);
}

// ============================================================================
// Compiled builders (Expression-only), GradFn, HessFn
// ============================================================================

struct Compiled {
    ADGraphPtr             g;
    ADNodePtr              root;
    std::vector<ADNodePtr> vars;
};

template <ArgPolicy P>
[[gnu::hot]]
static inline Compiled compile_to_graph(nb::object f, size_t arity, bool vector_input) {
    Compiled out;
    out.g = std::make_shared<ADGraph>();
    std::vector<nb::object> expr_args;
    expr_args.reserve(arity);
    out.vars.reserve(arity);
    for (size_t i = 0; i < arity; ++i) {
        auto v = std::make_shared<Variable>("x" + std::to_string(i), 0.0);
        auto e = std::make_shared<Expression>(v, 1.0, out.g);
        out.vars.push_back(e->node);
        expr_args.emplace_back(nb::cast(e));
    }
    nb::object ret = vector_input ? call_py_fn<ArgPolicy::List>(f, expr_args) : call_py_fn<P>(f, expr_args);
    if (ret.is_none()) [[unlikely]]
        throw std::invalid_argument("compile: function returned None");
    auto expr = nb::cast<std::shared_ptr<Expression>>(ret);
    out.g->adoptSubgraph(expr->node);
    out.root = expr->node;
    return out;
}

class GradFn {
public:
    ADGraphPtr             g;
    ADNodePtr              expr_root;
    std::vector<ADNodePtr> var_nodes;
    bool                   vector_mode{};
    nb::object             python_func;

    [[gnu::pure]] std::string expr_str() const {
        return (g && expr_root) ? g->getExpression(expr_root) : std::string{};
    }

    GradFn(nb::object f, size_t arity, bool vector_input) : vector_mode(vector_input), python_func(f) {
        auto C = compile_to_graph<ArgPolicy::Tuple>(f, arity, vector_input);
        g      = C.g;
        g->simplifyGraph();
        expr_root = C.root;
        var_nodes = std::move(C.vars);
    }

    nb::list operator()(nb::object x) {
        set_inputs_from_seq(var_nodes, x);
        {
            nb::gil_scoped_release nogil;
            g->resetGradients();
            g->computeForwardPass();
            set_epoch_value(expr_root->gradient, expr_root->grad_epoch, g->cur_grad_epoch_, 1.0);
            g->initiateBackwardPass(expr_root);
        }
        nb::list out;
        for (auto &nd : var_nodes) out.append(nb::float_(nd->gradient));
        return out;
    }

    [[gnu::hot]] std::pair<double, Eigen::VectorXd> value_grad_eigen(const Eigen::VectorXd &x_in) {
        // convert x_in to span 1d
        auto x = std::span<const double>(x_in.data(), (size_t)x_in.size());
        if ((size_t)x.size() != var_nodes.size()) [[unlikely]]
            throw std::invalid_argument("GradFn.value_grad: wrong input length");
        for (size_t i = 0; i < var_nodes.size(); ++i) var_nodes[i]->value = x[i]; // implicit cast double <- scalar

        double fval;
        {
            nb::gil_scoped_release nogil;
            g->resetGradients();
            g->resetForwardPass();
            g->computeForwardPass();
            fval = expr_root->value;
            set_epoch_value(expr_root->gradient, expr_root->grad_epoch, g->cur_grad_epoch_, 1.0);
            g->initiateBackwardPass(expr_root);
        }
        Eigen::VectorXd grad(var_nodes.size());
        for (size_t i = 0; i < var_nodes.size(); ++i) grad[i] = var_nodes[i]->gradient;
        return {fval, std::move(grad)};
    }

    [[gnu::hot]]
    void value_grad_into_nogil(const double *x, std::size_t n, double *f_out, double *g_out) {
        if (n != var_nodes.size()) throw std::invalid_argument("value_grad_into_nogil: wrong input length");

        for (std::size_t i = 0; i < n; ++i) var_nodes[i]->value = x[i];

        g->resetGradients();
        g->resetForwardPass();
        g->computeForwardPass();
        const double fval = expr_root->value;

        set_epoch_value(expr_root->gradient, expr_root->grad_epoch, g->cur_grad_epoch_, 1.0);
        g->initiateBackwardPass(expr_root);

        if (g_out)
            for (std::size_t i = 0; i < n; ++i) g_out[i] = var_nodes[i]->gradient;
        if (f_out) *f_out = fval;
    }

    [[gnu::hot]] Out1D call_numpy(Arr1D x_in) {
        set_inputs_from_arr(var_nodes, x_in);
        {
            nb::gil_scoped_release nogil;
            g->resetGradients();
            g->computeForwardPass();
            set_epoch_value(expr_root->gradient, expr_root->grad_epoch, g->cur_grad_epoch_, 1.0);
            g->initiateBackwardPass(expr_root);
        }
        const ssize_t n   = (ssize_t)var_nodes.size();
        Out1D         out = create_uninit_1d(n);
        double       *om  = out.data();
        for (ssize_t i = 0; i < n; ++i) om[i] = var_nodes[(size_t)i]->gradient;
        return out;
    }

    [[gnu::hot]] double value_numpy(Arr1D x_in) {
        set_inputs_from_arr(var_nodes, x_in);
        double fval;
        {
            nb::gil_scoped_release nogil;
            g->resetForwardPass();
            g->computeForwardPass();
            fval = expr_root->value;
        }
        return fval;
    }

    [[gnu::hot]] std::pair<double, Out1D> value_grad_numpy(Arr1D x_in) {
        auto x = as_span_1d(x_in);
        if (x.size() != var_nodes.size()) [[unlikely]]
            throw std::invalid_argument("GradFn.value_grad: wrong input length");
        for (size_t i = 0; i < x.size(); ++i) var_nodes[i]->value = x[i];

        double fval;
        {
            nb::gil_scoped_release nogil;
            g->resetGradients();
            g->resetForwardPass();
            g->computeForwardPass();
            fval = expr_root->value;
            set_epoch_value(expr_root->gradient, expr_root->grad_epoch, g->cur_grad_epoch_, 1.0);
            g->initiateBackwardPass(expr_root);
        }
        Out1D   grad = create_uninit_1d((ssize_t)var_nodes.size());
        double *gd   = grad.data();
        for (size_t i = 0; i < var_nodes.size(); ++i) gd[i] = var_nodes[i]->gradient;
        return {fval, std::move(grad)};
    }
};

class HessFn {
public:
    ADGraphPtr             g;
    ADNodePtr              expr_root;
    std::vector<ADNodePtr> var_nodes;
    bool                   vector_mode{};
    nb::object             python_func;
    bool                   order_ready_{false};
    std::vector<int>       x2g_, g2x_;

    [[gnu::pure]] std::string expr_str() const {
        return (g && expr_root) ? g->getExpression(expr_root) : std::string{};
    }

    HessFn(nb::object f, size_t arity, bool vector_input) : vector_mode(vector_input), python_func(f) {
        auto C    = compile_to_graph<ArgPolicy::Tuple>(f, arity, vector_input);
        g         = C.g;
        expr_root = C.root;
        var_nodes = std::move(C.vars);
        g->initializeNodeVariables();
        g->simplifyGraph();
    }

    void set_inputs_seq(const nb::object &x) { set_inputs_from_seq(var_nodes, x); }
    void set_inputs_arr(const Arr1D &x) { set_inputs_from_arr(var_nodes, x); }

    void build_permutations_once_() {
        if (order_ready_) return;
        g->initializeNodeVariables();
        const size_t n = var_nodes.size();
        x2g_.assign(n, -1);
        g2x_.assign(n, -1);
        for (size_t i = 0; i < n; ++i) {
            const int k = var_nodes[i]->order;
            if (k < 0 || static_cast<size_t>(k) >= n) throw std::runtime_error("HessFn: bad variable order");
            x2g_[i] = k;
            g2x_[k] = static_cast<int>(i);
        }
        order_ready_ = true;
    }

    std::vector<double> x_to_graph_order_(std::span<const double> v_x) const {
        const size_t        n = var_nodes.size();
        std::vector<double> v_g(n, 0.0);
        for (size_t i = 0; i < n; ++i) v_g[static_cast<size_t>(x2g_[i])] = v_x[i];
        return v_g;
    }

    std::vector<double> graph_to_x_order_(const std::vector<double> &w_g) const {
        const size_t        n = var_nodes.size();
        std::vector<double> w_x(n, 0.0);
        for (size_t k = 0; k < n; ++k) w_x[static_cast<size_t>(g2x_[k])] = w_g[k];
        return w_x;
    }

    [[gnu::hot]]
    Out2D call_numpy(Arr1D x_in) {
        set_inputs_arr(x_in);
        const size_t n = var_nodes.size();
        build_permutations_once_();
        {
            nb::gil_scoped_release nogil;
            g->resetForwardPass();
        }

        Out2D   H    = create_uninit_2d((ssize_t)n, (ssize_t)n);
        double *data = H.data();

        const size_t        L = std::min<size_t>(16, std::max<size_t>(1, n));
        std::vector<double> V(n * L, 0.0), Y(n * L, 0.0);

        for (size_t base = 0; base < n; base += L) {
            const size_t k = std::min(L, n - base);
            std::fill(V.begin(), V.end(), 0.0);
            for (size_t j = 0; j < k; ++j) {
                const size_t xj                     = base + j;
                const int    gij                    = x2g_[xj];
                V[static_cast<size_t>(gij) * L + j] = 1.0;
            }

            {
                nb::gil_scoped_release nogil;
                g->hessianMultiVectorProduct(expr_root, V.data(), L, Y.data(), L, k);
            }

            for (size_t j = 0; j < k; ++j) {
                const size_t col_x = base + j;
                for (size_t gi = 0; gi < n; ++gi) {
                    const size_t xi      = static_cast<size_t>(g2x_[gi]);
                    data[xi * n + col_x] = Y[gi * L + j];
                }
            }
        }
        return H;
    }

    [[gnu::hot]]
    Arr1D hvp_numpy(Arr1D x_in, Arr1D v_in) {
        set_inputs_arr(x_in);
        build_permutations_once_();
        auto v = as_span_1d(v_in);
        if (v.size() != var_nodes.size()) [[unlikely]]
            throw std::invalid_argument("HessFn.hvp: wrong vector length");
        const auto          v_g = x_to_graph_order_(v);
        std::vector<double> Hv_g;
        {
            nb::gil_scoped_release nogil;
            g->resetForwardPass();
            Hv_g = g->hessianVectorProduct(expr_root, v_g);
        }
        auto  Hv  = graph_to_x_order_(Hv_g);
        Arr1D out = create_zeros_1d((ssize_t)v.size());
        std::memcpy(out.data(), Hv.data(), Hv.size() * sizeof(double));
        return out;
    }
};

// ============================================================================
// Lagrangian Hessian (Expression-only accumulation) + matrix-free W operator
// ============================================================================

class LagHessFn {
public:
    std::vector<int>        x2g_, g2x_;
    ADGraphPtr              g;
    ADNodePtr               L_root;
    std::vector<ADNodePtr>  x_nodes, lam_nodes, nu_nodes;
    nb::object              f_fun;
    std::vector<nb::object> cI_funs, cE_funs;
    bool                    vector_mode{};
    bool                    order_ready_{false};

    void build_permutations_once_() {
        if (order_ready_) return;
        g->initializeNodeVariables();
        const size_t n = x_nodes.size();
        x2g_.assign(n, -1);
        g2x_.assign(n, -1);
        for (size_t i = 0; i < n; ++i) {
            const int k = x_nodes[i]->order;
            if (k < 0 || (size_t)k >= n) throw std::runtime_error("bad order");
            x2g_[i] = k;
            g2x_[k] = (int)i;
        }
        order_ready_ = true;
    }

    LagHessFn(nb::object f, const std::vector<nb::object> &cI, const std::vector<nb::object> &cE, size_t arity,
              bool vector_input)
        : g(std::make_shared<ADGraph>()), f_fun(f), cI_funs(cI), cE_funs(cE), vector_mode(vector_input) {
        std::vector<nb::object> args;
        args.reserve(arity);
        x_nodes.reserve(arity);
        for (size_t i = 0; i < arity; ++i) {
            auto v = std::make_shared<Variable>("x" + std::to_string(i), 0.0);
            auto e = std::make_shared<Expression>(v, 1.0, g);
            x_nodes.push_back(e->node);
            args.emplace_back(nb::cast(e));
        }

        auto f_ret = vector_mode ? call_py_fn<ArgPolicy::List>(f_fun, args) : call_py_fn<ArgPolicy::Tuple>(f_fun, args);
        if (f_ret.is_none()) throw std::invalid_argument("LagHessFn: f returned None");
        auto f_expr = nb::cast<std::shared_ptr<Expression>>(f_ret);
        g->adoptSubgraph(f_expr->node);
        auto acc = f_expr; // Expression

        auto add_lin = [&](const std::vector<nb::object> &funs, std::vector<ADNodePtr> &coeffs) {
            coeffs.reserve(funs.size());
            for (auto &fi : funs) {
                if (!fi.is_valid()) continue;
                nb::object ci_ret =
                    vector_mode ? call_py_fn<ArgPolicy::List>(fi, args) : call_py_fn<ArgPolicy::Tuple>(fi, args);
                if (ci_ret.is_none()) throw std::invalid_argument("LagHessFn: constraint returned None");
                auto ce = nb::cast<std::shared_ptr<Expression>>(ci_ret);
                g->adoptSubgraph(ce->node);

                // λ initialized as 0.0 in the same graph, stored for external updates
                auto lam_expr = coerce_expr(nb::float_(0.0), g);
                coeffs.push_back(lam_expr->node);

                auto term = (*lam_expr) * (*ce); // term: ExpressionPtr
                acc       = *acc + *term;        // deref both sides to Expression&
            }
        };
        add_lin(cI_funs, lam_nodes);
        add_lin(cE_funs, nu_nodes);

        L_root = acc->node;
        build_permutations_once_();

        g->initializeNodeVariables();
        g->simplifyGraph();

        {
            nb::gil_scoped_release nogil;
            g->resetForwardPass();
            g->computeForwardPass();
        }
    }

    Arr2D hvp_multi_numpy(Arr1D x_in, Arr1D lam_in, Arr1D nu_in, Arr2D V_in) {
        set_state_numpy(x_in, lam_in, nu_in);
        auto [n, k] = shape_2d(V_in);
        if ((size_t)n != x_nodes.size()) throw std::invalid_argument("hvp_multi: V.rows() != nvars");

        Arr2D         Y  = create_zeros_2d(n, k);
        double       *Yd = Y.data();
        const double *Vd = V_in.data();
        build_permutations_once_();

        std::vector<double> v_x((size_t)n), v_g, Hy_g, Hy_x;
        for (ssize_t j = 0; j < k; ++j) {
            for (ssize_t i = 0; i < n; ++i) v_x[(size_t)i] = Vd[(size_t)i * (size_t)k + (size_t)j];
            v_g = x_to_graph_order_(std::span<const double>(v_x.data(), (size_t)n));
            {
                nb::gil_scoped_release nogil;
                Hy_g = g->hessianVectorProduct(L_root, v_g);
            }
            Hy_x = graph_to_x_order_(Hy_g);
            for (ssize_t i = 0; i < n; ++i) Yd[(size_t)i * (size_t)k + (size_t)j] = Hy_x[(size_t)i];
        }
        return Y;
    }

    Eigen::SparseMatrix<double> hess_sparse(const Eigen::Ref<const Eigen::VectorXd> &x_in,
                                            const Eigen::Ref<const Eigen::VectorXd> &lam_in,
                                            const Eigen::Ref<const Eigen::VectorXd> &nu_in, double tol = 1e-12) {
        set_state_eigen(x_in, lam_in, nu_in);

        const size_t n = x_nodes.size();
        build_permutations_once_();

        constexpr size_t L = 16; // block width
        // Row-major scratch to match (gi*L + j) addressing
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> V(n, L);
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Y(n, L);

        std::vector<Eigen::Triplet<double>> trips;
        trips.reserve(std::min<size_t>(n * 16, n * n)); // rough guess; grows if needed

        for (size_t base = 0; base < n; base += L) {
            const size_t k = std::min(L, n - base);

            // Build k basis columns in graph index space
            V.setZero();
            for (size_t j = 0; j < k; ++j) {
                const size_t xj                                                 = base + j;
                const int    gij                                                = x2g_[xj];
                V(static_cast<Eigen::Index>(gij), static_cast<Eigen::Index>(j)) = 1.0;
            }

            {
                nb::gil_scoped_release nogil;
                g->hessianMultiVectorProduct(L_root, V.data(), L, Y.data(), L, k);
            }

            // Scatter: we only take i <= col_x (upper triangle)
            for (size_t j = 0; j < k; ++j) {
                const size_t col_x = base + j;
                for (size_t gi = 0; gi < n; ++gi) {
                    const size_t xi = static_cast<size_t>(g2x_[gi]);
                    if (xi > col_x) continue; // keep to upper triangle

                    const double val = Y(static_cast<Eigen::Index>(gi), static_cast<Eigen::Index>(j));
                    if (std::abs(val) >= tol) {
                        trips.emplace_back(static_cast<int>(xi), static_cast<int>(col_x), val);
                    }
                }
            }
        }

        // Build upper-triangular sparse
        Eigen::SparseMatrix<double> H_upper(static_cast<int>(n), static_cast<int>(n));
        H_upper.setFromTriplets(trips.begin(), trips.end());
        H_upper.makeCompressed();

        // Symmetrize: H = H_upper + H_upper.transpose() - diag(H_upper)
        // (avoid double-counting diagonal)
        Eigen::SparseMatrix<double> H = H_upper.selfadjointView<Eigen::Upper>();
        return H; // CSC by default
    }

    void set_state_eigen(const dvec &x, const dvec &lam, const dvec &nu) {
        if ((size_t)x.size() != x_nodes.size()) throw std::invalid_argument("set_state_eigen: x size mismatch");
        if ((size_t)lam.size() != lam_nodes.size() || (size_t)nu.size() != nu_nodes.size())
            throw std::invalid_argument("set_state_eigen: multiplier size mismatch");
        for (size_t i = 0; i < x_nodes.size(); ++i) x_nodes[i]->value = x[i];
        for (size_t i = 0; i < lam_nodes.size(); ++i) lam_nodes[i]->value = lam[i];
        for (size_t i = 0; i < nu_nodes.size(); ++i) nu_nodes[i]->value = nu[i];
        nb::gil_scoped_release nogil;
        g->resetForwardPass();
        g->computeForwardPass();
    }

    void set_state_numpy(Arr1D x_in, Arr1D lam_in, Arr1D nu_in) {
        auto span1 = [](const Arr1D &a) -> std::span<const double> {
            if (a.ndim() != 1) throw std::invalid_argument("expected 1D float64");
            return {a.data(), (size_t)a.shape(0)};
        };
        auto x = span1(x_in), l = span1(lam_in), n = span1(nu_in);
        dvec xe((int)x.size()), le((int)l.size()), ne((int)n.size());
        std::memcpy(xe.data(), x.data(), x.size() * sizeof(double));
        std::memcpy(le.data(), l.data(), l.size() * sizeof(double));
        std::memcpy(ne.data(), n.data(), n.size() * sizeof(double));
        set_state_eigen(xe, le, ne);
    }

    void refresh_orders_() const { g->initializeNodeVariables(); }

    std::vector<double> x_to_graph_order_(std::span<const double> v_x) const {
        const size_t        n = x_nodes.size();
        std::vector<double> v_g(n);
        for (size_t i = 0; i < n; ++i) v_g[(size_t)x2g_[i]] = v_x[i];
        return v_g;
    }
    std::vector<double> graph_to_x_order_(const std::vector<double> &w_g) const {
        const size_t        n = x_nodes.size();
        std::vector<double> w_x(n);
        for (size_t k = 0; k < n; ++k) w_x[(size_t)g2x_[k]] = w_g[k];
        return w_x;
    }

    Arr1D hvp_numpy(Arr1D x_in, Arr1D lam_in, Arr1D nu_in, Arr1D v_in) {
        set_state_numpy(x_in, lam_in, nu_in);
        build_permutations_once_();
        std::span<const double> vx{v_in.data(), (size_t)v_in.shape(0)};
        auto                    v_g = x_to_graph_order_(vx);
        std::vector<double>     Hv_g;
        {
            nb::gil_scoped_release nogil;
            Hv_g = g->hessianVectorProduct(L_root, v_g);
        }
        auto  Hv_x = graph_to_x_order_(Hv_g);
        Arr1D out  = create_zeros_1d((ssize_t)Hv_x.size());
        std::memcpy(out.data(), Hv_x.data(), Hv_x.size() * sizeof(double));
        return out;
    }

    Arr2D hess_numpy(Arr1D x_in, Arr1D lam_in, Arr1D nu_in) {
        set_state_numpy(x_in, lam_in, nu_in);
        const size_t n  = x_nodes.size();
        Arr2D        H  = create_zeros_2d((ssize_t)n, (ssize_t)n);
        double      *Hd = H.data();

        build_permutations_once_();
        const size_t        L = 16;
        std::vector<double> V(n * L, 0.0), Y(n * L, 0.0);

        for (size_t base = 0; base < n; base += L) {
            const size_t k = std::min(L, n - base);
            std::fill(V.begin(), V.end(), 0.0);
            for (size_t j = 0; j < k; ++j) {
                const size_t xj        = base + j;
                const int    gij       = x2g_[xj];
                V[(size_t)gij * L + j] = 1.0;
            }
            {
                nb::gil_scoped_release nogil;
                g->hessianMultiVectorProduct(L_root, V.data(), L, Y.data(), L, k);
            }
            for (size_t j = 0; j < k; ++j) {
                const size_t col_x = base + j;
                for (size_t gi = 0; gi < n; ++gi) {
                    const size_t xi    = (size_t)g2x_[gi];
                    Hd[xi * n + col_x] = Y[gi * L + j];
                }
            }
        }
        return H;
    }
};

// Matrix-free operator: y = (H_L + sigma I + diag(Sigma_x) + JIᵀ diag(Sigma_s) JI) x
class CompiledWOp {
public:
    using Vec = Eigen::VectorXd;
    using Sp  = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;

    CompiledWOp() = default;

    explicit CompiledWOp(std::shared_ptr<LagHessFn> L, Eigen::VectorXd Sigma_x, std::optional<Sp> JI = std::nullopt,
                         std::optional<Eigen::VectorXd> Sigma_s = std::nullopt, double sigma_isotropic = 0.0)
        : L_(std::move(L)), sigma_(sigma_isotropic), Sigma_x_(std::move(Sigma_x)), JI_(std::move(JI)),
          Sigma_s_(std::move(Sigma_s)) {
        L_->build_permutations_once_();
        const std::size_t n = L_->x_nodes.size();
        Vcol_.assign(n, 0.0);
        Ycol_.assign(n, 0.0);
        y_x_.assign(n, 0.0);
        has_Sx_ = (Sigma_x_.size() == static_cast<Eigen::Index>(n));
        if (JI_.has_value() || Sigma_s_.has_value()) {
            if (!JI_.has_value() || !Sigma_s_.has_value())
                throw std::invalid_argument("CompiledWOp: JI and Sigma_s must be provided together.");
            if (static_cast<std::size_t>(JI_->cols()) != n)
                throw std::invalid_argument("CompiledWOp: JI.cols() must equal n.");
            if (JI_->rows() != Sigma_s_->size())
                throw std::invalid_argument("CompiledWOp: JI.rows() must equal Sigma_s.size().");
            has_JI_ = true;
            t_.resize(JI_->rows());
        }
    }

    inline Eigen::Index rows() const { return static_cast<Eigen::Index>(L_->x_nodes.size()); }
    inline Eigen::Index cols() const { return rows(); }

    inline void   set_sigma(double s) { sigma_ = s; }
    inline double sigma() const { return sigma_; }

    inline void update_sigma_x(const Eigen::VectorXd &s) {
        if (s.size() == 0) {
            has_Sx_ = false;
            Sigma_x_.resize(0);
            return;
        }
        if (s.size() != rows()) throw std::invalid_argument("CompiledWOp::update_sigma_x: size mismatch.");
        Sigma_x_ = s;
        has_Sx_  = true;
    }

    inline void update_JI_and_sigma_s(const std::optional<Sp> &JI, const std::optional<Eigen::VectorXd> &Sigma_s) {
        if (JI.has_value() || Sigma_s.has_value()) {
            if (!JI.has_value() || !Sigma_s.has_value())
                throw std::invalid_argument("Both JI and Sigma_s must be provided together.");
            if (JI->cols() != rows()) throw std::invalid_argument("JI.cols() must equal n.");
            if (JI->rows() != Sigma_s->size()) throw std::invalid_argument("JI.rows() must equal Sigma_s.size().");
            JI_      = JI;
            Sigma_s_ = Sigma_s;
            has_JI_  = true;
            t_.resize(JI_->rows());
        } else {
            JI_.reset();
            Sigma_s_.reset();
            has_JI_ = false;
            t_.resize(0);
        }
    }

    inline void update_sigma_s_only(const Eigen::VectorXd &s) {
        if (!has_JI_) throw std::logic_error("update_sigma_s_only: JI/Sigma_s not enabled.");
        if (s.size() != Sigma_s_->size()) throw std::invalid_argument("update_sigma_s_only: size mismatch.");
        *Sigma_s_ = s;
    }

    template <typename DerivedIn, typename DerivedOut>
    inline void perform_op(const Eigen::MatrixBase<DerivedIn> &x, Eigen::MatrixBase<DerivedOut> &y) const {
        const std::size_t n = static_cast<std::size_t>(x.size());
        if (rows() != static_cast<Eigen::Index>(n))
            throw std::invalid_argument("CompiledWOp::perform_op: vector size mismatch.");

        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t gi = static_cast<std::size_t>(L_->x2g_[i]);
            Vcol_[gi]            = x[static_cast<Eigen::Index>(i)];
        }

        {
            nb::gil_scoped_release nogil;
            L_->g->hessianMultiVectorProduct(L_->L_root, Vcol_.data(), 1, Ycol_.data(), 1, 1);
        }

        for (std::size_t gi = 0; gi < n; ++gi) {
            const std::size_t xi = static_cast<std::size_t>(L_->g2x_[gi]);
            y_x_[xi]             = Ycol_[gi];
        }

        Eigen::Map<Vec> ymap(y.derived().data(), y.size());
        ymap = Eigen::Map<const Vec>(y_x_.data(), static_cast<Eigen::Index>(y_x_.size()));
        if (sigma_ != 0.0) ymap.noalias() += sigma_ * x.derived();
        if (has_Sx_) ymap.noalias() += Sigma_x_.cwiseProduct(x.derived());
        if (has_JI_) {
            t_.noalias() = (*JI_) * x.derived(); // mI
            t_.array() *= Sigma_s_->array();
            ymap.noalias() += JI_->transpose() * t_;
        }
    }

    inline Vec operator*(const Vec &x) const {
        Vec y(rows());
        perform_op(x, y);
        return y;
    }

private:
    std::shared_ptr<LagHessFn>     L_;
    double                         sigma_ = 0.0;
    Eigen::VectorXd                Sigma_x_;
    std::optional<Sp>              JI_;
    std::optional<Eigen::VectorXd> Sigma_s_;
    bool                           has_Sx_ = false, has_JI_ = false;

    mutable std::vector<double> Vcol_, Ycol_, y_x_;
    mutable Eigen::VectorXd     t_;
};

// ============================================================================
// Caches (unchanged semantics; cleaner wrapper)
// ============================================================================

struct FnKey {
    PyObject      *f;
    size_t         arity;
    bool           vector;
    constexpr bool operator==(const FnKey &o) const noexcept {
        return f == o.f && arity == o.arity && vector == o.vector;
    }
};
struct FnKeyHash {
    [[gnu::always_inline]] constexpr size_t operator()(const FnKey &k) const noexcept {
        size_t h = std::bit_cast<size_t>(k.f);
        h ^= h >> 30;
        h *= 0xbf58476d1ce4e5b9ULL;
        h ^= h >> 27;
        h *= 0x94d049bb133111ebULL;
        h ^= h >> 31;
        h ^= k.arity + 0x9e3779b97f4a7c15ULL;
        h ^= (size_t)k.vector << 1;
        return h;
    }
};

thread_local tsl::robin_map<FnKey, std::shared_ptr<GradFn>, FnKeyHash> tl_grad_cache;
thread_local tsl::robin_map<FnKey, std::shared_ptr<HessFn>, FnKeyHash> tl_hess_cache;
static std::shared_mutex                                               g_cache_mtx;
static tsl::robin_map<FnKey, std::weak_ptr<GradFn>, FnKeyHash>         g_grad_cache;
static tsl::robin_map<FnKey, std::weak_ptr<HessFn>, FnKeyHash>         g_hess_cache;

template <class T, class TLMap, class GlobalMap, class Maker>
[[gnu::hot]]
static inline std::shared_ptr<T> cache_get_or_make(TLMap &tl_map, GlobalMap &g_map, const FnKey &k, Maker &&make) {
    if (auto it = tl_map.find(k); it != tl_map.end()) [[likely]]
        return it->second;
    {
        std::shared_lock rlk(g_cache_mtx);
        if (auto it = g_map.find(k); it != g_map.end())
            if (auto sp = it->second.lock()) {
                tl_map[k] = sp;
                return sp;
            }
    }
    std::unique_lock wlk(g_cache_mtx);
    if (auto it = g_map.find(k); it != g_map.end())
        if (auto sp = it->second.lock()) {
            tl_map[k] = sp;
            return sp;
        }
    auto sp   = make();
    g_map[k]  = sp;
    tl_map[k] = sp;
    return sp;
}

[[gnu::always_inline]]
static inline std::shared_ptr<GradFn> get_or_make_grad(nb::object f, size_t n, bool vec) {
    FnKey k{f.ptr(), n, vec};
    return cache_get_or_make<GradFn>(tl_grad_cache, g_grad_cache, k,
                                     [&] { return std::make_shared<GradFn>(f, n, vec); });
}

[[gnu::always_inline]]
static inline std::shared_ptr<HessFn> get_or_make_hess(nb::object f, size_t n, bool vec) {
    FnKey k{f.ptr(), n, vec};
    return cache_get_or_make<HessFn>(tl_hess_cache, g_hess_cache, k,
                                     [&] { return std::make_shared<HessFn>(f, n, vec); });
}

// Release every cached GradFn/HessFn (and the Python callables they own) while
// the interpreter is still alive. Registered with Python's atexit in NB_MODULE.
// Without this, the thread_local caches are torn down by C++ TLS destructors
// after Py_Finalize, where dropping the captured nb::object segfaults in
// func_dealloc.
static inline void clear_fn_caches() noexcept {
    tl_grad_cache.clear();
    tl_hess_cache.clear();
    std::unique_lock wlk(g_cache_mtx);
    g_grad_cache.clear();
    g_hess_cache.clear();
}

// ============================================================================
// High-level APIs (thin wrappers; all algebra via Expression)
// ============================================================================

[[gnu::hot]]
static nb::list py_gradient(nb::object f, nb::args xs) {
    auto g = std::make_shared<ADGraph>();

    auto coerce_for_call = [&](nb::handle item, size_t idx, std::vector<nb::object> &dst,
                               std::vector<std::string> &names) {
        if (nb::isinstance<Variable>(item)) {
            auto v     = nb::cast<std::shared_ptr<Variable>>(item);
            names[idx] = v->getName();
            dst.emplace_back(nb::cast(std::make_shared<Expression>(v, 1.0, g)));
        } else if (is_number(item)) {
            const double val = nb::cast<double>(item);
            auto         vx  = std::make_shared<Variable>("x" + std::to_string(idx), val);
            names[idx]       = vx->getName();
            dst.emplace_back(nb::cast(std::make_shared<Expression>(vx, 1.0, g)));
        } else if (nb::isinstance<Expression>(item)) {
            auto e = nb::cast<std::shared_ptr<Expression>>(item);
            g->adoptSubgraph(e->node);
            dst.emplace_back(nb::cast(e));
        } else {
            throw std::invalid_argument("gradient: args must be Variable / Expression / number.");
        }
    };

    std::vector<nb::object>  expr_args;
    std::vector<std::string> var_names;

    if (nb::len(xs) == 1 && is_sequence(xs[0])) [[likely]] {
        nb::sequence seq = nb::borrow<nb::sequence>(xs[0]);
        const size_t n   = nb::len(seq);
        expr_args.reserve(n);
        var_names.resize(n);
        for (size_t i = 0; i < n; ++i) coerce_for_call(seq[i], i, expr_args, var_names);
        auto                                ret  = call_py_fn<ArgPolicy::List>(f, expr_args);
        auto                                expr = nb::cast<std::shared_ptr<Expression>>(ret);
        tsl::robin_map<std::string, double> grad_map;
        {
            nb::gil_scoped_release nogil;
            grad_map = expr->computeGradient();
        }
        nb::list out;
        for (size_t i = 0; i < n; ++i) {
            double gi = 0.0;
            if (!var_names[i].empty())
                if (auto it = grad_map.find(var_names[i]); it != grad_map.end()) gi = it->second;
            out.append(nb::float_(gi));
        }
        return out;
    }

    const size_t n = nb::len(xs);
    expr_args.reserve(n);
    var_names.resize(n);
    for (size_t i = 0; i < n; ++i) coerce_for_call(xs[i], i, expr_args, var_names);

    auto                                ret  = call_py_fn<ArgPolicy::Tuple>(f, expr_args);
    auto                                expr = nb::cast<std::shared_ptr<Expression>>(ret);
    tsl::robin_map<std::string, double> grad_map;
    {
        nb::gil_scoped_release nogil;
        grad_map = expr->computeGradient();
    }

    nb::list out;
    for (size_t i = 0; i < n; ++i) {
        double gi = 0.0;
        if (!var_names[i].empty())
            if (auto it = grad_map.find(var_names[i]); it != grad_map.end()) gi = it->second;
        out.append(nb::float_(gi));
    }
    return out;
}

// fused value+grad (cached)
[[gnu::hot]]
static std::pair<double, Out1D> py_value_grad_numpy(nb::object f, Arr1D x_in) {
    auto gf = get_or_make_grad(f, (size_t)as_span_1d(x_in).size(), /*vec=*/true);
    // Single forward+backward pass yields both value and gradient; the previous
    // value_numpy()+call_numpy() did the forward pass twice.
    return gf->value_grad_numpy(x_in);
}

[[gnu::hot]]
static Out1D py_gradient_numpy(nb::object f, Arr1D x_in) {
    auto gf = get_or_make_grad(f, (size_t)as_span_1d(x_in).size(), /*vec=*/true);
    return gf->call_numpy(x_in);
}

[[gnu::hot]]
static Out2D py_hessian_numpy(nb::object f, Arr1D x_in) {
    auto hf = get_or_make_hess(f, (size_t)as_span_1d(x_in).size(), /*vec=*/true);
    return hf->call_numpy(x_in);
}

// Batch fused value+grad over compiled GradFn objects
[[gnu::hot]]
static std::pair<Out1D, Out2D> batch_value_grad_from_gradfns(const std::vector<std::shared_ptr<GradFn>> &gfs,
                                                             Arr1D                                       x_in) {
    const auto    x = as_span_1d(x_in);
    const ssize_t m = (ssize_t)gfs.size();
    if (m == 0) return {create_uninit_1d(0), create_uninit_2d(0, (ssize_t)x.size())};

    for (const auto &gf : gfs) {
        if (!gf) throw std::invalid_argument("batch_valgrad: null GradFn");
        if (gf->var_nodes.size() != (size_t)x.size()) throw std::invalid_argument("batch_valgrad: inconsistent arity");
    }

    Out1D         vals = create_uninit_1d(m);
    Out2D         J    = create_uninit_2d(m, (ssize_t)x.size());
    double       *vd   = vals.data();
    double       *Jd   = J.data();
    const ssize_t n    = (ssize_t)x.size();

    {
        nb::gil_scoped_release nogil;
        for (ssize_t j = 0; j < m; ++j) {
            auto &gf = gfs[(size_t)j];
            for (ssize_t i = 0; i < n; ++i) gf->var_nodes[(size_t)i]->value = x[(size_t)i];

            gf->g->resetGradients();
            gf->g->resetForwardPass();
            gf->g->computeForwardPass();
            vd[j] = gf->expr_root->value;

            set_epoch_value(gf->expr_root->gradient, gf->expr_root->grad_epoch, gf->g->cur_grad_epoch_, 1.0);
            gf->g->initiateBackwardPass(gf->expr_root);

            for (ssize_t i = 0; i < n; ++i) Jd[(size_t)j * (size_t)n + (size_t)i] = gf->var_nodes[(size_t)i]->gradient;
        }
    }
    return {std::move(vals), std::move(J)};
}

// Overload: list of Python callables → compile/cache transparently
[[gnu::hot]]
static std::pair<Out1D, Out2D> batch_value_grad_from_callables(nb::list funcs, Arr1D x_in) {
    const ssize_t                        m = nb::len(funcs);
    std::vector<std::shared_ptr<GradFn>> gfs;
    gfs.reserve((size_t)m);
    const size_t n = (size_t)as_span_1d(x_in).size();
    for (ssize_t j = 0; j < m; ++j) {
        nb::object f = funcs[(size_t)j];
        gfs.emplace_back(get_or_make_grad(f, n, /*vec=*/true));
    }
    return batch_value_grad_from_gradfns(gfs, x_in);
}

// ============================================================================
// Other
// ============================================================================

static inline std::pair<dvec, spmat>
batch_value_grad_from_gradfns_sparse(const std::vector<std::shared_ptr<GradFn>> &gfs, const dvec &x, double tol = 0.0,
                                     int reserve_nnz_per_row = 0) {
    const Eigen::Index m = static_cast<Eigen::Index>(gfs.size());
    const Eigen::Index n = x.size();

    if (m == 0) { return {dvec(0), spmat(0, n)}; }

    // sanity: all arities must match x.size()
    for (const auto &gf : gfs) {
        if (!gf) throw std::invalid_argument("batch_valgrad: null GradFn");
        if (static_cast<Eigen::Index>(gf->var_nodes.size()) != n)
            throw std::invalid_argument("batch_valgrad: inconsistent arity");
    }

    dvec  vals(m);
    spmat J(m, n);

    using SIndex = typename spmat::StorageIndex;
    std::vector<Eigen::Triplet<double, SIndex>> triplets;
    {
        // reserve a sensible amount of space to avoid re-allocations
        const Eigen::Index hint_per_row =
            (reserve_nnz_per_row > 0) ? static_cast<Eigen::Index>(reserve_nnz_per_row) : std::min<Eigen::Index>(n, 64);
        triplets.reserve(static_cast<size_t>(m * hint_per_row));

        nb::gil_scoped_release nogil;

        for (Eigen::Index j = 0; j < m; ++j) {
            auto &gf = gfs[static_cast<size_t>(j)];

            // set inputs
            for (Eigen::Index i = 0; i < n; ++i) gf->var_nodes[static_cast<size_t>(i)]->value = x[i];

            // fused value + grad
            gf->g->resetGradients();
            gf->g->resetForwardPass();
            gf->g->computeForwardPass();
            vals[j] = gf->expr_root->value;

            set_epoch_value(gf->expr_root->gradient, gf->expr_root->grad_epoch, gf->g->cur_grad_epoch_, 1.0);
            gf->g->initiateBackwardPass(gf->expr_root);

            // append sparse row j
            if (tol <= 0.0) {
                for (Eigen::Index i = 0; i < n; ++i) {
                    const double g = gf->var_nodes[static_cast<size_t>(i)]->gradient;
                    if (g != 0.0) triplets.emplace_back(static_cast<SIndex>(j), static_cast<SIndex>(i), g);
                }
            } else {
                for (Eigen::Index i = 0; i < n; ++i) {
                    const double g = gf->var_nodes[static_cast<size_t>(i)]->gradient;
                    if (std::abs(g) > tol) triplets.emplace_back(static_cast<SIndex>(j), static_cast<SIndex>(i), g);
                }
            }
        }
    }

    J.setFromTriplets(triplets.begin(), triplets.end());
    J.makeCompressed();
    return {std::move(vals), std::move(J)};
}

[[gnu::hot]]
static std::pair<Out1D, Out2D> batch_value_grad_numpy(nb::sequence grads, Arr1D x_in) {
    const std::size_t m = static_cast<std::size_t>(nb::len(grads));
    auto              x = as_span_1d(x_in);
    const std::size_t n = x.size();

    // Collect raw pointers to compiled GradFn to avoid Python in parallel loop
    std::vector<GradFn *> gptrs;
    gptrs.reserve(m);
    for (std::size_t j = 0; j < m; ++j) {
        nb::handle h = grads[j];
        if (!nb::isinstance<GradFn>(h)) throw std::invalid_argument("batch_valgrad: grads must be GradFn objects");
        gptrs.push_back(nb::cast<std::shared_ptr<GradFn>>(h).get());
    }

    // Allocate outputs (C-contiguous row-major); every entry is written below.
    Out1D   vals   = create_uninit_1d(static_cast<ssize_t>(m));
    Out2D   J      = create_uninit_2d(static_cast<ssize_t>(m), static_cast<ssize_t>(n));
    double *vals_d = vals.data();
    double *J_d    = J.data(); // row j starts at J_d + j*n

    // Parallelize over rows (each GradFn has its own ADGraph -> thread-safe)
    {
        nb::gil_scoped_release nogil;

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (std::ptrdiff_t j = 0; j < static_cast<std::ptrdiff_t>(m); ++j) {
            GradFn *gf   = gptrs[static_cast<std::size_t>(j)];
            double *rowj = J_d + static_cast<std::size_t>(j) * n;
            double  fj   = 0.0;
            gf->value_grad_into_nogil(x.data(), n, &fj, rowj);
            vals_d[j] = fj;
        }
    }

    return {std::move(vals), std::move(J)};
}
