#pragma once

#include <nanobind/nanobind.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <variant>

using dvec = Eigen::VectorXd;
using dmat = Eigen::MatrixXd;
using spmat = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;

using Val = std::variant<double, dvec, dmat, spmat>;
using Dict = std::unordered_map<std::string, Val>;

namespace nb = nanobind;
using namespace nb::literals;

template <class T>
static inline T get_attr_or(const nb::handle& obj, const char* name,
                            const T& fallback) {
    if (!obj || !nb::hasattr(obj, name)) return fallback;
    try {
        return nb::cast<T>(obj.attr(name));
    } catch (const nb::cast_error&) {
        return fallback;
    }
}

namespace pyu {
[[nodiscard]] inline bool has_attr(const nb::object& o,
                                   const char* name) noexcept {
    return o.is_valid() && PyObject_HasAttrString(o.ptr(), name);
}
}  // namespace pyu
