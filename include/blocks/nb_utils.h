#pragma once

#include <nanobind/nanobind.h>

namespace blocks {
namespace nb = nanobind;

template <class T>
[[nodiscard]] inline T attr_or(const nb::object& obj, const char* name,
                               T fallback) {
    if (!obj || obj.is_none()) return fallback;
    if (nb::hasattr(obj, name)) return nb::cast<T>(obj.attr(name));
    return fallback;
}

[[nodiscard]] inline double attr_or_double(const nb::object& obj,
                                           const char* name, double fallback) {
    return attr_or<double>(obj, name, fallback);
}

[[nodiscard]] inline int attr_or_int(const nb::object& obj, const char* name,
                                     int fallback) {
    return attr_or<int>(obj, name, fallback);
}

[[nodiscard]] inline bool attr_or_bool(const nb::object& obj, const char* name,
                                       bool fallback) {
    return attr_or<bool>(obj, name, fallback);
}

}  // namespace blocks
