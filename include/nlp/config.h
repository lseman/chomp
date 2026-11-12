#pragma once

#include <core/definitions.h>
#include <nanobind/stl/string.h>
#include <string>
#include <utility>

namespace nlp {
template <typename T>
inline void ensure_attr_from_alias(nb::object &cfg, const char *name, const char *alias, const T &fallback) {
    if (!nb::hasattr(cfg, name)) { cfg.attr(name) = nb::cast(get_attr_or<T>(cfg, alias, fallback)); }
}

inline double get_float_attr_or(const nb::object &o, const char *name, double defval) {
    if (!nb::hasattr(o, name)) return defval;
    try {
        return nb::cast<double>(o.attr(name));
    } catch (...) { return defval; }
}

inline std::string get_str_attr_or(const nb::object &o, const char *name, const std::string &defval) {
    if (!nb::hasattr(o, name)) return defval;
    try {
        return nb::cast<std::string>(o.attr(name));
    } catch (...) { return defval; }
}

inline bool get_bool_attr_or(const nb::object &o, const char *name, bool defval) {
    if (!nb::hasattr(o, name)) return defval;
    try {
        return nb::cast<bool>(o.attr(name));
    } catch (...) { return defval; }
}

inline nb::object make_default_config_object() { return nb::cast(ChompConfig{}); }

inline void ensure_config_aliases(nb::object &cfg) {
    ensure_attr_from_alias(cfg, "tr_delta0", "delta0", 1.0);
    ensure_attr_from_alias(cfg, "delta0", "tr_delta0", 1.0);
    ensure_attr_from_alias(cfg, "tr_norm_type", "norm_type", std::string("2"));
    ensure_attr_from_alias(cfg, "norm_type", "tr_norm_type", std::string("2"));
}

} // namespace nlp
