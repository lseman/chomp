# Solver organization

`include/nlp/solver.h` owns the generic solve loop, model, selected backend,
primal/dual iterate, convergence checks and diagnostics. An optional iteration
observer handles reporting without coupling the loop to console output.
Construction and configuration still require Python and the GIL because the
model compiles Python objective/constraint functions through the AD module.

`src/chomp.cpp` adapts this driver to Python: configuration bindings, NumPy
conversion, diagnostic dictionaries, context management and interpreter-shutdown
cleanup. Only fully constructed Python wrappers enter the live-instance registry.

`include/nlp/config.h` normalizes Python configuration aliases.
`include/nlp/info.h` defines diagnostics shared by both algorithms;
`include/config/solver.h` remains a compatibility include.
`include/nlp/iteration_output.h` formats console progress.

The numerical implementations remain separate:

- `include/ip/`: barrier state, primal/dual steps, KKT solves and IP globalization.
- `include/sqp/`: SQP iteration and DOpt stabilization.
- `include/trustregion/`: constrained subproblems, restoration and corrections.
- `include/model/`: compiled evaluation, Hessian caching, shared L-BFGS Hessian
  approximation (`lbfgs.h`) and regularization.
- `include/blocks/`: shared bounds and globalization primitives.
- `solvers/`: external/submodule linear and quadratic solver implementations.

Explicit `ip` and `sqp` modes construct only their selected backend. `auto`
selects IP for bounds or inequalities and SQP otherwise, including equality-only
problems. It does not switch algorithms during iterations; the historical
`auto_*` configuration fields remain available for compatibility. DFO is not
implemented and raises at construction instead of returning false convergence.
Convergence uses `tol_stat`, `tol_feas` and `tol_comp`; the Python `tol` argument
is retained for compatibility and has no effect, as before this refactor.
