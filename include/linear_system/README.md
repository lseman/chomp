# Linear System Tools

High-performance C++23 sparse linear algebra library with state-of-the-art optimizations.

## Components

### QDLDL (qdldl.h)

Header-only LDLᵀ (Cholesky-like) factorization for upper triangular sparse matrices.

**Features:**
- Exact-zero pivot detection (no sign forcing)
- SIMD acceleration (AVX-512, AVX2, SSE2)
- Cache-line alignment and prefetching
- Iterative refinement for improved accuracy
- Symmetric matrix permutation & reordering
- Thread-safe work arrays with parallel elimination

**Usage:**
```cpp
#include "linear_system/qdldl.h"
using namespace qdldl23;

// Create sparse matrix (upper triangular CSC)
SparseUpperCSC<double, int32_t> A(n, Ap, Ai, Ax);

// Factor: A ≈ L D L^T
auto F = factorize(A);

// Solve: A*x = b
std::vector<double> x = b;
solve(F, x.data());

// With permutation
auto ord = Ordering<int32_t>::from_perm(perm);
auto B = permute_symmetric_upper(A, ord);
auto F_perm = factorize(B);
solve_with_ordering(F_perm, ord, x.data());

// Iterative refinement
refine(A, F, x.data(), b.data(), /*iters=*/2);
```

**API:**
- `factorize()` – full analysis + numeric factorization
- `factorize_with_ordering()` – apply permutation before factorizing
- `analyze_fast()` – symbolic analysis only (etree, column counts)
- `refactorize()` – numeric refactorization given symbolic structure
- `solve()` / `L_solve()` / `Lt_solve()` – triangular solves
- `solve_with_ordering()` – solve with permutation undone
- `refine()` – iterative refinement
- `sym_spmv_upper()` – symmetric sparse matrix-vector product
- `permute_symmetric_upper()` – apply symmetric permutation

**Type Aliases:**
- `SparseD32` = `SparseUpperCSC<double, int32_t>`
- `SparseD64` = `SparseUpperCSC<double, int64_t>`
- `SparseF32` = `SparseUpperCSC<float, int32_t>`
- `Symb32` = `Symbolic<int32_t>`
- `LDL32` = `LDLFactors<double, int32_t>`

### Supernodes (supernodes.h)

Supernode identification for packed data structures and supernode-level factorization.

**Features:**
- Etree-based supernode chain detection
- Relaxed matching with configurable tolerance
- Symbolic L-pattern computation
- Postorder enumeration
- Direct integration with QDLDL

**Usage:**
```cpp
#include "linear_system/qdldl.h"
#include "linear_system/supernodes.h"
using namespace qdldl23;
using namespace snode;

SparseD32 B = /* permuted upper CSC */;
Symb32 S = analyze_fast(B);

auto sn = identify_supernodes_qdldl(
    B, S,
    /*relax_abs=*/ 2,        // max symmetric difference
    /*relax_rel=*/ 0.10,     // fraction tolerance
    /*tau=*/ 0.70,           // Jaccard similarity threshold
    /*max_size=*/ 128        // max supernode width
);

// Access: sn.ranges, sn.col2sn, sn.etree, sn.post
```

**Output:**
- `ranges` – inclusive [lo, hi] pairs for each supernode (in permuted column space)
- `col2sn` – map from column to supernode id
- `etree` – elimination tree parent pointers
- `post` – postorder of etree

### AMD (amd.h)

Approximate Minimum Degree fill-reducing permutation for general sparse matrices.

**Features:**
- Hash-based variable coalescence by element signature
- Robin-Hood absorption of duplicate/subset elements
- Degree-based pivot selection with dense postponement
- Bandwidth and fill statistics
- CSR (pattern-only) format; works with or without symmetrization

**Usage:**
```cpp
#include "linear_system/amd.h"

CSR A(n);
A.indptr = /* size n+1 */;
A.indices = /* col indices */;

AMDReorderingArray amd;
auto [perm, stats] = amd.compute_fill_reducing_permutation(
    A,
    /*symmetrize=*/true  // symmetric or unsymmetric
);

// Access: perm, stats.bandwidth_reduction, stats.matrix_size, etc.
```

**Stats:**
- `original_nnz` – input nonzero count
- `original_bandwidth` – before permutation
- `reordered_bandwidth` – after permutation
- `bandwidth_reduction` – percentage improvement
- `absorbed_elements` – duplicate elements merged
- `coalesced_variables` – same-neighbor variables merged
- `inverse_permutation` – iperm[old] = new

## Compilation

### Standalone (header-only)
```bash
g++ -O3 -std=c++23 -I./include my_code.cpp -o my_app
```

### With CMake
```cmake
target_include_directories(mytarget PUBLIC ${CMAKE_SOURCE_DIR}/include)
```

### With SIMD
Automatic detection; control via:
- `__AVX512F__` / `__AVX512VL__` → AVX-512
- `__AVX2__` / `__FMA__` → AVX2
- `__SSE2__` → SSE2
- (nothing) → scalar fallback

### With stdexec (optional)
Define `QDLDL23_USE_STDEXEC` before include to enable parallel loops via stdexec.

## Testing

### Standalone C++ Tests
```bash
g++ -O3 -std=c++23 -I./include tests/test_linear_system_standalone.cpp -o test_ls
./test_ls
```

All 13 tests pass:
- QDLDL: identity, diagonal, tridiag, solve (2x2, 3x3), permutation, refinement
- Supernodes: trivial (diagonal), simple merge
- AMD: identity, path graph, stats
- Integration: AMD → QDLDL → solve

### Python Tests
```bash
pytest tests/test_linear_system.py -v
```
(Requires `_chomp` bindings; auto-skips if unavailable)

## Performance Notes

- **SIMD:** 4–8× speedup on dense operations (scale, SpMV, triangular solve)
- **Cache:** Prefetching on next column; ~64-byte line alignment
- **Memory:** Cache-aligned work arrays; scattered-access optimization
- **Parallelization:** Automatic via stdexec or pthreads (configurable threshold: 1000 elements)

## Format Specifications

### CSC Upper Triangular
- **Ap** (size n+1): column pointers
- **Ai** (size nnz): row indices (must satisfy i ≤ j for entry in column j)
- **Ax** (size nnz): numerical values
- All diagonals must be present (checked in `finalize_upper_inplace`)

### CSR (AMD only)
- **indptr** (size n+1): row pointers
- **indices** (size nnz): column indices

## Error Handling

- `InvalidMatrixError` – structural issues (OOB, mismatched sizes, missing diag)
- `FactorizationError` – zero/near-zero pivot, numerical instability
- Both inherit from `std::runtime_error`

## References

- **QDLDL:** Davis & Hager, "Row modifications of sparse Cholesky factorization" (2016)
- **Supernodes:** Gilbert & Ng, "Predicting structure in sparse matrix computations" (1993)
- **AMD:** Amestoy, Davis & Duff, "An approximate minimum degree ordering algorithm" (1996)

---

**Maintained:** 2025 | C++23 standard | MIT/Apache-2.0 compatible
