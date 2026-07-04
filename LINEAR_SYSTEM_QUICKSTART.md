# Linear System Tools Quick Start

## Build & Test (1 minute)

```bash
cd /data/dev/chomp
g++ -O3 -std=c++23 -I./include tests/test_linear_system_standalone.cpp -o test_ls
./test_ls
```

✓ All 13 tests pass in <1 second.

## Basic Usage (QDLDL)

```cpp
#include "linear_system/qdldl.h"
using namespace qdldl23;

// CSC upper: A = [ 4 1 ]
//                [ 1 3 ]
// col 0: [0], col 1: [0,1]
int32_t n = 2;
auto A = SparseUpperCSC<double, int32_t>(
    n,
    {0, 1, 3},           // Ap (column pointers)
    {0, 0, 1},           // Ai (row indices, i ≤ j)
    {4.0, 1.0, 3.0}      // Ax (values)
);

// Factor: A ≈ L D L^T
auto F = factorize(A);

// Solve: A x = b
std::vector<double> b = {6.0, 7.0};
solve(F, b.data());  // b now contains x = [1, 2]
```

## Workflows

### 1. Simple solve
```cpp
auto F = factorize(A);
solve(F, x.data());
```

### 2. Reorder + solve
```cpp
auto ord = Ordering::from_perm(amd_perm);
auto B = permute_symmetric_upper(A, ord);
auto F = factorize(B);
solve_with_ordering(F, ord, x.data());
```

### 3. Iterative refinement
```cpp
auto F = factorize(A);
solve(F, x.data());
refine(A, F, x.data(), b.data(), /*iters=*/2);
```

### 4. Supernode analysis
```cpp
auto S = analyze_fast(A);
auto sn = snode::identify_supernodes_qdldl(A, S);
// Use sn.ranges for supernode-level updates
```

### 5. AMD reordering
```cpp
AMDReorderingArray amd;
auto [perm, stats] = amd.compute_fill_reducing_permutation(csr_A, true);
std::cout << "Bandwidth reduction: " << stats.bandwidth_reduction << "%\n";
```

## Format: CSC Upper Triangular

For an n×n SPD matrix:
- **Ap** (size n+1): column start pointers
- **Ai** (size nnz): row indices (each i ≤ j for entry in column j)
- **Ax** (size nnz): numerical values

All **diagonals must be present** (checked at construction).

### Example: 3×3
```
A = [ 4  1  0.5]
    [ 1  3  0.2]
    [0.5 0.2 2 ]

Ap = [0, 1, 3, 6]  (col 0 starts at 0, col 1 at 1, col 2 at 3, end at 6)
Ai = [0, 0, 1, 0, 1, 2]  (col 0: row 0; col 1: rows 0,1; col 2: rows 0,1,2)
Ax = [4.0, 1.0, 3.0, 0.5, 0.2, 2.0]
```

## Optimization Flags

Automatically detected; no flags needed:
- AVX-512 if available (8 doubles/op)
- AVX2 if available (4 doubles/op)
- SSE2 fallback (2 doubles/op)
- Scalar if none available

## Error Handling

```cpp
try {
    auto F = factorize(A);
    solve(F, x.data());
} catch (const qdldl23::FactorizationError& e) {
    std::cerr << "Pivot is zero: " << e.what() << "\n";
} catch (const qdldl23::InvalidMatrixError& e) {
    std::cerr << "Bad format: " << e.what() << "\n";
}
```

## Key Type Aliases

| Alias | Type |
|-------|------|
| `SparseD32` | `SparseUpperCSC<double, int32_t>` |
| `SparseD64` | `SparseUpperCSC<double, int64_t>` |
| `LDL32` | `LDLFactors<double, int32_t>` |
| `Symb32` | `Symbolic<int32_t>` |

## Files

| File | Purpose |
|------|---------|
| `include/linear_system/qdldl.h` | Main factorization (1167 lines) |
| `include/linear_system/supernodes.h` | Supernode detection (231 lines) |
| `include/linear_system/amd.h` | Fill-reducing reordering (1203 lines) |
| `include/linear_system/README.md` | Full API reference |
| `tests/test_linear_system_standalone.cpp` | 13 unit tests (all pass ✓) |
| `tests/test_linear_system.py` | pytest framework (optional) |

## Verify Installation

```bash
# Should print "13/13 passed"
./test_ls
```

---

**All code is header-only C++23. No external dependencies except includes.**
