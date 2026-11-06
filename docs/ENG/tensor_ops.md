# Tensor Operations

**Module:** PHOBOS Core
**Header:** `ops.h`
**Implementation:** `ops.c`

This page documents the basic tensor operations of PHOBOS: matrix multiplication, element-wise arithmetic with broadcasting, and reductions.

| Function | Category | Returns |
|---|---|---|
| [`tensor_matmul`](#tensor_matmul) | Linear algebra | New tensor |
| [`tensor_add`](#tensor_add-tensor_sub-tensor_mul) | Element-wise (broadcasting) | New tensor |
| [`tensor_sub`](#tensor_add-tensor_sub-tensor_mul) | Element-wise (broadcasting) | New tensor |
| [`tensor_mul`](#tensor_add-tensor_sub-tensor_mul) | Element-wise (broadcasting) | New tensor |
| [`tensor_sum`](#tensor_sum) | Reduction | `float` |
| [`tensor_mean`](#tensor_mean) | Reduction | `float` |
| [`tensor_max`](#tensor_max) | Reduction | `float` |
| [`tensor_logreg_forward`](#tensor_logreg_forward) | Model (ML) | New tensor |

---

## General conventions

These rules apply to every operation on this page unless stated otherwise.

- **Supported dtype: `DTYPE_FLOAT32` only.** Any other dtype makes the function fail.
- **Inputs are never modified.** All inputs are taken as `const tensor_t*`.
- **Ownership.** Functions that return a `tensor_t*` allocate a new tensor. The caller owns it and must release it with `tensor_free`.
- **Error signaling.**
  - Functions returning `tensor_t*` return `NULL` on invalid input or allocation failure.
  - Reductions returning `float` return `0.0f` on invalid input. This value cannot be distinguished from a legitimate result of `0.0f`.

---

## `tensor_matmul`

```c
tensor_t* tensor_matmul(const tensor_t* A, const tensor_t* B);
```

Computes the matrix product $C = AB$.

### Parameters

| Name | Type | Description |
|---|---|---|
| `A` | `const tensor_t*` | Left matrix, shape `(m, k)`. |
| `B` | `const tensor_t*` | Right matrix, shape `(k, n)`. |

### Return value

A new `float32` tensor of shape `(m, n)`, where

$$
C_{ij} = \sum_{p=0}^{k-1} A_{ip}\,B_{pj}
$$

Returns `NULL` if:

- `A` or `B` is `NULL`;
- either tensor is not 2-dimensional;
- either tensor is not `float32`;
- the inner dimensions do not match (`A->shape[1] != B->shape[0]`);
- allocation of the result fails.

### Notes

- **No broadcasting** and no batched multiplication: both inputs must be exactly 2-D.
- **Inputs are assumed to be contiguous (row-major).** Elements are read as `a[i*k + p]` and `b[p*n + j]`, ignoring `strides`. Passing a non-contiguous view gives incorrect results without any error.
- Naive triple-loop implementation, $O(mnk)$. Accumulation is done in `float`.

### Example

```c
int sa[] = {2, 3}, sb[] = {3, 2};
tensor_t* A = tensor_create(DTYPE_FLOAT32, 2, sa);
tensor_t* B = tensor_create(DTYPE_FLOAT32, 2, sb);
/* ... fill A and B ... */

tensor_t* C = tensor_matmul(A, B);   /* shape (2, 2) */

tensor_free(A); tensor_free(B); tensor_free(C);
```

---

## `tensor_add`, `tensor_sub`, `tensor_mul`

```c
tensor_t* tensor_add(const tensor_t* A, const tensor_t* B);
tensor_t* tensor_sub(const tensor_t* A, const tensor_t* B);
tensor_t* tensor_mul(const tensor_t* A, const tensor_t* B);
```

Element-wise addition, subtraction and multiplication with **broadcasting**.

| Function | Operation |
|---|---|
| `tensor_add` | $C = A + B$ |
| `tensor_sub` | $C = A - B$ |
| `tensor_mul` | $C = A \odot B$ (Hadamard product, **not** matrix multiplication) |

### Parameters

| Name | Type | Description |
|---|---|---|
| `A` | `const tensor_t*` | First operand. |
| `B` | `const tensor_t*` | Second operand. |

### Return value

A new `float32` tensor with the broadcast shape of `A` and `B`. Returns `NULL` if either input is `NULL`, either is not `float32`, the shapes are not broadcast-compatible, or allocation fails.

### Broadcasting rules

Broadcasting follows the NumPy convention:

1. Shapes are aligned from the **last** dimension backwards.
2. A missing leading dimension is treated as size 1.
3. Two dimensions are compatible if they are equal or if one of them is 1.
4. The output dimension is the larger of the two.

| `A` shape | `B` shape | Result shape |
|---|---|---|
| `(3, 4)` | `(3, 4)` | `(3, 4)` |
| `(3, 4)` | `(4)` | `(3, 4)` |
| `(3, 1)` | `(1, 4)` | `(3, 4)` |
| `(m, 1)` | `(1, 1)` | `(m, 1)` |
| `(3, 4)` | `(3)` | ❌ incompatible → `NULL` |

### Notes

- Inputs may be **non-contiguous**: offsets are computed from each tensor's `strides`. The output is always contiguous.
- Broadcast dimensions get a stride of 0, so the broadcast operand is read repeatedly without copying any data.
- **Maximum of 8 dimensions.** The result shape is held in fixed-size arrays of length 8. Inputs whose broadcast rank exceeds 8 are not rejected and cause a buffer overflow.

### Example

```c
/* Add a bias row vector to every row of a matrix */
int sx[] = {4, 3}, sb[] = {1, 3};
tensor_t* X    = tensor_create(DTYPE_FLOAT32, 2, sx);
tensor_t* bias = tensor_create(DTYPE_FLOAT32, 2, sb);
/* ... fill X and bias ... */

tensor_t* Y = tensor_add(X, bias);   /* shape (4, 3) */

tensor_free(X); tensor_free(bias); tensor_free(Y);
```

---

## `tensor_sum`

```c
float tensor_sum(const tensor_t* A);
```

Returns the sum of all elements of `A`:

$$
\text{sum}(A) = \sum_{i=0}^{N-1} a_i
$$

### Return value

The sum as a `float`. Returns `0.0f` if `A` is `NULL` or not `float32`.

### Notes

- Reduces over **all** elements. There is no per-axis variant.
- Assumes contiguous data: iterates `data[0 .. size-1]`, ignoring `strides`.
- Accumulates in `float`, so precision degrades for large tensors or values of very different magnitude.

---

## `tensor_mean`

```c
float tensor_mean(const tensor_t* A);
```

Returns the arithmetic mean of all elements of `A`:

$$
\text{mean}(A) = \frac{1}{N}\sum_{i=0}^{N-1} a_i
$$

### Return value

The mean as a `float`. Returns `0.0f` if `A` is `NULL`, not `float32`, or has `size == 0`.

### Notes

Implemented as `tensor_sum(A) / size`, so it inherits the same contiguity and precision limitations as `tensor_sum`.

---

## `tensor_max`

```c
float tensor_max(const tensor_t* A);
```

Returns the largest element of `A`.

### Return value

The maximum as a `float`. Returns `0.0f` if `A` is `NULL`, not `float32`, or has `size == 0`.

### Notes

- Assumes contiguous data.
- `NaN` handling depends on position: a `NaN` at `data[0]` is returned (no comparison with `NaN` succeeds), while a `NaN` elsewhere is ignored.
- No index is returned. There is no `argmax`.

---

## `tensor_logreg_forward`

```c
tensor_t* tensor_logreg_forward(const tensor_t* X,
                                const tensor_t* W,
                                const tensor_t* b);
```

Forward pass of logistic regression:

$$
\hat{Y} = \sigma(XW + b), \qquad \sigma(z) = \frac{1}{1 + e^{-z}}
$$

> This is a **model** operation and belongs to the PHOBOS ML module. It currently lives in `ops.c` and is scheduled to move to `ml/` as part of the hardening EPIC.

### Parameters

| Name | Type | Description |
|---|---|---|
| `X` | `const tensor_t*` | Input features, shape `(m, n)`. |
| `W` | `const tensor_t*` | Weights, column vector of shape `(n, 1)`. |
| `b` | `const tensor_t*` | Bias, broadcastable to `(m, 1)`, e.g. `(1, 1)` or `(m, 1)`. May be `NULL` (no bias). |

### Return value

A new `float32` tensor of shape `(m, 1)` with predicted probabilities in $(0, 1)$. Returns `NULL` if `X` or `W` is `NULL`, shapes are invalid, or any intermediate step fails.

### Notes

- Internally calls `tensor_matmul`, `tensor_add` and `tensor_sigmoid`. Intermediate tensors are freed before returning.
- Inherits the `float32`-only and contiguity requirements of `tensor_matmul`.

---

## Internal helpers

These functions are `static` in `ops.c` and are not part of the public API. They are listed for maintainers.

| Helper | Purpose |
|---|---|
| `broadcast_shapes` | Computes the broadcast result shape of two tensors. Returns 0 if incompatible. |
| `compute_broadcast_strides` | Computes per-dimension strides **in elements**, with 0 for broadcast dimensions. |
| `tensor_elemwise_broadcast` | Generic broadcasting kernel applied by `tensor_add`, `tensor_sub` and `tensor_mul` via a function pointer. |
| `tensor_same_shape` | Checks whether two tensors have identical shapes. Currently unused. |
| `tensor_elemwise_alloc_like` | Allocates a tensor with the same dtype and shape as another. Currently unused. |
