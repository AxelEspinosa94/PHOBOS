# Activation Functions

**Module:** PHOBOS Core
**Implementation:** `ops_activation.c`

This page documents the activation functions of PHOBOS, plus the `tensor_copy` helper they are built on. Every activation returns a **new** tensor with the same shape as its input. The input is never modified.

| Function | Definition | Output range |
|---|---|---|
| [`tensor_copy`](#tensor_copy) | Deep copy | — |
| [`tensor_relu`](#tensor_relu) | $\max(x, 0)$ | $[0, \infty)$ |
| [`tensor_sigmoid`](#tensor_sigmoid) | $\dfrac{1}{1 + e^{-x}}$ | $[0, 1]$ |
| [`tensor_softmax`](#tensor_softmax) | $\dfrac{e^{x_i}}{\sum_j e^{x_j}}$ over the last dimension | $[0, 1]$, each row sums to 1 |

> ReLU, sigmoid and softmax are used by the ML module (e.g. `tensor_logreg_forward`). `tensor_copy` is a general tensor utility. Under the target architecture, `tensor_copy` belongs with the core tensor API in `tensor.c`.

---

## General conventions

- **Supported dtype: `DTYPE_FLOAT32` only** (activations). `tensor_copy` accepts any dtype but only copies correctly for 4-byte types; see its notes.
- **Inputs are never modified.** Each activation first copies its input with `tensor_copy`, then transforms the copy in place.
- **Ownership.** The returned tensor is new. The caller owns it and must release it with `tensor_free`.
- **Error signaling.** All functions return `NULL` on invalid input or allocation failure.
- **Contiguous data is assumed.** Elements are processed as `data[0 .. size-1]`, ignoring `strides`.

---

## `tensor_copy`

```c
tensor_t* tensor_copy(const tensor_t* src);
```

Creates a deep copy of `src`: a new tensor with the same dtype and shape, and its own copy of the data.

### Parameters

| Name | Type | Description |
|---|---|---|
| `src` | `const tensor_t*` | Tensor to copy. |

### Return value

A new tensor with `owns_data = true` and contiguous row-major strides. Returns `NULL` if `src` is `NULL` or allocation fails.

### Notes

- **Copies `size * sizeof(float)` bytes regardless of dtype.** Correct for `float32` and `int32`. For `float64` and `int64`, only the first half of the data is copied and the rest of the result is left uninitialized.
- **Non-contiguous sources are not handled.** The raw buffer is copied in storage order, but the result gets fresh contiguous strides. For a non-contiguous `src`, the logical values of the copy are wrong.
- The result always owns its data, even when `src` is a non-owning view created with `tensor_from_data`.

### Example

```c
tensor_t* b = tensor_copy(a);   /* b is independent of a */
/* modifying b->data does not affect a */
tensor_free(b);
```

---

## `tensor_relu`

```c
tensor_t* tensor_relu(const tensor_t* A);
```

Applies the rectified linear unit element-wise:

$$
\operatorname{ReLU}(x) = \max(x, 0) =
\begin{cases}
x & x > 0 \\
0 & x \le 0
\end{cases}
$$

### Parameters

| Name | Type | Description |
|---|---|---|
| `A` | `const tensor_t*` | Input tensor, any shape. |

### Return value

A new `float32` tensor with the same shape as `A`. Returns `NULL` if `A` is `NULL`, not `float32`, or allocation fails.

### Notes

- **`NaN` becomes `0`.** The test `x > 0.0f` is false for `NaN`, so `NaN` inputs are silently replaced by `0.0f` instead of propagating. This can hide numerical problems upstream.
- `-0.0f` becomes `+0.0f`.
- Its derivative, needed for backpropagation, is $1$ for $x > 0$ and $0$ otherwise. It is not computed here.

### Example

```c
/* A = [-2.0, -0.5, 0.0, 1.5] */
tensor_t* R = tensor_relu(A);   /* [0.0, 0.0, 0.0, 1.5] */
tensor_free(R);
```

---

## `tensor_sigmoid`

```c
tensor_t* tensor_sigmoid(const tensor_t* A);
```

Applies the logistic sigmoid element-wise:

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

### Parameters

| Name | Type | Description |
|---|---|---|
| `A` | `const tensor_t*` | Input tensor, any shape. |

### Return value

A new `float32` tensor with the same shape as `A`. Returns `NULL` if `A` is `NULL`, not `float32`, or allocation fails.

### Numerical stability

Computing $e^{-x}$ directly overflows for large negative $x$. The implementation uses two algebraically equivalent forms, each evaluated where its exponent is non-positive:

$$
\sigma(x) =
\begin{cases}
\dfrac{1}{1 + e^{-x}} & x \ge 0 \\[2ex]
\dfrac{e^{x}}{1 + e^{x}} & x < 0
\end{cases}
$$

Because the argument passed to `expf` is never positive, it can underflow to $0$ but never overflow to $\infty$.

### Notes

- **Saturation in `float32`.** Mathematically $\sigma(x) \in (0, 1)$, but in single precision the result rounds to exactly `1.0f` for $x \gtrsim 17$ and reaches `0.0f` for very negative $x$. Losses that take $\log \sigma(x)$, such as `tensor_bce_loss`, must clamp their input. `tensor_bce_loss` already does.
- `NaN` inputs propagate as `NaN`.
- $\sigma(0) = 0.5$, and $\sigma(-x) = 1 - \sigma(x)$.

### Example

```c
/* A = [-100.0, 0.0, 2.0] */
tensor_t* S = tensor_sigmoid(A);   /* [≈0.0, 0.5, ≈0.881] */
tensor_free(S);
```

---

## `tensor_softmax`

```c
tensor_t* tensor_softmax(const tensor_t* A);
```

Applies softmax along the **last dimension**. Each slice along that dimension (a "row") is turned into a probability distribution:

$$
\operatorname{softmax}(x)_i = \frac{e^{x_i - m}}{\sum_{j} e^{x_j - m}}, \qquad m = \max_j x_j
$$

### Parameters

| Name | Type | Description |
|---|---|---|
| `A` | `const tensor_t*` | Input tensor of shape `(..., K)`. Softmax is computed independently over each group of `K` consecutive elements. |

### Return value

A new `float32` tensor with the same shape as `A`, where every row has non-negative entries summing to 1 (up to rounding). Returns `NULL` if `A` is `NULL`, not `float32`, the last dimension is not positive, or allocation fails.

### Behavior

For each of the `size / K` rows:

1. Find the row maximum $m$.
2. Replace each element by $e^{x_i - m}$ and accumulate the sum.
3. Divide each element by the sum.

Subtracting $m$ does not change the result, since it cancels between numerator and denominator, but it keeps every exponent $\le 0$ and so prevents overflow.

| Input shape | Normalized over | Rows |
|---|---|---|
| `(K)` | all `K` elements | 1 |
| `(m, K)` | each row | `m` |
| `(b, m, K)` | each innermost row | `b·m` |

### Notes

- **Only the last dimension is supported.** There is no `axis` parameter.
- **Rows are assumed to be contiguous** (`row = data + o*K`), so a non-contiguous input gives wrong results.
- **Non-finite values.** If a row contains `+inf`, then $\infty - \infty = $ `NaN` and the whole row becomes `NaN`. The same happens if every element of a row is `-inf`. A `NaN` anywhere in a row also yields `NaN`s.
- The `sum > 0` guard before normalizing never triggers for finite input, because the maximum element always contributes $e^0 = 1$ to the sum.
- Accumulation is done in `float`.

### Example

```c
/* A = [[1.0, 2.0, 3.0],
        [1.0, 1.0, 1.0]]   shape (2, 3) */
tensor_t* P = tensor_softmax(A);
/* P ≈ [[0.090, 0.245, 0.665],
        [0.333, 0.333, 0.333]] */
tensor_free(P);
```

---

## Internal helpers

| Helper | Purpose |
|---|---|
| `sigmoid_stable` | `static` scalar sigmoid using the two-branch stable form described above. |
