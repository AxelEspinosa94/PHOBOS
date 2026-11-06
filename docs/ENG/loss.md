# Loss Functions

**Module:** PHOBOS ML
**Header:** `loss.h`
**Implementation:** `loss.c`

This page documents the loss functions of PHOBOS. They reduce a set of predictions and their targets to a single scalar that measures how wrong the model is.

| Function | Task | Returns |
|---|---|---|
| [`tensor_bce_loss`](#tensor_bce_loss) | Binary classification | `float` |

> Loss functions are part of the **ML module**. Under the target architecture they live in `ml/`, and TLALOC does not depend on them.

---

## General conventions

- **Supported dtype: `DTYPE_FLOAT32` only.**
- **Inputs are never modified.** Both are taken as `const tensor_t*`.
- **No allocation.** Losses return a plain `float`, so there is nothing to free.
- **Error signaling.** On invalid input the function returns `0.0f`. For a loss this is especially misleading, because `0.0f` is also the value of a **perfect** prediction. See [Notes](#notes).

---

## `tensor_bce_loss`

```c
float tensor_bce_loss(const tensor_t* y, const tensor_t* p);
```

Computes the **mean binary cross-entropy** between targets `y` and predicted probabilities `p`.

### Definition

$$
\mathcal{L}(y, p) = -\frac{1}{N}\sum_{i=0}^{N-1}\Big[\, y_i \log \tilde{p}_i + (1 - y_i)\log(1 - \tilde{p}_i) \,\Big]
$$

where $N$ is the number of elements and $\tilde{p}_i$ is the prediction clamped to avoid $\log(0)$:

$$
\tilde{p}_i = \min\big(\max(p_i,\ \varepsilon),\ 1 - \varepsilon\big), \qquad \varepsilon = 10^{-7}
$$

### Parameters

| Name | Type | Description |
|---|---|---|
| `y` | `const tensor_t*` | Targets. Normally `0.0` or `1.0`. Values in $[0, 1]$ (soft labels) are also valid. |
| `p` | `const tensor_t*` | Predicted probabilities, normally the output of a sigmoid (e.g. `tensor_logreg_forward`). Expected in $[0, 1]$. |

### Return value

The mean loss as a non-negative `float`. Lower is better.

Returns `0.0f` if:

- `y` or `p` is `NULL`;
- either tensor is not `float32`;
- `y->size != p->size`.

### Behavior

- **Reduction: mean.** The sum is divided by $N$, so the loss does not grow with batch size.
- **Clamping.** Each prediction is clamped to $[\varepsilon, 1 - \varepsilon]$ before taking logarithms. Consequences:
  - The loss is always finite, even for predictions of exactly `0.0` or `1.0`.
  - The maximum contribution of a single element is about $-\log(10^{-7}) \approx 16.12$.
  - Inside the clamped region the loss stops changing with $p$, so its gradient there is zero.
- **Interpretation.** A model that always predicts $p = 0.5$ has a loss of $\log 2 \approx 0.693$. This is a useful baseline: a trained model should be below it.

### Notes

- **Only `size` is checked, not `shape`.** Tensors are compared element by element in flat memory order, so `y` of shape `(m, 1)` and `p` of shape `(1, m)` are accepted. Make sure both have the same shape.
- **Contiguous data is assumed.** Elements are read as `data[0 .. size-1]`, ignoring `strides`.
- **Values are not validated.** Targets outside $[0, 1]$ are not rejected and produce meaningless (possibly negative) losses. Predictions outside $[0, 1]$ are silently clamped. A `NaN` in `p` propagates to the result.
- **Ambiguous error value.** Because `0.0f` means both "error" and "perfect prediction", an invalid call can look like a fully trained model. Validate inputs before calling.
- Accumulation is done in `float`.

### Example

```c
int shape[] = {4, 1};
float y_vals[] = {1.0f, 0.0f, 1.0f, 0.0f};
float p_vals[] = {0.9f, 0.2f, 0.7f, 0.1f};

tensor_t* y = tensor_from_data(y_vals, DTYPE_FLOAT32, 2, shape);
tensor_t* p = tensor_from_data(p_vals, DTYPE_FLOAT32, 2, shape);

float loss = tensor_bce_loss(y, p);   /* ≈ 0.198 */

tensor_free(y);
tensor_free(p);
```

### Typical use

```c
tensor_t* p = tensor_logreg_forward(X, W, b);   /* predictions (m, 1) */
float loss  = tensor_bce_loss(y, p);            /* y: targets (m, 1)  */
tensor_free(p);
```
