# Gradients

**Module:** PHOBOS ML
**Implementation:** `gradients.c`

This page documents the gradient functions of PHOBOS. A gradient function computes the derivatives of a loss with respect to the model parameters, which an optimizer such as `optimizer_sgd_step` then uses to update them.

| Function | Model / loss | Returns |
|---|---|---|
| [`tensor_logreg_gradients`](#tensor_logreg_gradients) | Logistic regression + binary cross-entropy | `int` status code |

> Gradient functions are part of the **ML module**. Under the target architecture they live in `ml/`, and TLALOC does not depend on them.

---

## Where this fits in training

One training step of logistic regression chains four functions:

```
X, W, b ──► tensor_logreg_forward ──► y_hat
y, y_hat ─► tensor_bce_loss ────────► loss            (monitoring)
X, y, y_hat ► tensor_logreg_gradients ► dW, db
W, b, dW, db ► optimizer_sgd_step ───► W, b updated in place
```

---

## `tensor_logreg_gradients`

```c
int tensor_logreg_gradients(const tensor_t* X,
                            const tensor_t* y,
                            const tensor_t* y_hat,
                            tensor_t** dW_out,
                            float* db_out);
```

Computes the gradients of the binary cross-entropy loss with respect to the weights and bias of a logistic regression model.

### Derivation

For logistic regression, $\hat{y} = \sigma(z)$ with $z = XW + b$, and the mean binary cross-entropy is

$$
\mathcal{L} = -\frac{1}{m}\sum_{i=1}^{m}\Big[y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i)\Big]
$$

Because $\sigma'(z) = \sigma(z)\,(1 - \sigma(z))$, the derivative of each term with respect to $z_i$ simplifies to the prediction error:

$$
\frac{\partial \mathcal{L}}{\partial z_i} = \frac{1}{m}\,(\hat{y}_i - y_i)
$$

Applying the chain rule gives

$$
\nabla_W \mathcal{L} = \frac{1}{m}\,X^\top(\hat{y} - y), \qquad
\frac{\partial \mathcal{L}}{\partial b} = \frac{1}{m}\sum_{i=1}^{m}(\hat{y}_i - y_i)
$$

This is why the function needs only `X`, `y` and `y_hat`, and not `W` or `b`: the parameters enter only through `y_hat`.

### What the function computes

With the error vector $e = \hat{y} - y$:

| Output | Computed as | Matches mean-BCE gradient? |
|---|---|---|
| `dW` | $X^\top e$ (**sum** over samples) | ❌ Larger by a factor of $m$ |
| `db` | $\frac{1}{m}\sum_i e_i$ (**mean** over samples) | ✅ |

See [Notes](#notes) for the consequences of this inconsistency.

### Parameters

| Name | Type | Direction | Description |
|---|---|---|---|
| `X` | `const tensor_t*` | in | Input features, shape `(m, n)`. |
| `y` | `const tensor_t*` | in | Targets, shape `(m, 1)`. |
| `y_hat` | `const tensor_t*` | in | Predicted probabilities from `tensor_logreg_forward`, shape `(m, 1)`. |
| `dW_out` | `tensor_t**` | out | On success, receives a **new** tensor of shape `(n, 1)` with the weight gradient. |
| `db_out` | `float*` | out | On success, receives the bias gradient. |

### Return value

| Code | Meaning |
|---|---|
| `0` | Success. `*dW_out` and `*db_out` were written. |
| `-1` | One of the pointers is `NULL`. |
| `-2` | `X`, `y` or `y_hat` is not 2-dimensional. |
| `-3` | `y` or `y_hat` does not have `m` rows (`X->shape[0]`). |
| `-4` | `y` or `y_hat` is not a column vector (second dimension `!= 1`). |
| `-5` | Allocation of the internal error vector failed. |
| `-6` | Allocation of `dW` failed. |

On any error, `*dW_out` and `*db_out` are left untouched and no memory is leaked.

### Ownership

The tensor written to `*dW_out` is newly allocated. The caller owns it and must release it with `tensor_free`, typically right after the optimizer step.

### Notes

- **Scaling mismatch between `dW` and `db`.** `dW` is the sum over samples, while `db` is the mean. Paired with `tensor_bce_loss`, which is a mean, the weight gradient is $m$ times larger than the true gradient. In practice:
  - The effective learning rate for `W` is `lr · m`, while for `b` it is `lr`.
  - A learning rate tuned for one batch size behaves differently with another.
  - The fix is to divide `dW` by `m` as well.
- **Only valid for sigmoid + BCE.** The shortcut $\partial\mathcal{L}/\partial z = \hat{y} - y$ holds only for that combination. It is not a general-purpose gradient.
- **dtype is not checked.** All tensors are read as `float*`.
- **Contiguous data is assumed.** `X` is indexed as `X[i*n + j]`, ignoring `strides`.
- **Clamping in `tensor_bce_loss` is ignored.** The analytical gradient does not account for the $\varepsilon$ clamp. This only matters for predictions within $10^{-7}$ of 0 or 1.
- An intermediate `(m, 1)` tensor is allocated and freed on every call to hold $e$.
- The error codes are plain integers and do not match the meanings of `optimizer_sgd_step`'s codes (for example, `-2` means different things in each function).

### Example

```c
tensor_t* y_hat = tensor_logreg_forward(X, W, NULL);

tensor_t* dW = NULL;
float db = 0.0f;

int rc = tensor_logreg_gradients(X, y, y_hat, &dW, &db);
if (rc == 0) {
    optimizer_sgd_step(W, &b, dW, db, lr);
    tensor_free(dW);
}

tensor_free(y_hat);
```
