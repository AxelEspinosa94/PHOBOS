# Optimizers

**Module:** PHOBOS ML
**Header:** `optimizer.h`
**Implementation:** `optimizer.c`

This page documents the optimizers of PHOBOS. An optimizer takes the current parameters of a model and their gradients, and updates the parameters to reduce the loss.

| Function | Algorithm | Returns |
|---|---|---|
| [`optimizer_sgd_step`](#optimizer_sgd_step) | Stochastic gradient descent (one step) | `int` status code |

> Optimizers are part of the **ML module**. Under the target architecture they live in `ml/`, and TLALOC does not depend on them.

---

## General conventions

- **Parameters are updated in place.** Unlike the tensor operations, optimizers do **not** return a new tensor: they overwrite the parameters they receive.
- **Gradients are read-only.** They are taken as `const`.
- **No allocation.** Nothing is created, so there is nothing to free.
- **Error signaling.** Optimizers return an `int` status code: `0` on success, a negative value on error. On error, no parameter is modified.

---

## `optimizer_sgd_step`

```c
int optimizer_sgd_step(tensor_t* W,
                       float* b,
                       const tensor_t* dW,
                       float db,
                       float lr);
```

Performs **one step** of vanilla stochastic gradient descent on a weight tensor and a scalar bias.

### Update rule

$$
W \leftarrow W - \eta\,\nabla_W \mathcal{L}, \qquad
b \leftarrow b - \eta\,\frac{\partial \mathcal{L}}{\partial b}
$$

where $\eta$ is the learning rate. Element-wise, $W_i \leftarrow W_i - \eta\, dW_i$ for every element $i$.

### Parameters

| Name | Type | Direction | Description |
|---|---|---|---|
| `W` | `tensor_t*` | in/out | Weight tensor. Overwritten with the updated weights. |
| `b` | `float*` | in/out | Pointer to the scalar bias. Overwritten with the updated bias. |
| `dW` | `const tensor_t*` | in | Gradient of the loss with respect to `W`. Same number of elements as `W`. |
| `db` | `float` | in | Gradient of the loss with respect to `b`. |
| `lr` | `float` | in | Learning rate $\eta$. Normally a small positive value such as `0.01f`. |

### Return value

| Code | Meaning |
|---|---|
| `0` | Success. `W` and `*b` were updated. |
| `-1` | `W`, `b` or `dW` is `NULL`. Nothing was modified. |
| `-2` | `W->size != dW->size`. Nothing was modified. |

### Behavior

- Each call applies exactly one update. The training loop is responsible for computing `dW` and `db` (e.g. by backpropagation or the mini-autograd) and calling this function once per batch.
- If `W` is a non-owning view created with `tensor_from_data`, the update writes directly into the caller's buffer.

### Notes

- **dtype is not checked.** Both tensors are read as `float*`. Passing a `float64` or integer tensor gives silently wrong results.
- **Only `size` is checked, not `shape`.** A `dW` of shape `(1, n)` is accepted for a `W` of shape `(n, 1)` and applied in flat memory order. Make sure both have the same shape.
- **Contiguous data is assumed.** Elements are processed as `data[0 .. size-1]`, ignoring `strides`.
- **`lr` is not validated.** A negative learning rate performs gradient *ascent*. A `NaN` learning rate turns every parameter into `NaN`.
- **The bias is a scalar.** Models with one bias per output, such as a bias tensor of shape `(m, 1)` passed to `tensor_logreg_forward`, cannot be updated with this function.
- The error codes are plain integers. There are no named constants for them yet.

### Example

```c
float lr = 0.1f;
float b  = 0.0f;

for (int epoch = 0; epoch < n_epochs; ++epoch) {
    /* forward + loss + gradients (dW, db) computed here */

    int rc = optimizer_sgd_step(W, &b, dW, db, lr);
    if (rc != 0) {
        fprintf(stderr, "SGD step failed (code %d)\n", rc);
        break;
    }
}
```
