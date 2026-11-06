# Training

**Module:** PHOBOS ML
**Header:** `train.h`
**Implementation:** `train.c`

This page documents the training functions of PHOBOS. They chain the forward pass, the loss, the gradients and the parameter update into complete training iterations for logistic regression.

| Function | Scope | Returns |
|---|---|---|
| [`tensor_logreg_train_step`](#tensor_logreg_train_step) | One training iteration | `int` status code |
| [`train_loop`](#train_loop) | Several epochs, with progress printed to `stdout` | `int` status code |

> Training functions are part of the **ML module**. Under the target architecture they live in `ml/`, and TLALOC does not depend on them.

---

## ⚠️ Known issue: the two functions train different models

Logistic regression is defined as $\hat{y} = \sigma(XW + b)$. Neither function currently computes exactly that, and they differ from each other:

| | Prediction actually computed | Bias used in prediction? |
|---|---|---|
| Expected model | $\sigma(XW + b)$ | ✅ |
| `tensor_logreg_train_step` | $\sigma\big(\sigma(XW) + b\big)$ | ✅, but after a first sigmoid |
| `train_loop` | $\sigma(XW)$ | ❌ (updated but never used) |

Meanwhile, `tensor_logreg_gradients` computes gradients for the expected model. In both functions the gradients therefore do not correspond to the predictions being made. Details are in each function's notes.

---

## General conventions

- **Parameters are updated in place.** `W` and `*b` are overwritten. No new parameters are returned.
- **Inputs `X` and `y` are read-only.**
- **No memory is leaked.** All intermediate tensors (`y_hat`, `dW`) are allocated and freed internally, also on error paths.
- **Error signaling.** `0` on success, a negative code on error.
- **Batch.** Each iteration uses the **entire** dataset (full-batch gradient descent), not random mini-batches.

---

## `tensor_logreg_train_step`

```c
int tensor_logreg_train_step(const tensor_t* X,
                             const tensor_t* y,
                             tensor_t* W,
                             float* b,
                             float lr,
                             float* loss_out);
```

Performs a single training iteration: forward pass, loss, gradients and gradient-descent update.

### Parameters

| Name | Type | Direction | Description |
|---|---|---|---|
| `X` | `const tensor_t*` | in | Input features, shape `(m, n)`. |
| `y` | `const tensor_t*` | in | Targets in $\{0, 1\}$, shape `(m, 1)`. |
| `W` | `tensor_t*` | in/out | Weights, shape `(n, 1)`. Updated in place. |
| `b` | `float*` | in/out | Scalar bias. Updated in place. |
| `lr` | `float` | in | Learning rate $\eta$. |
| `loss_out` | `float*` | out | Receives the loss computed **before** the update. |

### Steps

1. **Forward:** `tensor_logreg_forward(X, W, NULL)` → $\sigma(XW)$.
2. **Bias:** adds `*b` to every element of the result, then applies `tensor_sigmoid` again → $\sigma(\sigma(XW) + b)$.
3. **Loss:** `tensor_bce_loss(y, y_hat)`, written to `*loss_out`.
4. **Gradients:** `tensor_logreg_gradients(X, y, y_hat, &dW, &db)`.
5. **Update:** $W \leftarrow W - \eta\,dW$, $\; b \leftarrow b - \eta\,db$, applied inline.

### Return value

| Code | Meaning | `W`, `*b` modified? | `*loss_out` written? |
|---|---|---|---|
| `0` | Success. | Yes | Yes |
| `-1` | A pointer argument is `NULL`. | No | No |
| `-2` | The forward pass failed (invalid shapes, dtype, or allocation). | No | No |
| `-3` | The second sigmoid failed (allocation). | No | No |
| `-4` | Gradient computation failed. | No | **Yes** |

### Notes

- **Double sigmoid (incorrect model).** The forward pass already applies a sigmoid, so the bias is added to a probability and squashed a second time. With $b = 0$, every prediction lies in $(0.5,\ 0.731)$, because $\sigma(0) = 0.5$ and $\sigma(1) \approx 0.731$. The model cannot predict class 0 with confidence, and the gradients from `tensor_logreg_gradients` do not match this function.
- **Inline update.** The update duplicates the logic of `optimizer_sgd_step` instead of calling it.
- **Inherited issues.** The weight gradient from `tensor_logreg_gradients` is $m$ times larger than the true gradient (see the gradients documentation).
- The loss is not validated. If `tensor_bce_loss` fails, `*loss_out` is `0.0f` and training continues.

---

## `train_loop`

```c
int train_loop(const tensor_t* X,
               const tensor_t* y,
               tensor_t* W,
               float* b,
               float lr,
               int epochs);
```

Runs `epochs` iterations of full-batch gradient descent and prints the loss and bias after each epoch.

### Parameters

| Name | Type | Direction | Description |
|---|---|---|---|
| `X` | `const tensor_t*` | in | Input features, shape `(m, n)`. |
| `y` | `const tensor_t*` | in | Targets in $\{0, 1\}$, shape `(m, 1)`. |
| `W` | `tensor_t*` | in/out | Weights, shape `(n, 1)`. Updated in place. |
| `b` | `float*` | in/out | Scalar bias. Updated in place. |
| `lr` | `float` | in | Learning rate $\eta$. |
| `epochs` | `int` | in | Number of iterations. If `<= 0`, nothing happens and `0` is returned. |

### Each epoch

1. `tensor_logreg_forward(X, W, NULL)` → $\hat{y} = \sigma(XW)$.
2. `tensor_bce_loss(y, y_hat)`.
3. `tensor_logreg_gradients(X, y, y_hat, &dW, &db)`.
4. `optimizer_sgd_step(W, b, dW, db, lr)`.
5. Prints one line to `stdout`:

```
[Epoch 1/100] Loss = 0.693147 Bias=-0.001234
```

### Return value

| Code | Meaning |
|---|---|
| `0` | All epochs completed. |
| `-1` | A pointer argument is `NULL`. |
| `-2` | The forward pass failed. |
| `-3` | Gradient computation failed. |
| `-4` | The optimizer step failed. |

On error, training stops immediately. Updates from the epochs already completed are kept in `W` and `*b`.

### Notes

- **The bias is trained but never used.** The forward pass is called with `NULL` as the bias, so $\hat{y} = \sigma(XW)$. `db` is computed and `*b` is updated every epoch, and its value is printed, but it never affects a prediction. The model is effectively a logistic regression without intercept.
- **The printed loss is the pre-update loss** of each epoch.
- **Output is always printed.** There is no way to silence it, and `printf` is used without `#include <stdio.h>` in `train.c`.
- **No loss history is returned.** Only the printed output records the evolution of the loss. There is no early stopping.
- **Inherited issues.** The weight gradient is $m$ times larger than the true gradient (see the gradients documentation).
- Despite using `optimizer_sgd_step`, the algorithm is full-batch gradient descent: every epoch uses all `m` samples.

### Example

```c
float b = 0.0f;
int rc = train_loop(X, y, W, &b, 0.1f, 100);
if (rc != 0) {
    fprintf(stderr, "Training failed (code %d)\n", rc);
}
```
