# Model Persistence

**Module:** PHOBOS ML
**Header:** `model_io.h`
**Implementation:** `model_io.c`

This page documents how PHOBOS saves a trained logistic regression model to disk and loads it back.

| Function | Purpose | Returns |
|---|---|---|
| [`model_save`](#model_save) | Writes weights and bias to a binary file | `int` status code |
| [`model_load`](#model_load) | Reads weights and bias into an existing tensor | `int` status code |

> Model persistence is part of the **ML module**. Under the target architecture it lives in `ml/`, and TLALOC does not depend on it.

---

## File format

Models are stored in a raw binary file with the following layout, written in order and without padding:

| Field | Type | Size (x86-64 Linux) | Description |
|---|---|---|---|
| Magic | `char[8]` | 8 B | The ASCII bytes `PHOBOSML`, with no terminating `\0`. |
| `ndim` | `int` | 4 B | Number of dimensions of `W`. |
| `shape` | `int[ndim]` | 4·`ndim` B | Size of each dimension of `W`. |
| `size` | `size_t` | 8 B | Total number of elements of `W`. |
| `data` | `float[size]` | 4·`size` B | Weights, in memory (row-major) order. |
| `b` | `float` | 4 B | Scalar bias. |

**Total size:** $24 + 4\cdot\text{ndim} + 4\cdot\text{size}$ bytes. For example, a weight vector of shape `(10, 1)` produces a 72-byte file.

### Format properties

- **Native representation.** Integers and floats are written exactly as they are in memory. The file depends on the machine's endianness and on `sizeof(int)` and `sizeof(size_t)`. A file is guaranteed to load only on the same kind of platform that wrote it.
- **No version field.** Any change to the layout makes old files unreadable, with no way to detect which format a file uses.
- **No dtype field.** The data is always interpreted as `float32`.
- `size` is redundant with `shape` (it is their product). It is stored as an extra consistency check.

---

## General conventions

- **Status codes.** Both functions return `0` on success and a negative value on error.
- **No allocation.** `model_load` does not create a tensor. It fills one that the caller has already created with the expected shape.
- **Only `W` and a scalar `b` are stored.** No training state (learning rate, epoch, loss) is saved.

---

## `model_save`

```c
int model_save(const char* path, const tensor_t* W, float b);
```

Writes the weights `W` and the bias `b` to the file at `path`, replacing it if it exists.

### Parameters

| Name | Type | Description |
|---|---|---|
| `path` | `const char*` | Destination file path. |
| `W` | `const tensor_t*` | Weight tensor to save. Expected to be `float32` and contiguous. |
| `b` | `float` | Bias to save. |

### Return value

| Code | Meaning |
|---|---|
| `0` | `fopen` succeeded and all writes were issued. See notes: this does **not** guarantee the file is complete. |
| `-1` | `path` or `W` is `NULL`. |
| `-2` | The file could not be opened for writing (missing directory, permissions, etc.). |

### Notes

- **Write errors are not detected.** The return values of `fwrite` and `fclose` are ignored. If the disk fills up or an I/O error occurs, the function still returns `0` and the file is left incomplete.
- **Not atomic.** The existing file is truncated as soon as it is opened. If the program crashes during the save, both the old model and the new one are lost.
- **dtype is not checked.** `size · sizeof(float)` bytes are written regardless of `W->dtype`. For a `float64` tensor only the first half of the data is written.
- **Contiguous data is assumed.** `W->data` is written as one block, ignoring `strides`.

### Example

```c
if (model_save("spam_model.bin", W, b) != 0) {
    fprintf(stderr, "Could not save model\n");
}
```

---

## `model_load`

```c
int model_load(const char* path, tensor_t* W, float* b);
```

Reads a model from `path` into an **existing** tensor `W` and the float pointed to by `b`. The file must describe exactly the same shape as `W`.

### Parameters

| Name | Type | Direction | Description |
|---|---|---|---|
| `path` | `const char*` | in | Path of the file to read. |
| `W` | `tensor_t*` | in/out | Pre-allocated `float32` tensor with the expected shape. Its data is overwritten. |
| `b` | `float*` | out | Receives the stored bias. |

### Validation order

The header is checked field by field before any data is written into `W`:

1. Magic bytes equal `PHOBOSML`.
2. `ndim` equals `W->ndim`.
3. Every entry of `shape` equals `W->shape[i]`.
4. `size` equals `W->size`.

Only if all four checks pass are the weights and bias read.

### Return value

| Code | Meaning | `W` / `*b` modified? |
|---|---|---|
| `0` | Header matched and all reads were issued. See notes: this does **not** guarantee the data is complete. | Yes |
| `-1` | `path`, `W` or `b` is `NULL`. | No |
| `-2` | The file could not be opened. | No |
| `-3` | Wrong magic bytes: not a PHOBOS model file. | No |
| `-4` | `ndim` in the file differs from `W->ndim`. | No |
| `-5` | `shape` in the file differs from `W->shape`. | No |
| `-6` | `size` in the file differs from `W->size`. | No |

### Notes

- **Truncated files are not detected.** The return values of `fread` are ignored. A file cut off in the middle of the weights still returns `0`: `W` ends up partially overwritten with the new values and partially keeping its old ones, and `*b` is left unchanged. For instance, a 72-byte `(10, 1)` model truncated to 40 bytes loads "successfully" with only 3 of its 10 weights updated.
- **The shape must be known in advance.** The function cannot load a model of unknown shape, because it never allocates. The caller must create `W` with the right shape first.
- **dtype is not checked.** Data is read as `float32` regardless of `W->dtype`.
- **Trailing data is ignored.** Extra bytes after the bias are not reported.
- **Variable-length array.** The shape is read into `int shape[ndim]` on the stack. `ndim` is bounded by `W->ndim` before the array is declared, so this is safe, but VLAs are optional since C11 and unsupported by MSVC.
- The magic string `PHOBOSML` predates the rename of the project to PHOBOS. Changing it would make every existing model file unreadable.

### Example

```c
int shape[] = {n_features, 1};
tensor_t* W = tensor_create(DTYPE_FLOAT32, 2, shape);
float b = 0.0f;

int rc = model_load("spam_model.bin", W, &b);
if (rc != 0) {
    fprintf(stderr, "Could not load model (code %d)\n", rc);
}
```
