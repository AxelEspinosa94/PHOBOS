# Tensor Utility Functions

**Module:** PHOBOS Core
**Header:** `tensor.h`
**Implementation:** `tensor.c`

This page documents the utility functions of the PHOBOS tensor API. None of them create or modify tensors. They answer questions about a tensor (element size, memory layout, equality) or print it for debugging.

| Function | Purpose |
|---|---|
| [`tensor_dtype_size`](#tensor_dtype_size) | Size in bytes of one element of a given dtype |
| [`tensor_is_contiguous`](#tensor_is_contiguous) | Checks whether a tensor is stored contiguously (row-major) |
| [`tensor_print`](#tensor_print) | Prints tensor metadata to `stdout` |
| [`tensor_compare`](#tensor_compare) | Exact equality check of metadata and data |

---

## `tensor_dtype_size`

```c
size_t tensor_dtype_size(dtype_t dtype);
```

Returns the size in bytes of a single element of the given data type.

### Parameters

| Name | Type | Description |
|---|---|---|
| `dtype` | `dtype_t` | Element type to query. |

### Return value

| `dtype` | Returns |
|---|---|
| `DTYPE_FLOAT32` | `sizeof(float)` (4) |
| `DTYPE_FLOAT64` | `sizeof(double)` (8) |
| `DTYPE_INT32` | `sizeof(int32_t)` (4) |
| `DTYPE_INT64` | `sizeof(int64_t)` (8) |
| Any other value | `0` |

### Behavior

- A return value of `0` means the dtype is invalid or not supported. Callers should treat `0` as an error. `tensor_create`, `tensor_from_data` and `tensor_is_contiguous` all rely on this convention.
- The function has no side effects and is safe to call from any context.

### Example

```c
size_t elem = tensor_dtype_size(DTYPE_FLOAT64);   /* 8 */

if (tensor_dtype_size(dtype) == 0) {
    /* unsupported dtype: handle error */
}
```

---

## `tensor_is_contiguous`

```c
bool tensor_is_contiguous(const tensor_t* t);
```

Checks whether a tensor's data is laid out contiguously in memory in **row-major (C) order**, that is, whether its elements occupy one unbroken block with the last dimension varying fastest.

### Parameters

| Name | Type | Description |
|---|---|---|
| `t` | `const tensor_t*` | Tensor to inspect. Not modified. |

### Return value

- `true` if the strides match those of a row-major contiguous layout.
- `false` if the layout is not contiguous, **or** if `t`, `t->shape` or `t->strides` is `NULL`, **or** if `t->dtype` is invalid.

### Behavior

The function walks the dimensions from last to first and checks that each stride (in bytes) equals the expected value:

```
stride[ndim-1] = elem_size
stride[i]      = stride[i+1] * shape[i+1]
```

For a `float32` tensor of shape `[2, 3, 4]`, the expected strides are `[48, 16, 4]` bytes.

### Notes

- Tensors created with `tensor_create` or `tensor_from_data` are always contiguous, because both compute row-major strides.
- The check is strict: a tensor whose strides describe a contiguous **column-major** (Fortran) layout returns `false`.
- A dimension of size 1 must still have the exact expected stride, even though its stride never affects addressing.
- `false` does not distinguish "non-contiguous" from "invalid input". If that difference matters, validate `t` beforehand.

### Example

```c
int shape[] = {2, 3, 4};
tensor_t* t = tensor_create(DTYPE_FLOAT32, 3, shape);

if (tensor_is_contiguous(t)) {
    /* safe to treat t->data as a flat array of t->size elements */
}

tensor_free(t);
```

---

## `tensor_print`

```c
void tensor_print(const tensor_t* t);
```

Prints a tensor's **metadata** to `stdout`. Intended for debugging.

### Parameters

| Name | Type | Description |
|---|---|---|
| `t` | `const tensor_t*` | Tensor to print. May be `NULL`. |

### Output

```
tensor_t {
  dtype: 0
  ndim: 3
  size: 24
  bytes: 96
  owns_data: true
  shape: [2, 3, 4]
  strides (bytes): [48, 16, 4]
}
```

If `t` is `NULL`, it prints:

```
tensor_t: NULL
```

### Notes

- **Element values are not printed**, only metadata.
- `dtype` is printed as its integer enum value, not as a name.
- Strides are expressed in **bytes**, not in elements.
- Passing `NULL` is safe.

---

## `tensor_compare`

```c
bool tensor_compare(const tensor_t* a, const tensor_t* b);
```

Compares two tensors for **exact** equality of both metadata and data. Prints nothing. Intended for tests and debugging.

### Parameters

| Name | Type | Description |
|---|---|---|
| `a` | `const tensor_t*` | First tensor. |
| `b` | `const tensor_t*` | Second tensor. |

### Return value

`true` if every one of the following matches, in this order:

1. `dtype`
2. `ndim`
3. `shape` (every dimension)
4. `strides` (every dimension)
5. `size`
6. `bytes`
7. raw data contents (`memcmp` over `bytes` bytes)

`false` as soon as one check fails, or if either `a` or `b` is `NULL`. This includes the case where both are `NULL`.

### Notes

- **Bitwise comparison.** Data is compared with `memcmp`, so floating-point values must be bit-for-bit identical:
  - `0.0` and `-0.0` compare as **different**.
  - Two `NaN`s with the same bit pattern compare as **equal**.
  - Values that differ only by rounding error, such as `0.1 + 0.2` versus `0.3`, compare as **different**.
- **Layout matters.** Two tensors with the same logical values but different strides compare as different.
- `owns_data` is **not** compared. A tensor and a non-owning view of identical data can compare equal.
- `a->data` and `b->data` are assumed to be non-`NULL` and to span at least `bytes` bytes.

### Example

```c
int shape[] = {2, 2};
float values[] = {1.0f, 2.0f, 3.0f, 4.0f};

tensor_t* a = tensor_from_data(values, DTYPE_FLOAT32, 2, shape);
tensor_t* b = tensor_create(DTYPE_FLOAT32, 2, shape);
memcpy(b->data, values, sizeof(values));

assert(tensor_compare(a, b));   /* same metadata, same bytes */

tensor_free(a);   /* does not free `values` (owns_data = false) */
tensor_free(b);
```