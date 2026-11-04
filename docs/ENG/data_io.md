# Data I/O (CSV)

**Module:** PHOBOS Core
**Header:** `data_io.h`
**Implementation:** `data_io.c`

This page documents how PHOBOS reads numeric CSV files into tensors and writes tensors back to CSV.

| Function | Purpose | Returns |
|---|---|---|
| [`tensor_load_csv`](#tensor_load_csv) | Reads a numeric CSV file into a 2-D `float32` tensor | New tensor or `NULL` |
| [`tensor_save_csv`](#tensor_save_csv) | Writes a 2-D `float32` tensor to a CSV file | `int` status code |

> Data I/O is general-purpose and does not depend on the ML module. TLALOC can use it, for example, to export velocity profiles for plotting.

---

## Supported CSV dialect

Only a minimal subset of CSV is supported:

| Feature | Supported? |
|---|---|
| Comma (`,`) as separator | ✅ (the only separator) |
| Numeric values only (integers, decimals, scientific notation, `nan`, `inf`) | ✅ |
| One optional header line (skipped, not parsed) | ✅ |
| `\n` and `\r\n` line endings | ✅ |
| Last line without a final newline | ✅ |
| Spaces **after** a comma (`1, 2`) | ✅ |
| Spaces **before** a comma (`1 ,2`) | ❌ file rejected |
| Quoted fields, text columns, other separators (`;`, tab) | ❌ file rejected |
| Empty fields (`1,,3`) | ⚠️ **accepted but loaded incorrectly**; see notes |
| Blank lines, including a blank line at the end of the file | ❌ file rejected |
| Lines longer than 4095 characters | ❌ file rejected (in practice) |

---

## `tensor_load_csv`

```c
tensor_t* tensor_load_csv(const char* path, int skip_header);
```

Reads a numeric CSV file and returns its contents as a 2-D `float32` tensor of shape `(rows, cols)`.

### Parameters

| Name | Type | Description |
|---|---|---|
| `path` | `const char*` | Path of the CSV file. |
| `skip_header` | `int` | If non-zero, the first line is skipped without being parsed. |

### Return value

A new contiguous `float32` tensor of shape `(rows, cols)`. The caller owns it and must release it with `tensor_free`.

Returns `NULL` if:

- `path` is `NULL` or the file cannot be opened;
- `skip_header` is set and the file is empty;
- the file has no data rows;
- any row has a different number of commas than the first data row;
- any field cannot be fully parsed as a number;
- allocation fails.

### How it works

The file is read **twice**:

1. **First pass:** counts rows and checks that every row has the same number of columns (commas + 1).
2. The tensor is allocated with that shape.
3. **Second pass:** splits each line on commas with `strtok` and converts each field with `strtof`. A field is valid only if `strtof` consumes it entirely, allowing at most a trailing `\n` or `\r`.

### Notes

- **Empty fields corrupt the result silently.** The first pass counts commas, so `1,,3` has 3 columns. The second pass uses `strtok`, which skips empty tokens and finds only 2 fields. The values are written into the wrong columns and the last column of that row is left **uninitialized**. For example, loading `1,,3` / `4,5,6` returns `[[1, 3, ?], [4, 5, 6]]`, where `?` is whatever was in memory. No error is reported.
- **Errors carry no information.** Every failure returns `NULL`, so a caller cannot tell a missing file from a malformed row or tell which line failed.
- **A blank line rejects the whole file.** This includes a single extra blank line at the end of the file, which many editors and tools produce.
- **Line length limit.** Lines are read into a 4096-byte buffer. A longer line is split across several reads, which in practice makes the column check fail and the whole file be rejected. Assuming about 4 characters per value, this limits files to roughly 1000 columns.
- **Not usable with pipes or `stdin`.** The two-pass design relies on `rewind`, which requires a regular, seekable file.
- **Uses `strtok`.** It keeps hidden global state, so the function is not reentrant or thread-safe.
- **Locale-dependent.** `strtof` follows the program's `LC_NUMERIC` locale. If a program calls `setlocale` with a locale that uses a decimal comma, parsing breaks.
- Any value accepted by `strtof` is valid, including `nan`, `inf` and hexadecimal floats.

### Example

```c
/* data.csv:
   x1,x2,label
   0.5,1.2,1
   0.1,0.3,0          */

tensor_t* D = tensor_load_csv("data.csv", 1);   /* shape (2, 3) */
if (!D) {
    fprintf(stderr, "Could not load data.csv\n");
}
/* ... */
tensor_free(D);
```

---

## `tensor_save_csv`

```c
int tensor_save_csv(const char* path, const tensor_t* tensor);
```

Writes a 2-D tensor to a CSV file, one row per line, replacing the file if it exists.

### Parameters

| Name | Type | Description |
|---|---|---|
| `path` | `const char*` | Destination file path. |
| `tensor` | `const tensor_t*` | 2-D tensor to write. Expected to be `float32` and contiguous. |

### Return value

| Code | Meaning |
|---|---|
| `0` | The file was opened and all rows were issued. See notes: this does **not** guarantee the file is complete. |
| `-1` | `path` or `tensor` is `NULL`. |
| `-2` | `tensor` is not 2-dimensional. |
| `-3` | The file could not be opened for writing. |

### Output format

- Values are written with `printf`'s `%f`: fixed-point with **6 decimal places**.
- Columns are separated by `,` with no spaces. Each row ends with `\n`.
- **No header line is written.**

### Notes

- **Precision loss.** `%f` always prints exactly 6 decimals, so small values are lost and large ones get spurious digits. Saving the values `1e-7`, `0.1` and `123456.789` produces `0.000000,0.100000,123456.789062`. A save–load round trip does not reproduce the original tensor. Using `%.9g` would preserve every `float32` value exactly.
- **Write errors are not detected.** The return values of `fprintf` and `fclose` are ignored. If the disk fills up, the function still returns `0` with an incomplete file.
- **dtype is not checked.** Data is read as `float*` regardless of `tensor->dtype`.
- **Contiguous data is assumed.** Elements are read as `data[r*cols + c]`, ignoring `strides`.
- **Asymmetry with loading.** Because no header is written, loading the saved file with `skip_header = 1` silently drops the first data row.

### Example

```c
if (tensor_save_csv("predictions.csv", y_hat) != 0) {
    fprintf(stderr, "Could not save predictions\n");
}
```
