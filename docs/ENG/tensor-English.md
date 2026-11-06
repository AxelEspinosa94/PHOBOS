
---

# **PHOBOS‑ML — Tensor Module Documentation**

## **Overview**
The **`tensor`** module is the core data structure of PHOBOS‑ML.  
It defines how numerical data is represented in memory, including:

- data type  
- number of dimensions  
- shape  
- strides  
- total size  
- memory ownership  
- raw data access  

Every subsystem in the engine depends on this interface: operations (`ops`), autograd, model layers, optimizers, and external language bindings (Python, Rust, Java).

---

## **Tensor Structure**
The main type is:

```c
typedef struct {
    void* data;
    dtype_t dtype;
    int ndim;
    int* shape;
    size_t* strides;
    size_t size;
    size_t bytes;
    bool owns_data;
} tensor_t;
```

### **Field Description**
| Field | Type | Description |
|-------|------|-------------|
| `data` | `void*` | Pointer to raw memory where elements are stored. |
| `dtype` | `dtype_t` | Element type (float32, float64, int32, int64). |
| `ndim` | `int` | Number of dimensions. |
| `shape` | `int*` | Array containing the size of each dimension. |
| `strides` | `size_t*` | Array containing byte offsets per dimension. |
| `size` | `size_t` | Total number of elements. |
| `bytes` | `size_t` | Total memory footprint (size × sizeof(dtype)). |
| `owns_data` | `bool` | Indicates whether the tensor owns its memory or is a view. |

---

## **Data Types**
Supported element types are defined in `dtype_t`:

```c
typedef enum {
    DTYPE_FLOAT32,
    DTYPE_FLOAT64,
    DTYPE_INT32,
    DTYPE_INT64
} dtype_t;
```

These types ensure compatibility with:

- NumPy  
- PyTorch  
- TensorFlow Lite  
- JAX  

---

## **Memory Layout**
PHOBOS‑ML uses **row‑major (C‑style)** memory layout:

- The last dimension is contiguous  
- Strides are expressed in **bytes**, not elements  
- Enables slicing, views, and operations without copying memory  

Example for shape `[2, 3]` with `float32`:

```
strides[1] = 4 bytes
strides[0] = 3 * 4 = 12 bytes
```

---

## **Contiguity**
A tensor is contiguous if its strides follow:

```
stride[i] = sizeof(dtype) * product(shape[i+1 ... end])
```

Contiguity enables:

- vectorized operations  
- BLAS compatibility  
- optimized inner loops  

The function:

```c
bool tensor_is_contiguous(const tensor_t* t);
```

checks this automatically.

---

## **Creating Tensors**

### **1. Tensor with owned memory**
```c
tensor_t* tensor_create(dtype_t dtype, int ndim, const int* shape);
```

This function:

- allocates memory  
- computes strides  
- initializes metadata  
- sets `owns_data = true`  

### **2. Tensor wrapping external memory**
```c
tensor_t* tensor_from_data(void* data, dtype_t dtype, int ndim, const int* shape);
```

This function:

- does **not** allocate memory  
- wraps an external buffer  
- sets `owns_data = false`  

Useful for:

- Python CFFI  
- Rust FFI  
- Java JNI  
- embedding PHOBOS‑ML inside other systems  

---

## **Freeing Tensors**
```c
void tensor_free(tensor_t* t);
```

This function frees:

- `data` (only if `owns_data == true`)  
- `shape`  
- `strides`  
- the tensor struct  

Prevents memory leaks and double frees.

---

## **Utility Functions**

### **Element size**
```c
size_t tensor_dtype_size(dtype_t dtype);
```

Returns the size in bytes of a single element.

### **Debug printing**
```c
void tensor_print(const tensor_t* t);
```

Example output:

```
tensor_t {
  dtype: 0
  ndim: 2
  size: 6
  bytes: 24
  owns_data: true
  shape: [2, 3]
  strides (bytes): [12, 4]
}
```

---

## **Testing**
The file `tests/test_tensor.c` validates:

- tensor creation  
- contiguity  
- memory writes  
- freeing  
- metadata printing  

Run it with:

```bash
make build/test_tensor
./build/test_tensor
```

---

## **Design Goals**
The tensor module is designed with:

- **simplicity** — pure C, no dependencies  
- **interoperability** — FFI‑friendly layout  
- **performance** — contiguous layout, stride‑aware operations  
- **extensibility** — supports autograd, ops, models  
- **memory safety** — explicit ownership semantics  

---

## **Future Extensions**
The tensor design supports future features such as:

- slicing without copying  
- broadcasting  
- transposes  
- advanced views  
- BLAS integration  
- autograd graph nodes  
- vectorized kernels  

---

