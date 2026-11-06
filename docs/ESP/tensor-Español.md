
---

# **PHOBOS‑ML — Tensor Documentation**

## **Overview**
El módulo **`tensor`** es el núcleo de PHOBOS‑ML.  
Define la estructura fundamental para representar datos numéricos en memoria, incluyendo:

- tipo de dato  
- número de dimensiones  
- shape  
- strides  
- tamaño total  
- propiedad de memoria  
- acceso a datos  

Todos los módulos del engine dependen de esta interfaz: operaciones (`ops`), autograd, modelos, optimizadores y bindings externos (Python, Rust, Java).

---

## **Tensor Structure**
El tipo principal es:

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

### **Campos**
| Campo | Tipo | Descripción |
|-------|------|-------------|
| `data` | `void*` | Apuntador a la memoria cruda donde viven los elementos. |
| `dtype` | `dtype_t` | Tipo de dato (float32, float64, int32, int64). |
| `ndim` | `int` | Número de dimensiones del tensor. |
| `shape` | `int*` | Arreglo con el tamaño de cada dimensión. |
| `strides` | `size_t*` | Arreglo con el desplazamiento en bytes por dimensión. |
| `size` | `size_t` | Número total de elementos. |
| `bytes` | `size_t` | Tamaño total en memoria (size × sizeof(dtype)). |
| `owns_data` | `bool` | Indica si el tensor es dueño de la memoria o solo una vista. |

---

## **Data Types**
Los tipos soportados están definidos en `dtype_t`:

```c
typedef enum {
    DTYPE_FLOAT32,
    DTYPE_FLOAT64,
    DTYPE_INT32,
    DTYPE_INT64
} dtype_t;
```

Esto permite interoperabilidad con:

- NumPy  
- PyTorch  
- TensorFlow Lite  
- JAX  

---

## **Memory Layout**
PHOBOS‑ML usa **row-major (C‑style)**:

- La última dimensión es contigua  
- Los strides se expresan en **bytes**, no en elementos  
- Permite slicing, views y operaciones sin copiar memoria  

Ejemplo para shape `[2, 3]` con `float32`:

```
strides[1] = 4 bytes
strides[0] = 3 * 4 = 12 bytes
```

---

## **Contiguity**
Un tensor es contiguo si sus strides siguen el patrón:

```
stride[i] = sizeof(dtype) * product(shape[i+1 ... end])
```

Esto permite:

- operaciones vectorizadas  
- compatibilidad con BLAS  
- optimización de loops internos  

La función:

```c
bool tensor_is_contiguous(const tensor_t* t);
```

lo valida automáticamente.

---

## **Creating Tensors**
### **1. Tensor con memoria propia**
```c
tensor_t* tensor_create(dtype_t dtype, int ndim, const int* shape);
```

Hace:

- asignación de memoria  
- cálculo de strides  
- inicialización de metadata  
- marca `owns_data = true`  

### **2. Tensor desde memoria externa**
```c
tensor_t* tensor_from_data(void* data, dtype_t dtype, int ndim, const int* shape);
```

Hace:

- NO asigna memoria  
- envuelve un buffer externo  
- marca `owns_data = false`  

Ideal para:

- interoperar con Python (CFFI)  
- interoperar con Rust (FFI)  
- interoperar con Java (JNI)  

---

## **Freeing Tensors**
```c
void tensor_free(tensor_t* t);
```

Libera:

- `data` (solo si `owns_data == true`)  
- `shape`  
- `strides`  
- la estructura del tensor  

Evita memory leaks y doble free.

---

## **Utility Functions**
### **Tamaño del tipo**
```c
size_t tensor_dtype_size(dtype_t dtype);
```

### **Impresión para debugging**
```c
void tensor_print(const tensor_t* t);
```

Ejemplo de salida:

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
El archivo `tests/test_tensor.c` valida:

- creación  
- contiguidad  
- escritura en memoria  
- liberación  
- impresión  

Se ejecuta con:

```bash
make build/test_tensor
./build/test_tensor
```

---

## **Design Goals**
El diseño del tensor sigue principios de:

- **simplicidad** (C puro, sin dependencias)  
- **interoperabilidad** (FFI friendly)  
- **performance** (strides, contiguidad, layout C)  
- **extensibilidad** (autograd, ops, modelos)  
- **seguridad** (propiedad de memoria clara)  

---

## **Future Extensions**
El tensor está preparado para:

- slicing sin copiar  
- broadcasting  
- transposes  
- views avanzadas  
- integración con BLAS  
- integración con autograd  
- operaciones vectorizadas  

---

