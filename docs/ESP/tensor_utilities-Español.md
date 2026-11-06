# Funciones utilitarias del tensor

**Módulo:** PHOBOS Core
**Header:** `tensor.h`
**Implementación:** `tensor.c`

Esta página documenta las funciones utilitarias de la API de tensores de PHOBOS. Ninguna crea ni modifica tensores. Responden preguntas sobre un tensor (tamaño de elemento, disposición en memoria, igualdad) o lo imprimen para depuración.

| Función | Propósito |
|---|---|
| [`tensor_dtype_size`](#tensor_dtype_size) | Tamaño en bytes de un elemento de un dtype dado |
| [`tensor_is_contiguous`](#tensor_is_contiguous) | Verifica si un tensor está almacenado de forma contigua (row-major) |
| [`tensor_print`](#tensor_print) | Imprime los metadatos del tensor en `stdout` |
| [`tensor_compare`](#tensor_compare) | Comparación exacta de metadatos y datos |

---

## `tensor_dtype_size`

```c
size_t tensor_dtype_size(dtype_t dtype);
```

Devuelve el tamaño en bytes de un solo elemento del tipo de dato indicado.

### Parámetros

| Nombre | Tipo | Descripción |
|---|---|---|
| `dtype` | `dtype_t` | Tipo de elemento a consultar. |

### Valor de retorno

| `dtype` | Devuelve |
|---|---|
| `DTYPE_FLOAT32` | `sizeof(float)` (4) |
| `DTYPE_FLOAT64` | `sizeof(double)` (8) |
| `DTYPE_INT32` | `sizeof(int32_t)` (4) |
| `DTYPE_INT64` | `sizeof(int64_t)` (8) |
| Cualquier otro valor | `0` |

### Comportamiento

- Un valor de retorno de `0` indica que el dtype es inválido o no está soportado. Quien llama debe tratar `0` como error. `tensor_create`, `tensor_from_data` y `tensor_is_contiguous` dependen de esta convención.
- La función no tiene efectos secundarios y se puede llamar desde cualquier contexto.

### Ejemplo

```c
size_t elem = tensor_dtype_size(DTYPE_FLOAT64);   /* 8 */

if (tensor_dtype_size(dtype) == 0) {
    /* dtype no soportado: manejar el error */
}
```

---

## `tensor_is_contiguous`

```c
bool tensor_is_contiguous(const tensor_t* t);
```

Verifica si los datos de un tensor están dispuestos en memoria de forma contigua en **orden row-major (C)**, es decir, si sus elementos ocupan un solo bloque continuo con la última dimensión variando más rápido.

### Parámetros

| Nombre | Tipo | Descripción |
|---|---|---|
| `t` | `const tensor_t*` | Tensor a inspeccionar. No se modifica. |

### Valor de retorno

- `true` si los strides coinciden con los de una disposición row-major contigua.
- `false` si la disposición no es contigua, **o** si `t`, `t->shape` o `t->strides` es `NULL`, **o** si `t->dtype` es inválido.

### Comportamiento

La función recorre las dimensiones de la última a la primera y verifica que cada stride (en bytes) sea igual al valor esperado:

```
stride[ndim-1] = elem_size
stride[i]      = stride[i+1] * shape[i+1]
```

Para un tensor `float32` de forma `[2, 3, 4]`, los strides esperados son `[48, 16, 4]` bytes.

### Notas

- Los tensores creados con `tensor_create` o `tensor_from_data` siempre son contiguos, porque ambas funciones calculan strides row-major.
- La verificación es estricta: un tensor cuyos strides describen una disposición contigua **column-major** (Fortran) devuelve `false`.
- Una dimensión de tamaño 1 debe tener exactamente el stride esperado, aunque ese stride nunca afecte el direccionamiento.
- `false` no distingue entre "no contiguo" y "entrada inválida". Si esa diferencia importa, valida `t` antes de llamar a la función.

### Ejemplo

```c
int shape[] = {2, 3, 4};
tensor_t* t = tensor_create(DTYPE_FLOAT32, 3, shape);

if (tensor_is_contiguous(t)) {
    /* es seguro tratar t->data como un arreglo plano de t->size elementos */
}

tensor_free(t);
```

---

## `tensor_print`

```c
void tensor_print(const tensor_t* t);
```

Imprime los **metadatos** de un tensor en `stdout`. Pensada para depuración.

### Parámetros

| Nombre | Tipo | Descripción |
|---|---|---|
| `t` | `const tensor_t*` | Tensor a imprimir. Puede ser `NULL`. |

### Salida

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

Si `t` es `NULL`, imprime:

```
tensor_t: NULL
```

### Notas

- **No imprime los valores de los elementos**, solo metadatos.
- `dtype` se imprime como el valor entero del enum, no como nombre.
- Los strides se expresan en **bytes**, no en elementos.
- Pasar `NULL` es seguro.

---

## `tensor_compare`

```c
bool tensor_compare(const tensor_t* a, const tensor_t* b);
```

Compara dos tensores buscando igualdad **exacta** de metadatos y datos. No imprime nada. Pensada para pruebas y depuración.

### Parámetros

| Nombre | Tipo | Descripción |
|---|---|---|
| `a` | `const tensor_t*` | Primer tensor. |
| `b` | `const tensor_t*` | Segundo tensor. |

### Valor de retorno

`true` si coinciden todos los siguientes campos, en este orden:

1. `dtype`
2. `ndim`
3. `shape` (cada dimensión)
4. `strides` (cada dimensión)
5. `size`
6. `bytes`
7. contenido crudo de los datos (`memcmp` sobre `bytes` bytes)

`false` en cuanto falla una verificación, o si `a` o `b` es `NULL`. Esto incluye el caso en que ambos son `NULL`.

### Notas

- **Comparación bit a bit.** Los datos se comparan con `memcmp`, así que los valores de punto flotante deben ser idénticos bit por bit:
  - `0.0` y `-0.0` se consideran **distintos**.
  - Dos `NaN` con el mismo patrón de bits se consideran **iguales**.
  - Valores que difieren solo por error de redondeo, como `0.1 + 0.2` frente a `0.3`, se consideran **distintos**.
- **La disposición importa.** Dos tensores con los mismos valores lógicos pero strides distintos se consideran distintos.
- `owns_data` **no** se compara. Un tensor y una vista no propietaria de los mismos datos pueden considerarse iguales.
- Se asume que `a->data` y `b->data` no son `NULL` y que abarcan al menos `bytes` bytes.

### Ejemplo

```c
int shape[] = {2, 2};
float values[] = {1.0f, 2.0f, 3.0f, 4.0f};

tensor_t* a = tensor_from_data(values, DTYPE_FLOAT32, 2, shape);
tensor_t* b = tensor_create(DTYPE_FLOAT32, 2, shape);
memcpy(b->data, values, sizeof(values));

assert(tensor_compare(a, b));   /* mismos metadatos, mismos bytes */

tensor_free(a);   /* no libera `values` (owns_data = false) */
tensor_free(b);
```