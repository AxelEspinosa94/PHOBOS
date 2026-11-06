# Operaciones con tensores

**Módulo:** PHOBOS Core
**Header:** `ops.h`
**Implementación:** `ops.c`

Esta página documenta las operaciones básicas con tensores de PHOBOS: multiplicación de matrices, aritmética elemento a elemento con broadcasting y reducciones.

| Función | Categoría | Devuelve |
|---|---|---|
| [`tensor_matmul`](#tensor_matmul) | Álgebra lineal | Tensor nuevo |
| [`tensor_add`](#tensor_add-tensor_sub-tensor_mul) | Elemento a elemento (broadcasting) | Tensor nuevo |
| [`tensor_sub`](#tensor_add-tensor_sub-tensor_mul) | Elemento a elemento (broadcasting) | Tensor nuevo |
| [`tensor_mul`](#tensor_add-tensor_sub-tensor_mul) | Elemento a elemento (broadcasting) | Tensor nuevo |
| [`tensor_sum`](#tensor_sum) | Reducción | `float` |
| [`tensor_mean`](#tensor_mean) | Reducción | `float` |
| [`tensor_max`](#tensor_max) | Reducción | `float` |
| [`tensor_logreg_forward`](#tensor_logreg_forward) | Modelo (ML) | Tensor nuevo |

---

## Convenciones generales

Estas reglas aplican a todas las operaciones de esta página, salvo que se indique lo contrario.

- **Dtype soportado: solo `DTYPE_FLOAT32`.** Cualquier otro dtype hace que la función falle.
- **Las entradas nunca se modifican.** Todas se reciben como `const tensor_t*`.
- **Propiedad de la memoria.** Las funciones que devuelven `tensor_t*` reservan un tensor nuevo. Quien llama es dueño de ese tensor y debe liberarlo con `tensor_free`.
- **Manejo de errores.**
  - Las funciones que devuelven `tensor_t*` regresan `NULL` ante entradas inválidas o fallas de reserva de memoria.
  - Las reducciones que devuelven `float` regresan `0.0f` ante entradas inválidas. Ese valor no se puede distinguir de un resultado legítimo de `0.0f`.

---

## `tensor_matmul`

```c
tensor_t* tensor_matmul(const tensor_t* A, const tensor_t* B);
```

Calcula el producto matricial $C = AB$.

### Parámetros

| Nombre | Tipo | Descripción |
|---|---|---|
| `A` | `const tensor_t*` | Matriz izquierda, forma `(m, k)`. |
| `B` | `const tensor_t*` | Matriz derecha, forma `(k, n)`. |

### Valor de retorno

Un tensor `float32` nuevo de forma `(m, n)`, donde

$$
C_{ij} = \sum_{p=0}^{k-1} A_{ip}\,B_{pj}
$$

Devuelve `NULL` si:

- `A` o `B` es `NULL`;
- alguno de los tensores no es bidimensional;
- alguno de los tensores no es `float32`;
- las dimensiones internas no coinciden (`A->shape[1] != B->shape[0]`);
- falla la reserva de memoria del resultado.

### Notas

- **Sin broadcasting** ni multiplicación por lotes: ambas entradas deben ser exactamente 2-D.
- **Se asume que las entradas son contiguas (row-major).** Los elementos se leen como `a[i*k + p]` y `b[p*n + j]`, ignorando `strides`. Pasar una vista no contigua da resultados incorrectos sin ningún error.
- Implementación ingenua de triple ciclo, $O(mnk)$. La acumulación se hace en `float`.

### Ejemplo

```c
int sa[] = {2, 3}, sb[] = {3, 2};
tensor_t* A = tensor_create(DTYPE_FLOAT32, 2, sa);
tensor_t* B = tensor_create(DTYPE_FLOAT32, 2, sb);
/* ... llenar A y B ... */

tensor_t* C = tensor_matmul(A, B);   /* forma (2, 2) */

tensor_free(A); tensor_free(B); tensor_free(C);
```

---

## `tensor_add`, `tensor_sub`, `tensor_mul`

```c
tensor_t* tensor_add(const tensor_t* A, const tensor_t* B);
tensor_t* tensor_sub(const tensor_t* A, const tensor_t* B);
tensor_t* tensor_mul(const tensor_t* A, const tensor_t* B);
```

Suma, resta y multiplicación elemento a elemento con **broadcasting**.

| Función | Operación |
|---|---|
| `tensor_add` | $C = A + B$ |
| `tensor_sub` | $C = A - B$ |
| `tensor_mul` | $C = A \odot B$ (producto de Hadamard, **no** multiplicación matricial) |

### Parámetros

| Nombre | Tipo | Descripción |
|---|---|---|
| `A` | `const tensor_t*` | Primer operando. |
| `B` | `const tensor_t*` | Segundo operando. |

### Valor de retorno

Un tensor `float32` nuevo con la forma resultante del broadcasting de `A` y `B`. Devuelve `NULL` si alguna entrada es `NULL`, alguna no es `float32`, las formas no son compatibles para broadcasting o falla la reserva de memoria.

### Reglas de broadcasting

El broadcasting sigue la convención de NumPy:

1. Las formas se alinean desde la **última** dimensión hacia atrás.
2. Una dimensión inicial faltante se trata como tamaño 1.
3. Dos dimensiones son compatibles si son iguales o si una de ellas es 1.
4. La dimensión de salida es la mayor de las dos.

| Forma de `A` | Forma de `B` | Forma del resultado |
|---|---|---|
| `(3, 4)` | `(3, 4)` | `(3, 4)` |
| `(3, 4)` | `(4)` | `(3, 4)` |
| `(3, 1)` | `(1, 4)` | `(3, 4)` |
| `(m, 1)` | `(1, 1)` | `(m, 1)` |
| `(3, 4)` | `(3)` | ❌ incompatible → `NULL` |

### Notas

- Las entradas pueden ser **no contiguas**: los offsets se calculan a partir de los `strides` de cada tensor. La salida siempre es contigua.
- Las dimensiones con broadcasting reciben stride 0, así que el operando se lee repetidamente sin copiar datos.
- **Máximo 8 dimensiones.** La forma del resultado se guarda en arreglos de tamaño fijo 8. Entradas cuyo rango de broadcasting supera 8 no se rechazan y provocan un desbordamiento de buffer.

### Ejemplo

```c
/* Sumar un vector fila de sesgo a cada fila de una matriz */
int sx[] = {4, 3}, sb[] = {1, 3};
tensor_t* X    = tensor_create(DTYPE_FLOAT32, 2, sx);
tensor_t* bias = tensor_create(DTYPE_FLOAT32, 2, sb);
/* ... llenar X y bias ... */

tensor_t* Y = tensor_add(X, bias);   /* forma (4, 3) */

tensor_free(X); tensor_free(bias); tensor_free(Y);
```

---

## `tensor_sum`

```c
float tensor_sum(const tensor_t* A);
```

Devuelve la suma de todos los elementos de `A`:

$$
\text{sum}(A) = \sum_{i=0}^{N-1} a_i
$$

### Valor de retorno

La suma como `float`. Devuelve `0.0f` si `A` es `NULL` o no es `float32`.

### Notas

- Reduce sobre **todos** los elementos. No hay variante por eje.
- Asume datos contiguos: recorre `data[0 .. size-1]`, ignorando `strides`.
- Acumula en `float`, así que la precisión se degrada en tensores grandes o con valores de magnitudes muy distintas.

---

## `tensor_mean`

```c
float tensor_mean(const tensor_t* A);
```

Devuelve la media aritmética de todos los elementos de `A`:

$$
\text{mean}(A) = \frac{1}{N}\sum_{i=0}^{N-1} a_i
$$

### Valor de retorno

La media como `float`. Devuelve `0.0f` si `A` es `NULL`, no es `float32` o tiene `size == 0`.

### Notas

Se implementa como `tensor_sum(A) / size`, así que hereda las mismas limitaciones de contigüidad y precisión de `tensor_sum`.

---

## `tensor_max`

```c
float tensor_max(const tensor_t* A);
```

Devuelve el elemento más grande de `A`.

### Valor de retorno

El máximo como `float`. Devuelve `0.0f` si `A` es `NULL`, no es `float32` o tiene `size == 0`.

### Notas

- Asume datos contiguos.
- El manejo de `NaN` depende de su posición: un `NaN` en `data[0]` se devuelve (ninguna comparación con `NaN` es verdadera), mientras que un `NaN` en otra posición se ignora.
- No devuelve índice. No existe `argmax`.

---

## `tensor_logreg_forward`

```c
tensor_t* tensor_logreg_forward(const tensor_t* X,
                                const tensor_t* W,
                                const tensor_t* b);
```

Paso hacia adelante (forward) de la regresión logística:

$$
\hat{Y} = \sigma(XW + b), \qquad \sigma(z) = \frac{1}{1 + e^{-z}}
$$

> Esta es una operación de **modelo** y pertenece al módulo PHOBOS ML. Actualmente vive en `ops.c` y está programada para moverse a `ml/` como parte del EPIC de endurecimiento.

### Parámetros

| Nombre | Tipo | Descripción |
|---|---|---|
| `X` | `const tensor_t*` | Variables de entrada, forma `(m, n)`. |
| `W` | `const tensor_t*` | Pesos, vector columna de forma `(n, 1)`. |
| `b` | `const tensor_t*` | Sesgo, compatible por broadcasting con `(m, 1)`, por ejemplo `(1, 1)` o `(m, 1)`. Puede ser `NULL` (sin sesgo). |

### Valor de retorno

Un tensor `float32` nuevo de forma `(m, 1)` con probabilidades predichas en $(0, 1)$. Devuelve `NULL` si `X` o `W` es `NULL`, las formas son inválidas o falla algún paso intermedio.

### Notas

- Internamente llama a `tensor_matmul`, `tensor_add` y `tensor_sigmoid`. Los tensores intermedios se liberan antes de regresar.
- Hereda los requisitos de solo `float32` y de contigüidad de `tensor_matmul`.

---

## Funciones auxiliares internas

Estas funciones son `static` en `ops.c` y no forman parte de la API pública. Se listan para quienes mantienen el código.

| Función | Propósito |
|---|---|
| `broadcast_shapes` | Calcula la forma resultante del broadcasting de dos tensores. Devuelve 0 si son incompatibles. |
| `compute_broadcast_strides` | Calcula strides por dimensión **en elementos**, con 0 en las dimensiones con broadcasting. |
| `tensor_elemwise_broadcast` | Kernel genérico de broadcasting que usan `tensor_add`, `tensor_sub` y `tensor_mul` mediante un apuntador a función. |
| `tensor_same_shape` | Verifica si dos tensores tienen formas idénticas. Actualmente sin uso. |
| `tensor_elemwise_alloc_like` | Reserva un tensor con el mismo dtype y forma que otro. Actualmente sin uso. |
