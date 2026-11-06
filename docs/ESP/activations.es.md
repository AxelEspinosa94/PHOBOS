# Funciones de activación

**Módulo:** PHOBOS Core
**Implementación:** `ops_activation.c`

Esta página documenta las funciones de activación de PHOBOS, además de la función auxiliar `tensor_copy` en la que se apoyan. Cada activación devuelve un tensor **nuevo** con la misma forma que su entrada. La entrada nunca se modifica.

| Función | Definición | Rango de salida |
|---|---|---|
| [`tensor_copy`](#tensor_copy) | Copia profunda | — |
| [`tensor_relu`](#tensor_relu) | $\max(x, 0)$ | $[0, \infty)$ |
| [`tensor_sigmoid`](#tensor_sigmoid) | $\dfrac{1}{1 + e^{-x}}$ | $[0, 1]$ |
| [`tensor_softmax`](#tensor_softmax) | $\dfrac{e^{x_i}}{\sum_j e^{x_j}}$ sobre la última dimensión | $[0, 1]$, cada fila suma 1 |

> ReLU, sigmoide y softmax las usa el módulo de ML (p. ej. `tensor_logreg_forward`). `tensor_copy` es una utilidad general de tensores. En la arquitectura objetivo, `tensor_copy` pertenece a la API central de tensores en `tensor.c`.

---

## Convenciones generales

- **Dtype soportado: solo `DTYPE_FLOAT32`** (activaciones). `tensor_copy` acepta cualquier dtype, pero solo copia correctamente tipos de 4 bytes; ver sus notas.
- **Las entradas nunca se modifican.** Cada activación primero copia su entrada con `tensor_copy` y luego transforma la copia en su lugar.
- **Propiedad de la memoria.** El tensor devuelto es nuevo. Quien llama es su dueño y debe liberarlo con `tensor_free`.
- **Manejo de errores.** Todas las funciones devuelven `NULL` ante entradas inválidas o fallas de reserva de memoria.
- **Se asumen datos contiguos.** Los elementos se procesan como `data[0 .. size-1]`, ignorando `strides`.

---

## `tensor_copy`

```c
tensor_t* tensor_copy(const tensor_t* src);
```

Crea una copia profunda de `src`: un tensor nuevo con el mismo dtype y forma, y con su propia copia de los datos.

### Parámetros

| Nombre | Tipo | Descripción |
|---|---|---|
| `src` | `const tensor_t*` | Tensor a copiar. |

### Valor de retorno

Un tensor nuevo con `owns_data = true` y strides row-major contiguos. Devuelve `NULL` si `src` es `NULL` o falla la reserva de memoria.

### Notas

- **Copia `size * sizeof(float)` bytes sin importar el dtype.** Es correcto para `float32` e `int32`. Para `float64` e `int64` solo se copia la primera mitad de los datos y el resto del resultado queda sin inicializar.
- **No maneja fuentes no contiguas.** El buffer crudo se copia en orden de almacenamiento, pero el resultado recibe strides contiguos nuevos. Si `src` no es contiguo, los valores lógicos de la copia son incorrectos.
- El resultado siempre es dueño de sus datos, incluso cuando `src` es una vista no propietaria creada con `tensor_from_data`.

### Ejemplo

```c
tensor_t* b = tensor_copy(a);   /* b es independiente de a */
/* modificar b->data no afecta a a */
tensor_free(b);
```

---

## `tensor_relu`

```c
tensor_t* tensor_relu(const tensor_t* A);
```

Aplica la unidad lineal rectificada elemento a elemento:

$$
\operatorname{ReLU}(x) = \max(x, 0) =
\begin{cases}
x & x > 0 \\
0 & x \le 0
\end{cases}
$$

### Parámetros

| Nombre | Tipo | Descripción |
|---|---|---|
| `A` | `const tensor_t*` | Tensor de entrada, de cualquier forma. |

### Valor de retorno

Un tensor `float32` nuevo con la misma forma que `A`. Devuelve `NULL` si `A` es `NULL`, no es `float32` o falla la reserva de memoria.

### Notas

- **`NaN` se convierte en `0`.** La prueba `x > 0.0f` es falsa para `NaN`, así que las entradas `NaN` se reemplazan sin aviso por `0.0f` en lugar de propagarse. Esto puede ocultar problemas numéricos previos.
- `-0.0f` se convierte en `+0.0f`.
- Su derivada, necesaria para backpropagation, es $1$ para $x > 0$ y $0$ en otro caso. Aquí no se calcula.

### Ejemplo

```c
/* A = [-2.0, -0.5, 0.0, 1.5] */
tensor_t* R = tensor_relu(A);   /* [0.0, 0.0, 0.0, 1.5] */
tensor_free(R);
```

---

## `tensor_sigmoid`

```c
tensor_t* tensor_sigmoid(const tensor_t* A);
```

Aplica la sigmoide logística elemento a elemento:

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

### Parámetros

| Nombre | Tipo | Descripción |
|---|---|---|
| `A` | `const tensor_t*` | Tensor de entrada, de cualquier forma. |

### Valor de retorno

Un tensor `float32` nuevo con la misma forma que `A`. Devuelve `NULL` si `A` es `NULL`, no es `float32` o falla la reserva de memoria.

### Estabilidad numérica

Calcular $e^{-x}$ directamente desborda para $x$ muy negativos. La implementación usa dos formas algebraicamente equivalentes, cada una evaluada donde su exponente es no positivo:

$$
\sigma(x) =
\begin{cases}
\dfrac{1}{1 + e^{-x}} & x \ge 0 \\[2ex]
\dfrac{e^{x}}{1 + e^{x}} & x < 0
\end{cases}
$$

Como el argumento que recibe `expf` nunca es positivo, puede caer a $0$ por underflow pero nunca desbordar a $\infty$.

### Notas

- **Saturación en `float32`.** Matemáticamente $\sigma(x) \in (0, 1)$, pero en precisión simple el resultado se redondea exactamente a `1.0f` para $x \gtrsim 17$ y llega a `0.0f` para $x$ muy negativos. Las pérdidas que calculan $\log \sigma(x)$, como `tensor_bce_loss`, deben acotar su entrada. `tensor_bce_loss` ya lo hace.
- Las entradas `NaN` se propagan como `NaN`.
- $\sigma(0) = 0.5$ y $\sigma(-x) = 1 - \sigma(x)$.

### Ejemplo

```c
/* A = [-100.0, 0.0, 2.0] */
tensor_t* S = tensor_sigmoid(A);   /* [≈0.0, 0.5, ≈0.881] */
tensor_free(S);
```

---

## `tensor_softmax`

```c
tensor_t* tensor_softmax(const tensor_t* A);
```

Aplica softmax a lo largo de la **última dimensión**. Cada segmento a lo largo de esa dimensión (una "fila") se convierte en una distribución de probabilidad:

$$
\operatorname{softmax}(x)_i = \frac{e^{x_i - m}}{\sum_{j} e^{x_j - m}}, \qquad m = \max_j x_j
$$

### Parámetros

| Nombre | Tipo | Descripción |
|---|---|---|
| `A` | `const tensor_t*` | Tensor de entrada de forma `(..., K)`. Softmax se calcula de forma independiente sobre cada grupo de `K` elementos consecutivos. |

### Valor de retorno

Un tensor `float32` nuevo con la misma forma que `A`, donde cada fila tiene entradas no negativas que suman 1 (salvo redondeo). Devuelve `NULL` si `A` es `NULL`, no es `float32`, la última dimensión no es positiva o falla la reserva de memoria.

### Comportamiento

Para cada una de las `size / K` filas:

1. Encuentra el máximo $m$ de la fila.
2. Reemplaza cada elemento por $e^{x_i - m}$ y acumula la suma.
3. Divide cada elemento entre la suma.

Restar $m$ no cambia el resultado, porque se cancela entre numerador y denominador, pero mantiene cada exponente $\le 0$ y así evita desbordamientos.

| Forma de entrada | Se normaliza sobre | Filas |
|---|---|---|
| `(K)` | los `K` elementos | 1 |
| `(m, K)` | cada fila | `m` |
| `(b, m, K)` | cada fila más interna | `b·m` |

### Notas

- **Solo se soporta la última dimensión.** No hay parámetro `axis`.
- **Se asume que las filas son contiguas** (`row = data + o*K`), así que una entrada no contigua da resultados incorrectos.
- **Valores no finitos.** Si una fila contiene `+inf`, entonces $\infty - \infty = $ `NaN` y toda la fila se vuelve `NaN`. Lo mismo ocurre si todos los elementos de una fila son `-inf`. Un `NaN` en cualquier posición de una fila también produce `NaN`s.
- La guarda `sum > 0` antes de normalizar nunca se activa con entradas finitas, porque el elemento máximo siempre aporta $e^0 = 1$ a la suma.
- La acumulación se hace en `float`.

### Ejemplo

```c
/* A = [[1.0, 2.0, 3.0],
        [1.0, 1.0, 1.0]]   forma (2, 3) */
tensor_t* P = tensor_softmax(A);
/* P ≈ [[0.090, 0.245, 0.665],
        [0.333, 0.333, 0.333]] */
tensor_free(P);
```

---

## Funciones auxiliares internas

| Función | Propósito |
|---|---|
| `sigmoid_stable` | Sigmoide escalar `static` que usa la forma estable de dos ramas descrita arriba. |
