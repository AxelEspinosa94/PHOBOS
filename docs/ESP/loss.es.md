# Funciones de pérdida

**Módulo:** PHOBOS ML
**Header:** `loss.h`
**Implementación:** `loss.c`

Esta página documenta las funciones de pérdida de PHOBOS. Reducen un conjunto de predicciones y sus valores objetivo a un solo escalar que mide qué tan equivocado está el modelo.

| Función | Tarea | Devuelve |
|---|---|---|
| [`tensor_bce_loss`](#tensor_bce_loss) | Clasificación binaria | `float` |

> Las funciones de pérdida pertenecen al **módulo de ML**. En la arquitectura objetivo viven en `ml/`, y TLALOC no depende de ellas.

---

## Convenciones generales

- **Dtype soportado: solo `DTYPE_FLOAT32`.**
- **Las entradas nunca se modifican.** Ambas se reciben como `const tensor_t*`.
- **Sin reserva de memoria.** Las pérdidas devuelven un `float` simple, así que no hay nada que liberar.
- **Manejo de errores.** Ante entradas inválidas la función devuelve `0.0f`. En una pérdida esto es especialmente engañoso, porque `0.0f` también es el valor de una predicción **perfecta**. Ver [Notas](#notas).

---

## `tensor_bce_loss`

```c
float tensor_bce_loss(const tensor_t* y, const tensor_t* p);
```

Calcula la **entropía cruzada binaria promedio** entre los valores objetivo `y` y las probabilidades predichas `p`.

### Definición

$$
\mathcal{L}(y, p) = -\frac{1}{N}\sum_{i=0}^{N-1}\Big[\, y_i \log \tilde{p}_i + (1 - y_i)\log(1 - \tilde{p}_i) \,\Big]
$$

donde $N$ es el número de elementos y $\tilde{p}_i$ es la predicción acotada para evitar $\log(0)$:

$$
\tilde{p}_i = \min\big(\max(p_i,\ \varepsilon),\ 1 - \varepsilon\big), \qquad \varepsilon = 10^{-7}
$$

### Parámetros

| Nombre | Tipo | Descripción |
|---|---|---|
| `y` | `const tensor_t*` | Valores objetivo. Normalmente `0.0` o `1.0`. También se aceptan valores en $[0, 1]$ (etiquetas suaves). |
| `p` | `const tensor_t*` | Probabilidades predichas, normalmente la salida de una sigmoide (p. ej. `tensor_logreg_forward`). Se esperan en $[0, 1]$. |

### Valor de retorno

La pérdida promedio como `float` no negativo. Entre menor, mejor.

Devuelve `0.0f` si:

- `y` o `p` es `NULL`;
- alguno de los tensores no es `float32`;
- `y->size != p->size`.

### Comportamiento

- **Reducción: promedio.** La suma se divide entre $N$, así que la pérdida no crece con el tamaño del lote.
- **Acotamiento (clamping).** Cada predicción se acota a $[\varepsilon, 1 - \varepsilon]$ antes de calcular logaritmos. Consecuencias:
  - La pérdida siempre es finita, incluso con predicciones exactamente iguales a `0.0` o `1.0`.
  - La contribución máxima de un solo elemento es aproximadamente $-\log(10^{-7}) \approx 16.12$.
  - Dentro de la región acotada la pérdida deja de cambiar con $p$, así que su gradiente ahí es cero.
- **Interpretación.** Un modelo que siempre predice $p = 0.5$ tiene una pérdida de $\log 2 \approx 0.693$. Es una referencia útil: un modelo entrenado debería quedar por debajo.

### Notas

- **Solo se verifica `size`, no `shape`.** Los tensores se comparan elemento a elemento en el orden plano de memoria, así que se aceptan `y` de forma `(m, 1)` y `p` de forma `(1, m)`. Asegúrate de que ambos tengan la misma forma.
- **Se asumen datos contiguos.** Los elementos se leen como `data[0 .. size-1]`, ignorando `strides`.
- **Los valores no se validan.** Objetivos fuera de $[0, 1]$ no se rechazan y producen pérdidas sin sentido (posiblemente negativas). Predicciones fuera de $[0, 1]$ se acotan sin aviso. Un `NaN` en `p` se propaga al resultado.
- **Valor de error ambiguo.** Como `0.0f` significa tanto "error" como "predicción perfecta", una llamada inválida puede parecer un modelo perfectamente entrenado. Valida las entradas antes de llamar.
- La acumulación se hace en `float`.

### Ejemplo

```c
int shape[] = {4, 1};
float y_vals[] = {1.0f, 0.0f, 1.0f, 0.0f};
float p_vals[] = {0.9f, 0.2f, 0.7f, 0.1f};

tensor_t* y = tensor_from_data(y_vals, DTYPE_FLOAT32, 2, shape);
tensor_t* p = tensor_from_data(p_vals, DTYPE_FLOAT32, 2, shape);

float loss = tensor_bce_loss(y, p);   /* ≈ 0.198 */

tensor_free(y);
tensor_free(p);
```

### Uso típico

```c
tensor_t* p = tensor_logreg_forward(X, W, b);   /* predicciones (m, 1)  */
float loss  = tensor_bce_loss(y, p);            /* y: objetivos (m, 1)  */
tensor_free(p);
```
