# Entrenamiento

**Módulo:** PHOBOS ML
**Header:** `train.h`
**Implementación:** `train.c`

Esta página documenta las funciones de entrenamiento de PHOBOS. Encadenan el forward, la pérdida, los gradientes y la actualización de parámetros en iteraciones completas de entrenamiento para regresión logística.

| Función | Alcance | Devuelve |
|---|---|---|
| [`tensor_logreg_train_step`](#tensor_logreg_train_step) | Una iteración de entrenamiento | Código de estado `int` |
| [`train_loop`](#train_loop) | Varias épocas, con progreso impreso en `stdout` | Código de estado `int` |

> Las funciones de entrenamiento pertenecen al **módulo de ML**. En la arquitectura objetivo viven en `ml/`, y TLALOC no depende de ellas.

---

## ⚠️ Problema conocido: las dos funciones entrenan modelos distintos

La regresión logística se define como $\hat{y} = \sigma(XW + b)$. Actualmente ninguna de las dos funciones calcula exactamente eso, y además difieren entre sí:

| | Predicción que realmente se calcula | ¿Se usa el sesgo en la predicción? |
|---|---|---|
| Modelo esperado | $\sigma(XW + b)$ | ✅ |
| `tensor_logreg_train_step` | $\sigma\big(\sigma(XW) + b\big)$ | ✅, pero después de una primera sigmoide |
| `train_loop` | $\sigma(XW)$ | ❌ (se actualiza pero nunca se usa) |

Mientras tanto, `tensor_logreg_gradients` calcula los gradientes del modelo esperado. Por lo tanto, en ambas funciones los gradientes no corresponden a las predicciones que se hacen. Los detalles están en las notas de cada función.

---

## Convenciones generales

- **Los parámetros se actualizan en su lugar.** `W` y `*b` se sobrescriben. No se devuelven parámetros nuevos.
- **Las entradas `X` y `y` son de solo lectura.**
- **Sin fugas de memoria.** Todos los tensores intermedios (`y_hat`, `dW`) se reservan y liberan internamente, también en las rutas de error.
- **Manejo de errores.** `0` si tiene éxito, un código negativo si hay error.
- **Lote.** Cada iteración usa el conjunto de datos **completo** (descenso de gradiente por lote completo), no mini-lotes aleatorios.

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

Realiza una sola iteración de entrenamiento: forward, pérdida, gradientes y actualización por descenso de gradiente.

### Parámetros

| Nombre | Tipo | Dirección | Descripción |
|---|---|---|---|
| `X` | `const tensor_t*` | entrada | Variables de entrada, forma `(m, n)`. |
| `y` | `const tensor_t*` | entrada | Valores objetivo en $\{0, 1\}$, forma `(m, 1)`. |
| `W` | `tensor_t*` | entrada/salida | Pesos, forma `(n, 1)`. Se actualizan en su lugar. |
| `b` | `float*` | entrada/salida | Sesgo escalar. Se actualiza en su lugar. |
| `lr` | `float` | entrada | Tasa de aprendizaje $\eta$. |
| `loss_out` | `float*` | salida | Recibe la pérdida calculada **antes** de la actualización. |

### Pasos

1. **Forward:** `tensor_logreg_forward(X, W, NULL)` → $\sigma(XW)$.
2. **Sesgo:** suma `*b` a cada elemento del resultado y aplica `tensor_sigmoid` otra vez → $\sigma(\sigma(XW) + b)$.
3. **Pérdida:** `tensor_bce_loss(y, y_hat)`, que se escribe en `*loss_out`.
4. **Gradientes:** `tensor_logreg_gradients(X, y, y_hat, &dW, &db)`.
5. **Actualización:** $W \leftarrow W - \eta\,dW$, $\; b \leftarrow b - \eta\,db$, aplicada directamente en la función.

### Valor de retorno

| Código | Significado | ¿Se modifican `W` y `*b`? | ¿Se escribe `*loss_out`? |
|---|---|---|---|
| `0` | Éxito. | Sí | Sí |
| `-1` | Algún apuntador es `NULL`. | No | No |
| `-2` | Falló el forward (formas, dtype o reserva de memoria inválidos). | No | No |
| `-3` | Falló la segunda sigmoide (reserva de memoria). | No | No |
| `-4` | Falló el cálculo de gradientes. | No | **Sí** |

### Notas

- **Doble sigmoide (modelo incorrecto).** El forward ya aplica una sigmoide, así que el sesgo se suma a una probabilidad y se comprime por segunda vez. Con $b = 0$, todas las predicciones quedan en $(0.5,\ 0.731)$, porque $\sigma(0) = 0.5$ y $\sigma(1) \approx 0.731$. El modelo no puede predecir la clase 0 con confianza, y los gradientes de `tensor_logreg_gradients` no corresponden a esta función.
- **Actualización duplicada.** La actualización repite la lógica de `optimizer_sgd_step` en lugar de llamarla.
- **Problemas heredados.** El gradiente de los pesos de `tensor_logreg_gradients` es $m$ veces mayor que el gradiente real (ver la documentación de gradientes).
- La pérdida no se valida. Si `tensor_bce_loss` falla, `*loss_out` vale `0.0f` y el entrenamiento continúa.

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

Ejecuta `epochs` iteraciones de descenso de gradiente por lote completo e imprime la pérdida y el sesgo al final de cada época.

### Parámetros

| Nombre | Tipo | Dirección | Descripción |
|---|---|---|---|
| `X` | `const tensor_t*` | entrada | Variables de entrada, forma `(m, n)`. |
| `y` | `const tensor_t*` | entrada | Valores objetivo en $\{0, 1\}$, forma `(m, 1)`. |
| `W` | `tensor_t*` | entrada/salida | Pesos, forma `(n, 1)`. Se actualizan en su lugar. |
| `b` | `float*` | entrada/salida | Sesgo escalar. Se actualiza en su lugar. |
| `lr` | `float` | entrada | Tasa de aprendizaje $\eta$. |
| `epochs` | `int` | entrada | Número de iteraciones. Si es `<= 0`, no hace nada y devuelve `0`. |

### Cada época

1. `tensor_logreg_forward(X, W, NULL)` → $\hat{y} = \sigma(XW)$.
2. `tensor_bce_loss(y, y_hat)`.
3. `tensor_logreg_gradients(X, y, y_hat, &dW, &db)`.
4. `optimizer_sgd_step(W, b, dW, db, lr)`.
5. Imprime una línea en `stdout`:

```
[Epoch 1/100] Loss = 0.693147 Bias=-0.001234
```

### Valor de retorno

| Código | Significado |
|---|---|
| `0` | Se completaron todas las épocas. |
| `-1` | Algún apuntador es `NULL`. |
| `-2` | Falló el forward. |
| `-3` | Falló el cálculo de gradientes. |
| `-4` | Falló el paso del optimizador. |

Ante un error, el entrenamiento se detiene de inmediato. Las actualizaciones de las épocas ya completadas se conservan en `W` y `*b`.

### Notas

- **El sesgo se entrena pero nunca se usa.** El forward se llama con `NULL` como sesgo, así que $\hat{y} = \sigma(XW)$. En cada época se calcula `db`, se actualiza `*b` y se imprime su valor, pero nunca afecta una predicción. En la práctica el modelo es una regresión logística sin intercepto.
- **La pérdida impresa es la previa a la actualización** de cada época.
- **La salida siempre se imprime.** No hay forma de silenciarla, y `printf` se usa sin `#include <stdio.h>` en `train.c`.
- **No se devuelve el historial de pérdidas.** Solo la salida impresa registra la evolución de la pérdida. No hay detención temprana.
- **Problemas heredados.** El gradiente de los pesos es $m$ veces mayor que el gradiente real (ver la documentación de gradientes).
- Aunque usa `optimizer_sgd_step`, el algoritmo es descenso de gradiente por lote completo: cada época usa las `m` muestras.

### Ejemplo

```c
float b = 0.0f;
int rc = train_loop(X, y, W, &b, 0.1f, 100);
if (rc != 0) {
    fprintf(stderr, "Falló el entrenamiento (código %d)\n", rc);
}
```
