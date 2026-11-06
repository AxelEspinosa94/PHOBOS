# Gradientes

**Módulo:** PHOBOS ML
**Implementación:** `gradients.c`

Esta página documenta las funciones de gradiente de PHOBOS. Una función de gradiente calcula las derivadas de una pérdida respecto a los parámetros del modelo, que luego usa un optimizador como `optimizer_sgd_step` para actualizarlos.

| Función | Modelo / pérdida | Devuelve |
|---|---|---|
| [`tensor_logreg_gradients`](#tensor_logreg_gradients) | Regresión logística + entropía cruzada binaria | Código de estado `int` |

> Las funciones de gradiente pertenecen al **módulo de ML**. En la arquitectura objetivo viven en `ml/`, y TLALOC no depende de ellas.

---

## Lugar en el entrenamiento

Un paso de entrenamiento de regresión logística encadena cuatro funciones:

```
X, W, b ──► tensor_logreg_forward ──► y_hat
y, y_hat ─► tensor_bce_loss ────────► pérdida         (monitoreo)
X, y, y_hat ► tensor_logreg_gradients ► dW, db
W, b, dW, db ► optimizer_sgd_step ───► W, b actualizados en su lugar
```

---

## `tensor_logreg_gradients`

```c
int tensor_logreg_gradients(const tensor_t* X,
                            const tensor_t* y,
                            const tensor_t* y_hat,
                            tensor_t** dW_out,
                            float* db_out);
```

Calcula los gradientes de la pérdida de entropía cruzada binaria respecto a los pesos y el sesgo de un modelo de regresión logística.

### Derivación

En la regresión logística, $\hat{y} = \sigma(z)$ con $z = XW + b$, y la entropía cruzada binaria promedio es

$$
\mathcal{L} = -\frac{1}{m}\sum_{i=1}^{m}\Big[y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i)\Big]
$$

Como $\sigma'(z) = \sigma(z)\,(1 - \sigma(z))$, la derivada de cada término respecto a $z_i$ se simplifica al error de predicción:

$$
\frac{\partial \mathcal{L}}{\partial z_i} = \frac{1}{m}\,(\hat{y}_i - y_i)
$$

Aplicando la regla de la cadena:

$$
\nabla_W \mathcal{L} = \frac{1}{m}\,X^\top(\hat{y} - y), \qquad
\frac{\partial \mathcal{L}}{\partial b} = \frac{1}{m}\sum_{i=1}^{m}(\hat{y}_i - y_i)
$$

Por eso la función solo necesita `X`, `y` y `y_hat`, y no `W` ni `b`: los parámetros solo intervienen a través de `y_hat`.

### Qué calcula la función

Con el vector de error $e = \hat{y} - y$:

| Salida | Se calcula como | ¿Coincide con el gradiente de la BCE promedio? |
|---|---|---|
| `dW` | $X^\top e$ (**suma** sobre las muestras) | ❌ Mayor por un factor $m$ |
| `db` | $\frac{1}{m}\sum_i e_i$ (**promedio** sobre las muestras) | ✅ |

Ver [Notas](#notas) para las consecuencias de esta inconsistencia.

### Parámetros

| Nombre | Tipo | Dirección | Descripción |
|---|---|---|---|
| `X` | `const tensor_t*` | entrada | Variables de entrada, forma `(m, n)`. |
| `y` | `const tensor_t*` | entrada | Valores objetivo, forma `(m, 1)`. |
| `y_hat` | `const tensor_t*` | entrada | Probabilidades predichas por `tensor_logreg_forward`, forma `(m, 1)`. |
| `dW_out` | `tensor_t**` | salida | Si tiene éxito, recibe un tensor **nuevo** de forma `(n, 1)` con el gradiente de los pesos. |
| `db_out` | `float*` | salida | Si tiene éxito, recibe el gradiente del sesgo. |

### Valor de retorno

| Código | Significado |
|---|---|
| `0` | Éxito. Se escribieron `*dW_out` y `*db_out`. |
| `-1` | Alguno de los apuntadores es `NULL`. |
| `-2` | `X`, `y` o `y_hat` no es bidimensional. |
| `-3` | `y` o `y_hat` no tiene `m` filas (`X->shape[0]`). |
| `-4` | `y` o `y_hat` no es un vector columna (segunda dimensión `!= 1`). |
| `-5` | Falló la reserva de memoria del vector de error interno. |
| `-6` | Falló la reserva de memoria de `dW`. |

Ante cualquier error, `*dW_out` y `*db_out` no se modifican y no hay fugas de memoria.

### Propiedad de la memoria

El tensor escrito en `*dW_out` es nuevo. Quien llama es su dueño y debe liberarlo con `tensor_free`, normalmente justo después del paso del optimizador.

### Notas

- **Escalas inconsistentes entre `dW` y `db`.** `dW` es la suma sobre las muestras, mientras que `db` es el promedio. Junto con `tensor_bce_loss`, que es un promedio, el gradiente de los pesos resulta $m$ veces mayor que el gradiente real. En la práctica:
  - La tasa de aprendizaje efectiva para `W` es `lr · m`, mientras que para `b` es `lr`.
  - Una tasa de aprendizaje ajustada para un tamaño de lote se comporta distinto con otro.
  - La corrección es dividir también `dW` entre `m`.
- **Solo es válida para sigmoide + BCE.** El atajo $\partial\mathcal{L}/\partial z = \hat{y} - y$ solo se cumple para esa combinación. No es un gradiente de propósito general.
- **No se verifica el dtype.** Todos los tensores se leen como `float*`.
- **Se asumen datos contiguos.** `X` se indexa como `X[i*n + j]`, ignorando `strides`.
- **Se ignora el acotamiento de `tensor_bce_loss`.** El gradiente analítico no considera el acotamiento por $\varepsilon$. Esto solo importa para predicciones a menos de $10^{-7}$ de 0 o 1.
- En cada llamada se reserva y libera un tensor intermedio `(m, 1)` para guardar $e$.
- Los códigos de error son enteros simples y no coinciden con los significados de los códigos de `optimizer_sgd_step` (por ejemplo, `-2` significa cosas distintas en cada función).

### Ejemplo

```c
tensor_t* y_hat = tensor_logreg_forward(X, W, NULL);

tensor_t* dW = NULL;
float db = 0.0f;

int rc = tensor_logreg_gradients(X, y, y_hat, &dW, &db);
if (rc == 0) {
    optimizer_sgd_step(W, &b, dW, db, lr);
    tensor_free(dW);
}

tensor_free(y_hat);
```
