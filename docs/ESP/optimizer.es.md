# Optimizadores

**Módulo:** PHOBOS ML
**Header:** `optimizer.h`
**Implementación:** `optimizer.c`

Esta página documenta los optimizadores de PHOBOS. Un optimizador recibe los parámetros actuales de un modelo y sus gradientes, y actualiza los parámetros para reducir la pérdida.

| Función | Algoritmo | Devuelve |
|---|---|---|
| [`optimizer_sgd_step`](#optimizer_sgd_step) | Descenso de gradiente estocástico (un paso) | Código de estado `int` |

> Los optimizadores pertenecen al **módulo de ML**. En la arquitectura objetivo viven en `ml/`, y TLALOC no depende de ellos.

---

## Convenciones generales

- **Los parámetros se actualizan en su lugar.** A diferencia de las operaciones con tensores, los optimizadores **no** devuelven un tensor nuevo: sobrescriben los parámetros que reciben.
- **Los gradientes son de solo lectura.** Se reciben como `const`.
- **Sin reserva de memoria.** No se crea nada, así que no hay nada que liberar.
- **Manejo de errores.** Los optimizadores devuelven un código de estado `int`: `0` si tienen éxito y un valor negativo si hay error. Ante un error, ningún parámetro se modifica.

---

## `optimizer_sgd_step`

```c
int optimizer_sgd_step(tensor_t* W,
                       float* b,
                       const tensor_t* dW,
                       float db,
                       float lr);
```

Realiza **un paso** de descenso de gradiente estocástico simple sobre un tensor de pesos y un sesgo escalar.

### Regla de actualización

$$
W \leftarrow W - \eta\,\nabla_W \mathcal{L}, \qquad
b \leftarrow b - \eta\,\frac{\partial \mathcal{L}}{\partial b}
$$

donde $\eta$ es la tasa de aprendizaje. Elemento a elemento, $W_i \leftarrow W_i - \eta\, dW_i$ para cada elemento $i$.

### Parámetros

| Nombre | Tipo | Dirección | Descripción |
|---|---|---|---|
| `W` | `tensor_t*` | entrada/salida | Tensor de pesos. Se sobrescribe con los pesos actualizados. |
| `b` | `float*` | entrada/salida | Apuntador al sesgo escalar. Se sobrescribe con el sesgo actualizado. |
| `dW` | `const tensor_t*` | entrada | Gradiente de la pérdida respecto a `W`. Mismo número de elementos que `W`. |
| `db` | `float` | entrada | Gradiente de la pérdida respecto a `b`. |
| `lr` | `float` | entrada | Tasa de aprendizaje $\eta$. Normalmente un valor positivo pequeño como `0.01f`. |

### Valor de retorno

| Código | Significado |
|---|---|
| `0` | Éxito. Se actualizaron `W` y `*b`. |
| `-1` | `W`, `b` o `dW` es `NULL`. No se modificó nada. |
| `-2` | `W->size != dW->size`. No se modificó nada. |

### Comportamiento

- Cada llamada aplica exactamente una actualización. El ciclo de entrenamiento es responsable de calcular `dW` y `db` (p. ej. por backpropagation o con el mini-autograd) y de llamar a esta función una vez por lote.
- Si `W` es una vista no propietaria creada con `tensor_from_data`, la actualización escribe directamente en el buffer de quien llama.

### Notas

- **No se verifica el dtype.** Ambos tensores se leen como `float*`. Pasar un tensor `float64` o entero da resultados incorrectos sin aviso.
- **Solo se verifica `size`, no `shape`.** Se acepta un `dW` de forma `(1, n)` para un `W` de forma `(n, 1)`, y se aplica en el orden plano de memoria. Asegúrate de que ambos tengan la misma forma.
- **Se asumen datos contiguos.** Los elementos se procesan como `data[0 .. size-1]`, ignorando `strides`.
- **`lr` no se valida.** Una tasa de aprendizaje negativa realiza *ascenso* de gradiente. Una tasa `NaN` convierte todos los parámetros en `NaN`.
- **El sesgo es un escalar.** Los modelos con un sesgo por salida, como un sesgo de forma `(m, 1)` pasado a `tensor_logreg_forward`, no se pueden actualizar con esta función.
- Los códigos de error son enteros simples. Todavía no tienen constantes con nombre.

### Ejemplo

```c
float lr = 0.1f;
float b  = 0.0f;

for (int epoch = 0; epoch < n_epochs; ++epoch) {
    /* aquí se calculan forward, pérdida y gradientes (dW, db) */

    int rc = optimizer_sgd_step(W, &b, dW, db, lr);
    if (rc != 0) {
        fprintf(stderr, "Falló el paso de SGD (código %d)\n", rc);
        break;
    }
}
```
