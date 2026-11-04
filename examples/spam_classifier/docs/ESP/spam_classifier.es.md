# Caso de uso: clasificador de spam

**Ubicación:** `examples/spam_classifier/`
**Módulos usados:** PHOBOS Core (tensores, lectura y escritura de datos) y PHOBOS ML (regresión logística, entrenamiento, persistencia del modelo)

Este ejemplo entrena un modelo de regresión logística que decide si un mensaje SMS es **spam** o **ham** (legítimo). Recorre el pipeline completo de PHOBOS de principio a fin: carga de datos desde CSV, entrenamiento, guardado del modelo, carga y predicción.

---

## Visión general del pipeline

El flujo tiene cuatro etapas. El preprocesamiento se hace en Python; todo lo demás corre sobre el motor en C de PHOBOS.

```
                         dataset.csv
                              │
                              ▼
                  ┌───────────────────────┐
                  │   preprocess.py       │  Python
                  └───────────┬───────────┘
              ┌───────────────┼────────────────┐
              ▼               ▼                ▼
       vocabulary.txt       X.csv            y.csv
              │               │                │
              │               └───────┬────────┘
              │                       ▼
              │           ┌───────────────────────┐
              │           │   train_spam          │  C
              │           └───────────┬───────────┘
              │                       ▼
              │                spam_model.bin
              │                       │
              │        ┌──────────────┴──────────────┐
              ▼        ▼                             ▼
   ┌───────────────────────┐            ┌───────────────────────┐
   │   predict_spam        │  C         │   accuracy            │  C
   │   un mensaje          │            │   dataset completo    │
   └───────────────────────┘            └───────────────────────┘
```

| Archivo | Tipo | Lo produce | Lo usa |
|---|---|---|---|
| `dataset.csv` | Datos de entrada | — | `preprocess.py` |
| `vocabulary.txt` | Texto, una palabra por línea | `preprocess.py` | `predict_spam` |
| `X.csv` | Matriz de características (0/1), con encabezado | `preprocess.py` | `train_spam`, `accuracy` |
| `y.csv` | Etiquetas (0/1), con encabezado | `preprocess.py` | `train_spam`, `accuracy` |
| `spam_model.bin` | Modelo entrenado (formato binario de PHOBOS) | `train_spam` | `predict_spam`, `accuracy` |

---

## Dataset

`dataset.csv` contiene 5,573 mensajes SMS en dos columnas:

| Columna | Contenido |
|---|---|
| `label` | `ham` o `spam` |
| `text` | Texto crudo del mensaje |

| Clase | Mensajes | Proporción |
|---|---|---|
| ham | 4,826 | 86.6 % |
| spam | 747 | 13.4 % |

Las clases están **desbalanceadas**. Un modelo que siempre responde "ham" ya alcanza 86.6 % de exactitud, así que esa cifra es la referencia que cualquier modelo real tiene que superar.

---

## Etapa 1 — Preprocesamiento (`preprocess.py`)

Convierte el texto crudo en una representación numérica que el motor en C puede cargar.

### Pasos

1. **Etiquetas.** `label` se pasa a minúsculas, se recorta y se mapea a `ham → 0`, `spam → 1`.
2. **Limpieza del texto.** Cada mensaje se pasa a minúsculas, todo carácter fuera de `a–z`, `0–9` y espacios se reemplaza por un espacio, y los espacios consecutivos se reducen a uno.
3. **Tokenización.** El texto limpio se divide por espacios.
4. **Vocabulario.** Se eligen los 200 tokens más frecuentes de todo el dataset (`VOCAB_SIZE = 200`).
5. **Vectorización.** Cada mensaje se convierte en un vector binario de longitud 200: la posición $j$ vale 1 si la $j$-ésima palabra del vocabulario aparece **como token** en el mensaje, y 0 en otro caso (bolsa de palabras, solo presencia).

### Salidas

| Archivo | Forma | Notas |
|---|---|---|
| `vocabulary.txt` | 200 líneas | Ordenado de la palabra más frecuente a la menos frecuente. El orden define el índice de cada característica. |
| `X.csv` | `(5573, 200)` | Fila de encabezado con las palabras del vocabulario. |
| `y.csv` | `(5573, 1)` | Fila de encabezado `label`. |

Ambos CSV tienen encabezado, por eso los programas en C los cargan con `skip_header = 1`.

---

## Etapa 2 — Entrenamiento (`train_spam.c`)

Entrena el modelo con `X.csv` y `y.csv` y lo guarda en `spam_model.bin`.

### Pasos

1. Carga `X` y `y` con `tensor_load_csv(path, 1)`.
2. Crea `W` de forma `(200, 1)` y lo inicializa en cero. Fija `b = 0`.
3. Llama a `train_loop(X, y, W, &b, LEARNING_RATE, EPOCHS)`, que imprime la pérdida y el sesgo después de cada época.
4. Guarda el resultado con `model_save("spam_model.bin", W, b)`.

### Hiperparámetros

| Constante | Valor |
|---|---|
| `LEARNING_RATE` | `0.03` |
| `EPOCHS` | `2500` |
| Inicialización de pesos | ceros |
| Lote | dataset completo en cada época |

---

## Etapa 3 — Predicción individual (`predict_spam.c`)

Clasifica un mensaje que escribe el usuario.

### Pasos

1. Lee hasta 200 palabras de `vocabulary.txt`. El número de palabras leídas define la forma de `W`.
2. Crea `W` con forma `(vocab_count, 1)` y lo carga con `model_load`. Si el modelo se entrenó con otro tamaño de vocabulario, la carga falla con un error de forma.
3. Lee una línea de `stdin` (hasta 4,095 caracteres).
4. Construye el vector de características: la posición $j$ vale 1 si la $j$-ésima palabra del vocabulario aparece en el mensaje, usando `strstr`.
5. Ejecuta `tensor_logreg_forward(X, W, NULL)` e imprime la probabilidad.
6. Imprime `SPAM` si la probabilidad es ≥ 0.5 y `HAM` en otro caso.

### Ejemplo de sesión

```
Enter message:
free entry win a prize now

Spam probability: 0.9561
Prediction: SPAM
```

> ⚠️ Aquí las características se construyen distinto que en el entrenamiento. Ver [Desajuste entre entrenamiento y predicción](#desajuste-entre-entrenamiento-y-predicción-en-predict_spam).

---

## Etapa 4 — Evaluación (`accuracy.c`)

Ejecuta el modelo sobre cada fila de `X.csv` y compara las predicciones con `y.csv`.

### Pasos

1. Carga `X`, `y` y el modelo.
2. Para cada fila, la copia en un tensor `(1, 200)`, ejecuta `tensor_logreg_forward(sample, W, NULL)` y aplica el umbral de 0.5.
3. Cuenta los cuatro resultados de la matriz de confusión y los imprime junto con la exactitud.

### Matriz de confusión

|  | Predicho spam | Predicho ham |
|---|---|---|
| **Real spam** | TP | FN |
| **Real ham** | FP | TN |

$$
\text{Exactitud} = \frac{TP + TN}{TP + TN + FP + FN}
$$

---

## Resultados

Las cifras siguientes provienen de una réplica en NumPy del pipeline que sigue exactamente la semántica del código en C (mismo preprocesamiento, inicialización en cero, misma tasa de aprendizaje y épocas, mismas fórmulas de gradiente). Reproduce la exactitud de ~96 % observada con el build en C. Los conteos exactos de los binarios en C pueden variar ligeramente por detalles de punto flotante.

| Métrica | Valor |
|---|---|
| Exactitud | **96.61 %** (referencia: 86.60 %) |
| TP / FN | 678 / 69 |
| FP / TN | 120 / 4,706 |
| Recall de spam $\frac{TP}{TP+FN}$ | 90.8 % |
| Precisión de spam $\frac{TP}{TP+FP}$ | 85.0 % |

### Cómo leer estas cifras

- **Es exactitud de entrenamiento.** `accuracy` evalúa el modelo sobre las mismas filas con las que se entrenó. No hay división entre entrenamiento y prueba, así que la cifra mide qué tan bien se ajusta el modelo a los datos que ya vio, no qué tan bien generaliza a mensajes nuevos.
- **Compárala con la referencia.** Como el 86.6 % de los mensajes son ham, la ganancia real es de 86.6 % a 96.6 %, y el recall y la precisión sobre la clase spam describen mejor al modelo que la exactitud sola.

---

## Cómo ejecutarlo

Todos los programas usan rutas relativas a la raíz del repositorio, así que deben ejecutarse **desde la raíz** del repositorio.

```bash
# 1. Preprocesar (requiere pandas y numpy)
python3 examples/spam_classifier/preprocess.py

# 2. Entrenar   — escribe examples/spam_classifier/spam_model.bin
./train_spam

# 3. Evaluar sobre el dataset
./accuracy

# 4. Clasificar un mensaje
./predict_spam
```

> Los nombres y ubicaciones de los ejecutables dependen de cómo el Makefile compile los ejemplos.

---

## Limitaciones conocidas

### Desajuste entre entrenamiento y predicción en `predict_spam`

Las características de entrenamiento vienen de `preprocess.py`, que pasa el texto a minúsculas, quita la puntuación y busca **tokens completos**. `predict_spam`, en cambio, busca en el mensaje crudo con `strstr`, que **distingue mayúsculas** y encuentra **subcadenas**. Por eso las palabras cortas del vocabulario, como `a`, `i`, `u`, `s`, `t` o `m`, aparecen dentro de casi cualquier palabra, mientras que palabras en mayúsculas como `FREE` no coinciden con `free`.

Pasar el dataset por la extracción de características de `predict_spam` en lugar de la de `preprocess.py` muestra el impacto:

| Extracción de características | Exactitud | Recall de spam |
|---|---|---|
| Tokens (`preprocess.py`, como en entrenamiento y `accuracy`) | 96.61 % | 90.8 % |
| Subcadenas (`strstr`, como en `predict_spam`) | 88.48 % | 29.0 % |

Con búsqueda por subcadenas, el modelo deja pasar unos 7 de cada 10 mensajes de spam. Para corregirlo, `predict_spam` debe aplicar la misma limpieza y tokenización que `preprocess.py`.

### El sesgo no se usa

El entrenamiento, la predicción y la evaluación llaman a `tensor_logreg_forward` con `NULL` como sesgo. El sesgo se entrena y se guarda (en la réplica converge a aproximadamente −2.0), pero nunca afecta una predicción. Es consistente entre los tres programas, pero cuando se corrija `train_loop` para usar el sesgo, `predict_spam` y `accuracy` también deberán pasarlo.

### Otras limitaciones

- **Las probabilidades se saturan.** Muchas predicciones se imprimen como `1.0000` o `0.0000`. En parte se debe a que el gradiente de los pesos es mayor de lo debido (ver la documentación de gradientes).
- **El tamaño del vocabulario aparece en dos lugares.** `VOCAB_SIZE = 200` está definido tanto en `preprocess.py` como en `predict_spam.c`. Si el valor de Python crece por encima de 200, `predict_spam` lee solo las primeras 200 palabras y `model_load` rechaza el modelo.
- **Etiquetas desconocidas.** Una etiqueta distinta de `ham` o `spam` se convierte en `NaN` en `y.csv`, y el lector de CSV la rechaza.
- **Evaluación ineficiente.** `accuracy` reserva un tensor nuevo para cada fila en lugar de hacer un solo forward sobre toda `X`.
- **Entrada sin verificar.** `predict_spam` no revisa el resultado de `fgets`. Si `stdin` está cerrado, el buffer del mensaje se usa sin inicializar.
