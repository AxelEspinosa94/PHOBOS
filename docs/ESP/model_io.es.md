# Persistencia del modelo

**Módulo:** PHOBOS ML
**Header:** `model_io.h`
**Implementación:** `model_io.c`

Esta página documenta cómo PHOBOS guarda en disco un modelo de regresión logística entrenado y cómo lo vuelve a cargar.

| Función | Propósito | Devuelve |
|---|---|---|
| [`model_save`](#model_save) | Escribe pesos y sesgo en un archivo binario | Código de estado `int` |
| [`model_load`](#model_load) | Lee pesos y sesgo en un tensor existente | Código de estado `int` |

> La persistencia del modelo pertenece al **módulo de ML**. En la arquitectura objetivo vive en `ml/`, y TLALOC no depende de ella.

---

## Formato del archivo

Los modelos se guardan en un archivo binario crudo con la siguiente disposición, escrita en orden y sin relleno:

| Campo | Tipo | Tamaño (Linux x86-64) | Descripción |
|---|---|---|---|
| Magic | `char[8]` | 8 B | Los bytes ASCII `PHOBOSML`, sin `\0` final. |
| `ndim` | `int` | 4 B | Número de dimensiones de `W`. |
| `shape` | `int[ndim]` | 4·`ndim` B | Tamaño de cada dimensión de `W`. |
| `size` | `size_t` | 8 B | Número total de elementos de `W`. |
| `data` | `float[size]` | 4·`size` B | Pesos, en el orden de memoria (row-major). |
| `b` | `float` | 4 B | Sesgo escalar. |

**Tamaño total:** $24 + 4\cdot\text{ndim} + 4\cdot\text{size}$ bytes. Por ejemplo, un vector de pesos de forma `(10, 1)` produce un archivo de 72 bytes.

### Propiedades del formato

- **Representación nativa.** Enteros y flotantes se escriben tal como están en memoria. El archivo depende del endianness de la máquina y de `sizeof(int)` y `sizeof(size_t)`. Solo se garantiza que cargue en el mismo tipo de plataforma que lo escribió.
- **Sin campo de versión.** Cualquier cambio en la disposición hace ilegibles los archivos anteriores, sin forma de detectar qué formato usa un archivo.
- **Sin campo de dtype.** Los datos siempre se interpretan como `float32`.
- `size` es redundante con `shape` (es su producto). Se guarda como verificación adicional de consistencia.

---

## Convenciones generales

- **Códigos de estado.** Ambas funciones devuelven `0` si tienen éxito y un valor negativo si hay error.
- **Sin reserva de memoria.** `model_load` no crea un tensor. Llena uno que quien llama ya creó con la forma esperada.
- **Solo se guardan `W` y un `b` escalar.** No se guarda estado de entrenamiento (tasa de aprendizaje, época, pérdida).

---

## `model_save`

```c
int model_save(const char* path, const tensor_t* W, float b);
```

Escribe los pesos `W` y el sesgo `b` en el archivo `path`, reemplazándolo si existe.

### Parámetros

| Nombre | Tipo | Descripción |
|---|---|---|
| `path` | `const char*` | Ruta del archivo de destino. |
| `W` | `const tensor_t*` | Tensor de pesos a guardar. Se espera `float32` y contiguo. |
| `b` | `float` | Sesgo a guardar. |

### Valor de retorno

| Código | Significado |
|---|---|
| `0` | `fopen` tuvo éxito y se emitieron todas las escrituras. Ver notas: esto **no** garantiza que el archivo esté completo. |
| `-1` | `path` o `W` es `NULL`. |
| `-2` | No se pudo abrir el archivo para escritura (directorio inexistente, permisos, etc.). |

### Notas

- **No se detectan errores de escritura.** Se ignoran los valores de retorno de `fwrite` y `fclose`. Si el disco se llena o hay un error de E/S, la función devuelve `0` de todos modos y el archivo queda incompleto.
- **No es atómica.** El archivo existente se trunca en cuanto se abre. Si el programa falla durante el guardado, se pierden tanto el modelo anterior como el nuevo.
- **No se verifica el dtype.** Se escriben `size · sizeof(float)` bytes sin importar `W->dtype`. Con un tensor `float64` solo se escribe la primera mitad de los datos.
- **Se asumen datos contiguos.** `W->data` se escribe como un solo bloque, ignorando `strides`.

### Ejemplo

```c
if (model_save("spam_model.bin", W, b) != 0) {
    fprintf(stderr, "No se pudo guardar el modelo\n");
}
```

---

## `model_load`

```c
int model_load(const char* path, tensor_t* W, float* b);
```

Lee un modelo desde `path` en un tensor `W` **existente** y en el flotante al que apunta `b`. El archivo debe describir exactamente la misma forma que `W`.

### Parámetros

| Nombre | Tipo | Dirección | Descripción |
|---|---|---|---|
| `path` | `const char*` | entrada | Ruta del archivo a leer. |
| `W` | `tensor_t*` | entrada/salida | Tensor `float32` ya reservado con la forma esperada. Sus datos se sobrescriben. |
| `b` | `float*` | salida | Recibe el sesgo guardado. |

### Orden de validación

El encabezado se verifica campo por campo antes de escribir cualquier dato en `W`:

1. Los bytes mágicos son `PHOBOSML`.
2. `ndim` es igual a `W->ndim`.
3. Cada entrada de `shape` es igual a `W->shape[i]`.
4. `size` es igual a `W->size`.

Solo si pasan las cuatro verificaciones se leen los pesos y el sesgo.

### Valor de retorno

| Código | Significado | ¿Se modifican `W` / `*b`? |
|---|---|---|
| `0` | El encabezado coincidió y se emitieron todas las lecturas. Ver notas: esto **no** garantiza que los datos estén completos. | Sí |
| `-1` | `path`, `W` o `b` es `NULL`. | No |
| `-2` | No se pudo abrir el archivo. | No |
| `-3` | Bytes mágicos incorrectos: no es un archivo de modelo de PHOBOS. | No |
| `-4` | `ndim` del archivo difiere de `W->ndim`. | No |
| `-5` | `shape` del archivo difiere de `W->shape`. | No |
| `-6` | `size` del archivo difiere de `W->size`. | No |

### Notas

- **No se detectan archivos truncados.** Se ignoran los valores de retorno de `fread`. Un archivo cortado a la mitad de los pesos devuelve `0` de todos modos: `W` queda parcialmente sobrescrito con los valores nuevos y parcialmente con los anteriores, y `*b` no cambia. Por ejemplo, un modelo `(10, 1)` de 72 bytes truncado a 40 bytes se carga "con éxito" con solo 3 de sus 10 pesos actualizados.
- **La forma debe conocerse de antemano.** La función no puede cargar un modelo de forma desconocida, porque nunca reserva memoria. Quien llama debe crear `W` con la forma correcta primero.
- **No se verifica el dtype.** Los datos se leen como `float32` sin importar `W->dtype`.
- **Se ignoran datos sobrantes.** Los bytes extra después del sesgo no se reportan.
- **Arreglo de longitud variable.** La forma se lee en `int shape[ndim]` en el stack. `ndim` se acota con `W->ndim` antes de declarar el arreglo, así que es seguro, pero los VLA son opcionales desde C11 y MSVC no los soporta.
- La cadena mágica `PHOBOSML` es anterior al cambio de nombre del proyecto a PHOBOS. Cambiarla haría ilegibles todos los archivos de modelo existentes.

### Ejemplo

```c
int shape[] = {n_features, 1};
tensor_t* W = tensor_create(DTYPE_FLOAT32, 2, shape);
float b = 0.0f;

int rc = model_load("spam_model.bin", W, &b);
if (rc != 0) {
    fprintf(stderr, "No se pudo cargar el modelo (código %d)\n", rc);
}
```
