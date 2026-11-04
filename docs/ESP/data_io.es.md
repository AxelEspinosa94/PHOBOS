# Lectura y escritura de datos (CSV)

**Módulo:** PHOBOS Core
**Header:** `data_io.h`
**Implementación:** `data_io.c`

Esta página documenta cómo PHOBOS lee archivos CSV numéricos en tensores y cómo escribe tensores de vuelta a CSV.

| Función | Propósito | Devuelve |
|---|---|---|
| [`tensor_load_csv`](#tensor_load_csv) | Lee un archivo CSV numérico en un tensor `float32` 2-D | Tensor nuevo o `NULL` |
| [`tensor_save_csv`](#tensor_save_csv) | Escribe un tensor `float32` 2-D en un archivo CSV | Código de estado `int` |

> La lectura y escritura de datos es de propósito general y no depende del módulo de ML. TLALOC puede usarla, por ejemplo, para exportar perfiles de velocidad y graficarlos.

---

## Dialecto de CSV soportado

Solo se soporta un subconjunto mínimo de CSV:

| Característica | ¿Soportada? |
|---|---|
| Coma (`,`) como separador | ✅ (el único separador) |
| Solo valores numéricos (enteros, decimales, notación científica, `nan`, `inf`) | ✅ |
| Una línea de encabezado opcional (se omite, no se interpreta) | ✅ |
| Fines de línea `\n` y `\r\n` | ✅ |
| Última línea sin salto de línea final | ✅ |
| Espacios **después** de una coma (`1, 2`) | ✅ |
| Espacios **antes** de una coma (`1 ,2`) | ❌ se rechaza el archivo |
| Campos entre comillas, columnas de texto, otros separadores (`;`, tabulador) | ❌ se rechaza el archivo |
| Campos vacíos (`1,,3`) | ⚠️ **se aceptan pero se cargan mal**; ver notas |
| Líneas en blanco, incluida una línea en blanco al final del archivo | ❌ se rechaza el archivo |
| Líneas de más de 4095 caracteres | ❌ se rechaza el archivo (en la práctica) |

---

## `tensor_load_csv`

```c
tensor_t* tensor_load_csv(const char* path, int skip_header);
```

Lee un archivo CSV numérico y devuelve su contenido como un tensor `float32` 2-D de forma `(filas, columnas)`.

### Parámetros

| Nombre | Tipo | Descripción |
|---|---|---|
| `path` | `const char*` | Ruta del archivo CSV. |
| `skip_header` | `int` | Si es distinto de cero, la primera línea se omite sin interpretarse. |

### Valor de retorno

Un tensor `float32` nuevo y contiguo de forma `(filas, columnas)`. Quien llama es su dueño y debe liberarlo con `tensor_free`.

Devuelve `NULL` si:

- `path` es `NULL` o el archivo no se puede abrir;
- `skip_header` está activo y el archivo está vacío;
- el archivo no tiene filas de datos;
- alguna fila tiene un número de comas distinto al de la primera fila de datos;
- algún campo no se puede interpretar completamente como número;
- falla la reserva de memoria.

### Cómo funciona

El archivo se lee **dos veces**:

1. **Primera pasada:** cuenta las filas y verifica que todas tengan el mismo número de columnas (comas + 1).
2. Se reserva el tensor con esa forma.
3. **Segunda pasada:** divide cada línea por comas con `strtok` y convierte cada campo con `strtof`. Un campo es válido solo si `strtof` lo consume por completo, permitiendo como máximo un `\n` o `\r` al final.

### Notas

- **Los campos vacíos corrompen el resultado en silencio.** La primera pasada cuenta comas, así que `1,,3` tiene 3 columnas. La segunda pasada usa `strtok`, que se salta los tokens vacíos y solo encuentra 2 campos. Los valores se escriben en las columnas equivocadas y la última columna de esa fila queda **sin inicializar**. Por ejemplo, cargar `1,,3` / `4,5,6` devuelve `[[1, 3, ?], [4, 5, 6]]`, donde `?` es lo que hubiera en memoria. No se reporta ningún error.
- **Los errores no dan información.** Toda falla devuelve `NULL`, así que quien llama no puede distinguir un archivo inexistente de una fila mal formada, ni saber qué línea falló.
- **Una línea en blanco rechaza todo el archivo.** Esto incluye una sola línea en blanco extra al final del archivo, algo que producen muchos editores y herramientas.
- **Límite de longitud de línea.** Las líneas se leen en un buffer de 4096 bytes. Una línea más larga se divide en varias lecturas, lo que en la práctica hace fallar la verificación de columnas y rechaza el archivo completo. Suponiendo unos 4 caracteres por valor, esto limita los archivos a unas 1000 columnas.
- **No sirve con pipes ni con `stdin`.** El diseño de dos pasadas depende de `rewind`, que requiere un archivo regular en el que se pueda hacer seek.
- **Usa `strtok`.** Mantiene estado global oculto, así que la función no es reentrante ni segura para hilos.
- **Depende del locale.** `strtof` sigue el locale `LC_NUMERIC` del programa. Si un programa llama a `setlocale` con un locale que usa coma decimal, la interpretación se rompe.
- Cualquier valor que acepte `strtof` es válido, incluidos `nan`, `inf` y flotantes hexadecimales.

### Ejemplo

```c
/* data.csv:
   x1,x2,label
   0.5,1.2,1
   0.1,0.3,0          */

tensor_t* D = tensor_load_csv("data.csv", 1);   /* forma (2, 3) */
if (!D) {
    fprintf(stderr, "No se pudo cargar data.csv\n");
}
/* ... */
tensor_free(D);
```

---

## `tensor_save_csv`

```c
int tensor_save_csv(const char* path, const tensor_t* tensor);
```

Escribe un tensor 2-D en un archivo CSV, una fila por línea, reemplazando el archivo si existe.

### Parámetros

| Nombre | Tipo | Descripción |
|---|---|---|
| `path` | `const char*` | Ruta del archivo de destino. |
| `tensor` | `const tensor_t*` | Tensor 2-D a escribir. Se espera `float32` y contiguo. |

### Valor de retorno

| Código | Significado |
|---|---|
| `0` | Se abrió el archivo y se emitieron todas las filas. Ver notas: esto **no** garantiza que el archivo esté completo. |
| `-1` | `path` o `tensor` es `NULL`. |
| `-2` | `tensor` no es bidimensional. |
| `-3` | No se pudo abrir el archivo para escritura. |

### Formato de salida

- Los valores se escriben con el `%f` de `printf`: punto fijo con **6 decimales**.
- Las columnas se separan con `,` sin espacios. Cada fila termina con `\n`.
- **No se escribe línea de encabezado.**

### Notas

- **Pérdida de precisión.** `%f` siempre imprime exactamente 6 decimales, así que los valores pequeños se pierden y los grandes reciben dígitos espurios. Guardar los valores `1e-7`, `0.1` y `123456.789` produce `0.000000,0.100000,123456.789062`. Un ciclo de guardar y cargar no reproduce el tensor original. Usar `%.9g` preservaría exactamente cualquier valor `float32`.
- **No se detectan errores de escritura.** Se ignoran los valores de retorno de `fprintf` y `fclose`. Si el disco se llena, la función devuelve `0` de todos modos con un archivo incompleto.
- **No se verifica el dtype.** Los datos se leen como `float*` sin importar `tensor->dtype`.
- **Se asumen datos contiguos.** Los elementos se leen como `data[r*cols + c]`, ignorando `strides`.
- **Asimetría con la carga.** Como no se escribe encabezado, cargar el archivo guardado con `skip_header = 1` descarta en silencio la primera fila de datos.

### Ejemplo

```c
if (tensor_save_csv("predictions.csv", y_hat) != 0) {
    fprintf(stderr, "No se pudieron guardar las predicciones\n");
}
```
