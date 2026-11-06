
---

# **CI‑PHILOSOPHY — PHOBOS‑ML**

## **Visión general**
PHOBOS‑ML sigue la filosofía de NA‑Engine para control de calidad, integración continua y despliegue.  
El flujo de trabajo se basa en **tres capas**:

- **dev** → desarrollo activo  
- **qa** → validación de calidad  
- **main** → versión estable  

Cada capa tiene un propósito distinto y el CI se adapta a ese propósito.

---

# **1. Branches y ambientes**

## **dev — Development Layer**
La capa donde se desarrolla todo el código nuevo.

### Objetivo
Detectar errores **lo antes posible**.

### Qué corre el CI aquí
- Lint (clang-format)  
- Build completo  
- Todos los tests  
- Valgrind (detección de fugas de memoria)  
- Sin cobertura  
- Sin empaquetado  

### Filosofía
> “En dev debe fallar todo lo que pueda fallar.”

Es la capa más estricta en términos de pruebas dinámicas.

---

## **qa — Quality Assurance Layer**
La capa donde se valida la calidad del código antes de integrarlo a main.

### Objetivo
Garantizar que el código es **correcto, estable y cubierto**.

### Qué corre el CI aquí
- Lint  
- Build completo  
- Todos los tests  
- Valgrind  
- Cobertura (lcov)  
- Sin empaquetado  

### Filosofía
> “En qa debe demostrarse que el código es confiable.”

Aquí se mide la calidad, no la compatibilidad multiplataforma.

---

## **main — Stable Layer**
La capa estable del proyecto.  
Todo lo que llega aquí debe estar probado y validado.

### Objetivo
Generar artefactos estables y reproducibles.

### Qué corre el CI aquí
- Lint  
- Build completo  
- Tests esenciales  
- Sin valgrind  
- Sin cobertura  
- Empaquetado (si aplica)  

### Filosofía
> “main nunca debe romperse.”

Es la capa más ligera del CI, enfocada en estabilidad y release.

---

# **2. ¿Por qué no hay multiplataforma en dev/qa?**

### Razón 1 — Velocidad  
dev y qa deben ser rápidas.  
Multiplataforma hace el CI lento y costoso.

### Razón 2 — Herramientas  
Valgrind no corre en Windows.  
Cobertura no es igual en macOS.  
Linux es la plataforma ideal para análisis profundo.

### Razón 3 — Filosofía NA‑Engine  
La compatibilidad multiplataforma se valida en:

- **main**
- o en una capa adicional llamada **release‑candidate**

Pero **no en QA**.

---

# **3. Flujo de integración NA‑Engine**

```
dev → qa → main
```

### dev
Se desarrolla, se prueba exhaustivamente, se corrigen errores.

### qa
Se valida calidad, cobertura, estabilidad.

### main
Se libera, se empaqueta, se distribuye.

---

# **4. Automatización de tests**

PHOBOS‑ML adopta el patrón NA‑Engine:

- Construir **todos** los tests automáticamente  
- Ejecutar **todos** los tests automáticamente  
- Valgrind para **todos** los tests en dev/qa  
- Sin mantenimiento manual del CI cuando se agregan nuevos tests  

Esto garantiza que el CI siempre está actualizado sin esfuerzo.

---

# **5. Filosofía general del CI**

### ✔ El CI debe ser estricto en dev  
### ✔ El CI debe ser completo en qa  
### ✔ El CI debe ser ligero en main  
### ✔ El CI debe ser automático  
### ✔ El CI no debe requerir mantenimiento manual  
### ✔ El CI debe fallar rápido  
### ✔ El CI debe ser reproducible  

---

# **6. Beneficios del diseño**

- Menos mantenimiento  
- Menos duplicación  
- Más confiabilidad  
- Más velocidad  
- Más alineación con NA‑Engine  
- Más escalabilidad para futuros módulos (autograd, optimizers, conv2d, etc.)

---

# **7. Conclusión**

Este diseño de CI permite que PHOBOS‑ML crezca como un engine real:

- capas claras  
- responsabilidades claras  
- pruebas automáticas  
- calidad garantizada  
- estabilidad en main  

Es exactamente el tipo de arquitectura que usan engines como:

- NA‑Engine  
- PyTorch  
- TensorFlow  
- ONNX Runtime  

---
