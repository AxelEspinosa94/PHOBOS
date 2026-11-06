
---

# **CI‑PHILOSOPHY — PHOBOS‑ML (English Version)**

## **Overview**
PHOBOS‑ML follows the NA‑Engine philosophy for quality control, continuous integration, and deployment.  
The workflow is built around **three layers**:

- **dev** → active development  
- **qa** → quality assurance  
- **main** → stable release  

Each layer has a distinct purpose, and the CI pipeline adapts to that purpose.

---

# **1. Branches and Environments**

## **dev — Development Layer**
This is where all new code is created and iterated.

### Purpose
Catch errors **as early as possible**.

### What CI runs here
- Linting (clang-format)  
- Full build  
- All tests  
- Valgrind (memory leak detection)  
- No coverage  
- No packaging  

### Philosophy
> “In dev, everything that can fail should fail.”

This is the strictest layer in terms of dynamic testing.

---

## **qa — Quality Assurance Layer**
This layer validates code quality before merging into main.

### Purpose
Ensure the code is **correct, stable, and covered**.

### What CI runs here
- Linting  
- Full build  
- All tests  
- Valgrind  
- Coverage (lcov)  
- No packaging  

### Philosophy
> “In qa, the code must prove it is reliable.”

This layer measures quality, not cross‑platform compatibility.

---

## **main — Stable Layer**
The stable branch of the project.  
Everything that reaches main must already be tested and validated.

### Purpose
Produce stable and reproducible artifacts.

### What CI runs here
- Linting  
- Full build  
- Essential tests  
- No valgrind  
- No coverage  
- Packaging (if applicable)  

### Philosophy
> “main must never break.”

This is the lightest CI layer, focused on stability and release.

---

# **2. Why dev/qa do NOT run multiplatform**

### Reason 1 — Speed  
dev and qa must be fast.  
Multiplatform testing makes CI slow and expensive.

### Reason 2 — Tools  
Valgrind does not run on Windows.  
Coverage tools differ across macOS.  
Linux is the ideal platform for deep analysis.

### Reason 3 — NA‑Engine Philosophy  
Cross‑platform validation belongs in:

- **main**
- or a dedicated **release‑candidate** layer

But **not in QA**.

---

# **3. NA‑Engine Integration Flow**

```
dev → qa → main
```

### dev  
Development, exhaustive testing, error correction.

### qa  
Quality validation, coverage, stability.

### main  
Release, packaging, distribution.

---

# **4. Automated Testing Philosophy**

PHOBOS‑ML adopts the NA‑Engine pattern:

- Automatically build **all** tests  
- Automatically run **all** tests  
- Run valgrind on **all** tests in dev/qa  
- CI requires **zero manual maintenance** when new tests are added  

This ensures the CI pipeline is always up‑to‑date without effort.

---

# **5. General CI Philosophy**

### ✔ CI must be strict in dev  
### ✔ CI must be complete in qa  
### ✔ CI must be lightweight in main  
### ✔ CI must be automatic  
### ✔ CI must require no manual updates  
### ✔ CI must fail fast  
### ✔ CI must be reproducible  

---

# **6. Benefits of This Design**

- Less maintenance  
- No duplication  
- Higher reliability  
- Faster pipelines  
- Full alignment with NA‑Engine  
- Scalable for future modules (autograd, optimizers, conv2d, etc.)

---

# **7. Conclusion**

This CI design allows PHOBOS‑ML to grow like a real ML engine:

- clear layers  
- clear responsibilities  
- automatic testing  
- guaranteed quality  
- stable releases  

It mirrors the architecture used by engines such as:

- NA‑Engine  
- PyTorch  
- TensorFlow  
- ONNX Runtime  

---
