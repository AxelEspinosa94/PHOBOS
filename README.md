
---

# PHOBOS Project

[![CI](https://github.com/AxelEspinosa94/PHOBOS-ML/actions/workflows/ci.yml/badge.svg)](https://github.com/AxelEspinosa94/PHOBOS-ML/actions/workflows/ci.yml)
[![Version](https://img.shields.io/github/v/tag/AxelEspinosa94/PHOBOS?label=version)]()
![Coverage](https://img.shields.io/badge/coverage-lcov-blue.svg)
![Valgrind](https://img.shields.io/badge/memory%20check-valgrind-critical.svg)
![Static Library](https://img.shields.io/badge/library-.a-blue.svg)
![Shared Library](https://img.shields.io/badge/library-.so-green.svg)
![Auto Tagging](https://img.shields.io/badge/release-auto--tagging-purple.svg)
![Tests](https://img.shields.io/badge/tests-automated-success.svg)
![Language](https://img.shields.io/badge/language-C-blue.svg)
![Build](https://img.shields.io/badge/build-Makefile-orange.svg)
![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20WSL2-green.svg)
![License](https://img.shields.io/badge/license-MIT-yellow.svg)
![Status](https://img.shields.io/badge/status-Active-success.svg)

**PHOBOS** is a lightweight **tensor engine written in pure C**. On top of it sit independent modules, each built as its own library:

- **PHOBOS Core** — tensors and tensor operations (the foundation everything else links against)
- **PHOBOS ML** — models and optimizers (logistic regression, SGD)
- **TLALOC** — *Tensor Library for Analysis of Continuum Systems*, a fluid mechanics module that solves canonical viscous flow problems using PHOBOS tensors

Design goals:

- **Modularity** — each module is a separate library with explicit, one-way dependencies
- **Performance** — minimal overhead, close to the metal
- **Interoperability** — built to integrate with Python (ctypes/cffi) and Java (JNI)
- **Extensibility** — new domain modules can be added without touching the core

This project is part of a broader ecosystem:

- **A)** PHOBOS — Core tensor engine in C, plus domain modules (ML, TLALOC)
- **B)** Spark‑ML Java Project — Custom Transformers & Estimators for MLlib (TBD)
- **C)** Hybrid Engine — C + Python + Java + Spark integration layer (TBD)

---

## 🏗️ Architecture

Every module compiles into its own static (`.a`) and shared (`.so`) library. Dependencies point in one direction only: modules depend on the core, the core depends on nothing, and modules never depend on each other.

```
                ┌──────────────────────┐
                │     PHOBOS Core      │
                │   libphobos.a / .so  │
                │   tensor · ops       │
                └──────────┬───────────┘
                           │
             ┌─────────────┴─────────────┐
             │                           │
  ┌──────────▼───────────┐    ┌──────────▼───────────┐
  │      PHOBOS ML       │    │        TLALOC        │
  │  libphobos_ml.a/.so  │    │   libtlaloc.a / .so  │
  │  model · optimizer   │    │   fluid mechanics    │
  └──────────────────────┘    └──────────┬───────────┘
                                         │
                              ┌──────────▼───────────┐
                              │   Python UI (ctypes) │
                              └──────────────────────┘
```

Because TLALOC never links `libphobos_ml`, changes to the ML module cannot affect fluid mechanics, and vice versa.

### Naming conventions

C has no namespaces, so each module scopes its public API in two ways:

| Module      | Header include                 | Symbol prefix  |
|-------------|--------------------------------|----------------|
| Core        | `#include <phobos/tensor.h>`   | `phobos_`      |
| ML          | `#include <phobos_ml/model.h>` | `phobos_ml_`   |
| TLALOC      | `#include <tlaloc/couette.h>`  | `tlaloc_`      |

This prevents symbol collisions when several shared libraries are loaded into the same Python process.

---

## 📁 Project Structure

```
PHOBOS/
│
├── core/                       # PHOBOS Core → libphobos
│   ├── include/phobos/
│   │   ├── tensor.h
│   │   └── ops.h
│   ├── src/
│   │   ├── tensor.c
│   │   └── ops.c
│   ├── tests/
│   │   ├── test_tensor.c
│   │   └── test_ops.c
│   └── Makefile
│
├── ml/                         # PHOBOS ML → libphobos_ml (depends on core)
│   ├── include/phobos_ml/
│   │   ├── model.h
│   │   └── optimizer.h
│   ├── src/
│   │   ├── model.c
│   │   └── optimizer.c
│   ├── tests/
│   │   └── test_model.c
│   ├── examples/
│   │   └── example_lr.c
│   └── Makefile
│
├── tlaloc/                     # TLALOC → libtlaloc (depends on core)
│   ├── include/tlaloc/
│   │   ├── tlaloc.h            # Umbrella header
│   │   ├── couette.h
│   │   ├── poiseuille.h
│   │   └── stokes.h
│   ├── src/
│   │   ├── couette.c
│   │   ├── poiseuille.c
│   │   └── stokes.c
│   ├── tests/
│   │   ├── test_couette.c
│   │   ├── test_poiseuille.c
│   │   └── test_stokes.c
│   ├── examples/
│   │   ├── example_couette.c
│   │   ├── example_poiseuille.c
│   │   └── example_stokes_first.c
│   └── Makefile
│
├── bindings/
│   └── python/                 # ctypes wrapper + Python UI for TLALOC
│       ├── tlaloc/
│       └── ui/
│
├── docs/                       # Theory notes, derivations, references
├── build/                      # Compiled artifacts (ignored by Git)
│
├── Makefile                    # Top-level: delegates to each module's Makefile
├── README.md
└── .gitignore
```

> **Migration note:** the repository currently uses a flat `include/` + `src/` layout. Moving to the structure above, including splitting the ML module out of the core, is part of the hardening EPIC.

---

## 🌊 TLALOC — Canonical Problems

TLALOC solves classical incompressible, viscous, Newtonian flows for which analytical solutions exist. That makes every numerical result verifiable against a closed form. The current scope covers three problems:

### 1. Couette flow

Steady flow between two infinite parallel plates separated by a distance $h$, with the upper plate moving at constant velocity $U$ and the lower plate fixed. With no imposed pressure gradient, the velocity profile is linear:

$$
u(y) = U\,\frac{y}{h}
$$

With an imposed pressure gradient $\frac{dp}{dx}$, the profile combines the shear-driven and pressure-driven parts:

$$
u(y) = U\,\frac{y}{h} - \frac{h^2}{2\mu}\frac{dp}{dx}\,\frac{y}{h}\left(1 - \frac{y}{h}\right)
$$

### 2. Plane Poiseuille flow

Steady, pressure-driven flow between two fixed infinite parallel plates. The velocity profile is parabolic:

$$
u(y) = -\frac{1}{2\mu}\frac{dp}{dx}\,y\,(h - y)
$$

with maximum velocity at the channel centerline, $y = h/2$.

### 3. Stokes' first problem (impulsively started plate)

An infinite flat plate in a fluid at rest is suddenly set in motion at velocity $U$ at $t = 0$. Momentum diffuses into the fluid according to

$$
\frac{\partial u}{\partial t} = \nu\,\frac{\partial^2 u}{\partial y^2}
$$

whose similarity solution is

$$
u(y,t) = U\,\operatorname{erfc}\!\left(\frac{y}{2\sqrt{\nu t}}\right)
$$

This is the first time-dependent problem in TLALOC and the first one that requires time integration in the numerical solver.

### Planned API

```c
#include <tlaloc/tlaloc.h>

/* Example: Couette flow profile sampled on n points */
phobos_tensor *u = tlaloc_couette_profile(U, h, dpdx, mu, n);
```

> Function signatures are subject to change until the first TLALOC release.

---

## 🚀 Build Instructions

Build every module:

```bash
make
```

Build a single module (dependencies are built automatically):

```bash
make core
make ml
make tlaloc
```

Clean build artifacts:

```bash
make clean
```

---

## 🧪 Running Tests

Run the full test suite:

```bash
make test
```

Run tests for a single module:

```bash
make test-core
make test-tlaloc
```

---

## 📜 License (MIT)

This project is licensed under the **MIT License**, a permissive open‑source license that allows commercial and private use.

```
MIT License

Copyright (c) 2026 Axel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## 🧩 Roadmap

### PHOBOS Core & ML
- [X] Implement `tensor.c` (core tensor structure)
- [X] Add basic operations (MatMul, ReLU, Softmax)
- [X] Implement Logistic Regression model
- [X] Add SGD optimizer
- [X] Build shared library `.so` for Python bindings
- [ ] Split repository into `core/`, `ml/` and `tlaloc/` modules (hardening EPIC)
- [ ] Create JNI wrapper for Java
- [ ] Integrate with Spark MLlib (Transformer + Estimator)
- [ ] Add benchmarks and profiling tools

### TLALOC
- [ ] Couette flow (with and without pressure gradient)
- [ ] Plane Poiseuille flow
- [ ] Stokes' first problem
- [ ] Validation of numerical results against analytical solutions
- [ ] Python bindings (ctypes) and UI

---

## 📚 References

Batchelor, G. K. (1967). *An introduction to fluid dynamics*. Cambridge University Press.

Stokes, G. G. (1851). On the effect of the internal friction of fluids on the motion of pendulums. *Transactions of the Cambridge Philosophical Society, 9*, 8–106.

White, F. M. (2006). *Viscous fluid flow* (3rd ed.). McGraw-Hill.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome.
Feel free to open a PR or start a discussion.

---

## ⭐ Acknowledgements

This project is inspired by the internals of frameworks such as:

- PyTorch (C++ backend)
- TensorFlow (C++ core)
- ONNX Runtime
- XGBoost / LightGBM

PHOBOS aims to provide an educational yet production‑oriented view of how numerical and ML engines work under the hood.

---

# 🚀 VSCode + Ubuntu (WSL) Integration — Quick Guide

### **1. Install the VSCode extension**
In VSCode:

- Go to **Extensions**
- Install the **WSL** extension (by Microsoft)

### **2. Open Ubuntu in VSCode**
In Windows:

- Open VSCode
- Press `Ctrl + Shift + P`
- Run:

```
WSL: Connect to WSL using distro
```

and select Ubuntu.

### **3. Open your project folder**
In the Ubuntu terminal:

```bash
cd ~/PHOBOS
code .
```

VSCode will open **inside Ubuntu**, using Linux compilers, paths, and terminals.

### **4. Install Git (if not installed)**
```bash
sudo apt install git
```

Configure:

```bash
git config --global user.name <user-name>
git config --global user.email <user-email>
```

### **5. Push to GitHub**
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/youruser/PHOBOS.git
git push -u origin main
```