# ================================
# PHOBOS-ML Makefile
# ================================

# Directories
SRC_DIR := src
TEST_DIR := tests
BUILD_DIR := build
INC_DIR := include
EXAMPLE_DIR := examples/spam_classifier

# Compiler
CC := gcc
CFLAGS += -Wall -Wextra -O2 -I$(INC_DIR)

# Linker flags (math library)
LDLIBS := -lm

# Source files
SRCS := $(wildcard $(SRC_DIR)/*.c)
TESTS := $(wildcard $(TEST_DIR)/*.c)

# Object files
OBJS := $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.o, $(SRCS))
TEST_OBJS := $(patsubst $(TEST_DIR)/%.c, $(BUILD_DIR)/%.o, $(TESTS))
EXAMPLE_OBJS := $(patsubst $(EXAMPLE_DIR)/%.c, $(BUILD_DIR)/%.o, $(EXAMPLES))

# Remove main.o from engine objects for tests
ENGINE_OBJS := $(filter-out $(BUILD_DIR)/main.o, $(OBJS))

# Binaries
TARGET := $(BUILD_DIR)/phobosml

# ================================
# Default target
# ================================
all: $(TARGET)

# ================================
# Main binary
# ================================
$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $(TARGET) $(LDLIBS) $(LDFLAGS)

# ================================
# Comp binary
# ================================
build/%: build/%.o $(ENGINE_OBJS)
	$(CC) $^ -o $@ $(LDLIBS) $(LDFLAGS)

build/%.o: $(EXAMPLE_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

# ================================
# Test binaries (auto-detected)
# ================================
# Example: build/test_tensor
$(BUILD_DIR)/test_%: $(BUILD_DIR)/test_%.o $(ENGINE_OBJS)
	$(CC) $^ -o $@ $(LDLIBS) $(LDFLAGS)

$(BUILD_DIR)/train_%: $(BUILD_DIR)/train_%.o $(ENGINE_OBJS)
	$(CC) $^ -o $@ $(LDLIBS) $(LDFLAGS)

$(BUILD_DIR)/predict_%: $(BUILD_DIR)/predict_%.o $(ENGINE_OBJS)
	$(CC) $^ -o $@ $(LDLIBS) $(LDFLAGS)

# ================================
# Generic compilation rules
# ================================
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(TEST_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(EXAMPLE_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

# ================================
# Formatting
# ================================
format:
	clang-format-18 -i $(SRC_DIR)/*.c $(TEST_DIR)/*.c $(INC_DIR)/*.h

# ================================
# Clean
# ================================
clean:
	rm -f $(BUILD_DIR)/*.o
	rm -f $(TARGET)
	rm -f $(BUILD_DIR)/test_*

# ================================
# Static library
# ================================
lib: $(ENGINE_OBJS)
	ar rcs $(BUILD_DIR)/libphobosml.a $(ENGINE_OBJS)

# ================================
# Shared library
# ================================
shared: CFLAGS += -fPIC
shared: $(ENGINE_OBJS)
	$(CC) -shared -o $(BUILD_DIR)/libphobosml.so $(ENGINE_OBJS)


# ============================
# PHOBOS-ML Release Automation
# ============================

ifeq ($(OS),Windows_NT)
WSL := wsl
PY := python.exe
PIP := pip.exe
else
WSL :=
PY := python3
PIP := pip3
endif

# Get latest tag (e.g., v0.1.1)
CURRENT_TAG := $(shell git describe --tags --abbrev=0 2>/dev/null)
ifeq ($(CURRENT_TAG),)
CURRENT_TAG := v0.0.1
endif


# Extract numeric version (remove leading "v")
VERSION := $(subst v,,$(CURRENT_TAG))

# Split version into components
MAJOR := $(word 1,$(subst ., ,$(VERSION)))
MINOR := $(word 2,$(subst ., ,$(VERSION)))
PATCH := $(word 3,$(subst ., ,$(VERSION)))

ifeq ($(OS),Windows_NT)
INC_PATCH = $(shell powershell -Command "$(PATCH) + 1")
INC_MINOR = $(shell powershell -Command "$(MINOR) + 1")
else
INC_PATCH = $(shell echo $$(($(PATCH)+1)))
INC_MINOR = $(shell echo $$(($(MINOR)+1)))
endif

NEXT_PATCH := $(INC_PATCH)
NEXT_PATCH_TAG := v$(MAJOR).$(MINOR).$(NEXT_PATCH)

NEXT_MINOR := $(INC_MINOR)
NEXT_MINOR_TAG := v$(MAJOR).$(NEXT_MINOR).0

# ============================
# Commands
# ============================

# Create a PATCH release (0.x.(y+1))
release-patch:
	@echo "Current tag: $(CURRENT_TAG)"
	@echo "Next patch tag: $(NEXT_PATCH_TAG)"
	git tag -a $(NEXT_PATCH_TAG) -m "Release $(NEXT_PATCH_TAG)"
	git push origin $(NEXT_PATCH_TAG)
	@echo "Patch release created."

# Create a MINOR release (0.(x+1).0)
release-minor:
	@echo "Current tag: $(CURRENT_TAG)"
	@echo "Next minor tag: $(NEXT_MINOR_TAG)"
	git tag -a $(NEXT_MINOR_TAG) -m "Release $(NEXT_MINOR_TAG)"
	git push origin $(NEXT_MINOR_TAG)
	@echo "Minor release created."

release-patch-dry:
	@echo "Current tag: $(CURRENT_TAG)"
	@echo "Next patch tag: $(NEXT_PATCH_TAG)"
	@echo "[DRY RUN] Would run: git tag -a $(NEXT_PATCH_TAG) -m 'Release $(NEXT_PATCH_TAG)'"
	@echo "[DRY RUN] Would run: git push origin $(NEXT_PATCH_TAG)"
	@echo "Dry run completed."
