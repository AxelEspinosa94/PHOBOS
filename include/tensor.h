#ifndef PHOBOSML_TENSOR_H
#define PHOBOSML_TENSOR_H

#include <stdbool.h>  // bool
#include <stddef.h>   // size_t
#include <stdint.h>   // uint32_t, uint64_t

/**
 * @file tensor.h
 * @brief Core tensor structure and API for PHOBOS-ML.
 *
 * This header defines the fundamental tensor type used throughout the
 * PHOBOS-ML engine. It includes metadata such as shape, strides, dtype,
 * and memory ownership flags. All operations, autograd components, and
 * model layers depend on this interface.
 *
 * The header is designed to compile standalone and serve as a stable
 * public API for C, Python (via FFI), and other language bindings.
 */

/* ============================================================
   Supported data types
   ============================================================ */

/**
 * @enum dtype_t
 * @brief Enumerates supported tensor element types.
 */
typedef enum {
    DTYPE_FLOAT32, /**< 32-bit floating point */
    DTYPE_FLOAT64, /**< 64-bit floating point */
    DTYPE_INT32,   /**< 32-bit integer */
    DTYPE_INT64    /**< 64-bit integer */
} dtype_t;

/* ============================================================
   Tensor structure
   ============================================================ */

/**
 * @struct tensor_t
 * @brief Represents an n-dimensional array with metadata.
 *
 * A tensor contains:
 * - A raw memory pointer (`data`)
 * - A data type (`dtype`)
 * - Dimensionality (`ndim`)
 * - Shape array (`shape`)
 * - Strides array (`strides`)
 * - Total number of elements (`size`)
 * - Total memory footprint (`bytes`)
 * - Ownership flag (`owns_data`)
 *
 * Strides are expressed in **bytes**, not elements.
 */
typedef struct {
    void* data;      /**< Pointer to raw memory */
    dtype_t dtype;   /**< Element type */
    int ndim;        /**< Number of dimensions */
    int* shape;      /**< Array of dimension sizes */
    size_t* strides; /**< Array of strides (in bytes) */
    size_t size;     /**< Total number of elements */
    size_t bytes;    /**< Total memory size in bytes */
    bool owns_data;  /**< True if tensor owns its memory */
} tensor_t;

/* ============================================================
   Creation / destruction
   ============================================================ */

/**
 * @brief Creates a new tensor and allocates memory.
 *
 * Computes strides, total size, and memory footprint. Allocates
 * contiguous memory for the tensor elements.
 *
 * @param dtype Element type.
 * @param ndim Number of dimensions.
 * @param shape Array of dimension sizes.
 * @return Pointer to a newly allocated tensor, or NULL on failure.
 */
tensor_t* tensor_create(dtype_t dtype, int ndim, const int* shape);

/**
 * @brief Creates a tensor that views external memory.
 *
 * Does NOT allocate memory. Useful for wrapping buffers from Python,
 * Java, or other languages.
 *
 * @param data External memory pointer.
 * @param dtype Element type.
 * @param ndim Number of dimensions.
 * @param shape Array of dimension sizes.
 * @return Pointer to a tensor view, or NULL on failure.
 */
tensor_t* tensor_from_data(void* data, dtype_t dtype, int ndim, const int* shape);

/**
 * @brief Frees a tensor and its memory if owned.
 *
 * If `owns_data` is false, only metadata is freed.
 *
 * @param t Tensor to free.
 */
void tensor_free(tensor_t* t);

/* ============================================================
   Utility functions
   ============================================================ */

/**
 * @brief Returns the size in bytes of a single element of the given dtype.
 *
 * @param dtype Element type.
 * @return Size in bytes.
 */
size_t tensor_dtype_size(dtype_t dtype);

/**
 * @brief Checks whether the tensor is stored contiguously in memory.
 *
 * @param t Tensor to inspect.
 * @return True if contiguous, false otherwise.
 */
bool tensor_is_contiguous(const tensor_t* t);

/**
 * @brief Prints tensor metadata for debugging.
 *
 * @param t Tensor to print.
 */
void tensor_print(const tensor_t* t);

#endif  // PHOBOSML_TENSOR_H

/**
 * @brief Compares two tensors for equality (metadata + data).
 *
 * Does not print anything. Intended for tests and debugging.
 *
 * @param a First tensor.
 * @param b Second tensor.
 * @return true if tensors are identical, false otherwise.
 */
bool tensor_compare(const tensor_t* a, const tensor_t* b);

/**
 * @brief Matrix multiplication: C = A × B
 *
 * @param A Left matrix (m × k)
 * @param B Right matrix (k × n)
 * @return tensor_t* Result matrix (m × n), or NULL on shape mismatch.
 */
tensor_t* tensor_matmul(const tensor_t* A, const tensor_t* B);

/**
 * @brief Element-wise addition: C = A + B
 *
 * Supports NumPy-style broadcasting.
 *
 * @param A First tensor.
 * @param B Second tensor.
 * @return New tensor with result, or NULL on incompatible shapes.
 */
tensor_t* tensor_add(const tensor_t* A, const tensor_t* B);

/**
 * @brief Element-wise subtraction: C = A - B
 */
tensor_t* tensor_sub(const tensor_t* A, const tensor_t* B);

/**
 * @brief Element-wise multiplication: C = A * B
 */
tensor_t* tensor_mul(const tensor_t* A, const tensor_t* B);

/**
 * @brief tensor creation from another one
 */
tensor_t* tensor_copy(const tensor_t* src);

/* ============================================================
   Activation functions
   ============================================================ */

/**
 * @brief ReLU activation: max(x, 0)
 *
 * Supports broadcasting.
 */
tensor_t* tensor_relu(const tensor_t* A);

/**
 * @brief Sigmoid activation: 1 / (1 + exp(-x))
 *
 * Numerically stable implementation.
 */
tensor_t* tensor_sigmoid(const tensor_t* A);

/**
 * @brief Softmax activation along the last dimension.
 *
 * Uses max-subtraction trick to avoid overflow.
 */
tensor_t* tensor_softmax(const tensor_t* A);

/**
 * @brief Logistic Regression forward pass: sigmoid(XW + b)
 *
 * X: (m × n)
 * W: (n × 1)
 * b: (1 × 1) or (m × 1)
 *
 * @return Predictions (m × 1), or NULL on shape mismatch.
 */
tensor_t* tensor_logreg_forward(const tensor_t* X, const tensor_t* W, const tensor_t* b);

/**
 * @brief Sum of all elements in a tensor.
 */
float tensor_sum(const tensor_t* A);

/**
 * @brief Mean of all elements in a tensor.
 */
float tensor_mean(const tensor_t* A);

/**
 * @brief Max of all elements in a tensor.
 */
float tensor_max(const tensor_t* A);

/**
 * @brief Binary Cross-Entropy loss (mean reduction).
 *
 * Computes:
 *   BCE = -mean( y*log(p) + (1-y)*log(1-p) )
 *
 * Stable for extreme values using clamping.
 *
 * @param y  Ground truth labels (m × 1)
 * @param p  Predictions (m × 1), output of sigmoid
 * @return   Scalar loss (float)
 */
float tensor_bce_loss(const tensor_t* y, const tensor_t* p);

/**
 * @brief Compute gradients for logistic regression.
 *
 * Given:
 *   y_hat = sigmoid(XW + b)
 *
 * Gradients:
 *   dW = X^T (y_hat - y)
 *   db = sum(y_hat - y)
 *
 * Shapes:
 *   X: (m × n)
 *   y: (m × 1)
 *   y_hat: (m × 1)
 *   dW: (n × 1)
 *   db: scalar (float)
 *
 * @return 0 on success, non-zero on error.
 */
int tensor_logreg_gradients(
    const tensor_t* X,
    const tensor_t* y,
    const tensor_t* y_hat,
    tensor_t** dW_out,
    float* db_out);

/**
 * @brief Perform a single SGD training step for logistic regression.
 *
 * Computes:
 *   y_hat = sigmoid(XW + b)
 *   loss  = BCE(y, y_hat)
 *   dW, db = gradients
 *
 * Updates:
 *   W := W - lr * dW
 *   b := b - lr * db
 *
 * @param X   Input matrix (m × n)
 * @param y   Labels (m × 1)
 * @param W   Weights (n × 1) — updated in-place
 * @param b   Bias (scalar) — updated in-place
 * @param lr  Learning rate
 * @param loss_out  Output loss (float)
 *
 * @return 0 on success, non-zero on error.
 */
int tensor_logreg_train_step(
    const tensor_t* X,
    const tensor_t* y,
    tensor_t* W,
    float* b,
    float lr,
    float* loss_out);
