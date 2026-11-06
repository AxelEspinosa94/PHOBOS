#include <stdlib.h>

#include "tensor.h"

int tensor_logreg_gradients(
    const tensor_t* X,
    const tensor_t* y,
    const tensor_t* y_hat,
    tensor_t** dW_out,
    float* db_out) {
    if (!X || !y || !y_hat || !dW_out || !db_out)
        return -1;

    if (X->ndim != 2 || y->ndim != 2 || y_hat->ndim != 2)
        return -2;

    size_t m = X->shape[0];
    size_t n = X->shape[1];

    if (y->shape[0] != m || y_hat->shape[0] != m)
        return -3;

    if (y->shape[1] != 1 || y_hat->shape[1] != 1)
        return -4;

    // Compute error: e = y_hat - y
    int shape_e[2] = {(int)m, 1};
    tensor_t* e = tensor_create(DTYPE_FLOAT32, 2, shape_e);
    if (!e) return -5;

    float* ed = (float*)e->data;
    float* yd = (float*)y->data;
    float* yhd = (float*)y_hat->data;

    for (size_t i = 0; i < m; ++i)
        ed[i] = yhd[i] - yd[i];

    // dW = X^T e
    int shape_dW[2] = {(int)n, 1};
    tensor_t* dW = tensor_create(DTYPE_FLOAT32, 2, shape_dW);
    if (!dW) {
        tensor_free(e);
        return -6;
    }

    float* dWd = (float*)dW->data;
    float* Xd = (float*)X->data;

    for (size_t j = 0; j < n; ++j) {
        float sum = 0.0f;
        for (size_t i = 0; i < m; ++i) {
            float xij = Xd[i * n + j];
            sum += xij * ed[i];
        }
        dWd[j] = sum;
    }

    // db = sum(e)
    float db = 0.0f;
    for (size_t i = 0; i < m; ++i)
        db += ed[i];

    tensor_free(e);

    *dW_out = dW;
    *db_out = db;

    return 0;
}
