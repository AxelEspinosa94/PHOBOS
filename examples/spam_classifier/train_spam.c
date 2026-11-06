#include <stdio.h>
#include <stdlib.h>

#include "data_io.h"
#include "model_io.h"
#include "tensor.h"
#include "train.h"

#define LEARNING_RATE 0.03f
#define EPOCHS 2500

int main(void) {
    printf("Loading dataset...\n");

    tensor_t* X =
        tensor_load_csv(
            "./examples/spam_classifier/X.csv",
            1);

    tensor_t* y =
        tensor_load_csv(
            "./examples/spam_classifier/y.csv",
            1);

    if (!X || !y) {
        printf(
            "ERROR: Failed to load dataset\n");

        return 1;
    }

    printf(
        "X shape: [%d, %d]\n",
        X->shape[0],
        X->shape[1]);

    printf(
        "y shape: [%d, %d]\n",
        y->shape[0],
        y->shape[1]);

    /*
        W shape:
        (n_features × 1)
    */
    int w_shape[2] = {
        X->shape[1],
        1};

    tensor_t* W =
        tensor_create(
            DTYPE_FLOAT32,
            2,
            w_shape);

    if (!W) {
        printf(
            "ERROR: Failed to allocate weights\n");

        return 1;
    }

    /*
        Initialize weights to zero
    */
    float* wd =
        (float*)W->data;

    for (size_t i = 0;
         i < W->size;
         ++i) {
        wd[i] = 0.0f;
    }

    float b = 0.0f;

    printf(
        "\nTraining...\n\n");

    int err =
        train_loop(
            X,
            y,
            W,
            &b,
            LEARNING_RATE,
            EPOCHS);

    if (err != 0) {
        printf(
            "ERROR: train_loop failed (%d)\n",
            err);

        tensor_free(X);
        tensor_free(y);
        tensor_free(W);

        return 1;
    }

    printf(
        "\nSaving model...\n");

    err =
        model_save(
            "./examples/spam_classifier/spam_model.bin",
            W,
            b);

    if (err != 0) {
        printf(
            "ERROR: model_save failed (%d)\n",
            err);

        tensor_free(X);
        tensor_free(y);
        tensor_free(W);

        return 1;
    }

    printf(
        "Model saved successfully.\n");

    printf(
        "Final bias = %.6f\n",
        b);

    tensor_free(X);
    tensor_free(y);
    tensor_free(W);

    printf(
        "\nDone.\n");

    return 0;
}