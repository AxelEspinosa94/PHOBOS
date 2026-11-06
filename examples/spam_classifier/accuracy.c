#include <stdio.h>

#include "data_io.h"
#include "model_io.h"
#include "tensor.h"

#define THRESHOLD 0.5f

int main(void) {

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
            "ERROR: Failed loading dataset\n");

        return 1;
    }

    int w_shape[2] = {
        X->shape[1],
        1
    };

    tensor_t* W =
        tensor_create(
            DTYPE_FLOAT32,
            2,
            w_shape);

    float bias = 0.0f;

    if (model_load(
            "./examples/spam_classifier/spam_model.bin",
            W,
            &bias) != 0) {

        printf(
            "ERROR: Failed loading model\n");

        return 1;
    }

    int rows =
        X->shape[0];

    int cols =
        X->shape[1];

    float* Xd =
        (float*)X->data;

    float* yd =
        (float*)y->data;

    int TP = 0;
    int TN = 0;
    int FP = 0;
    int FN = 0;

    for (int r = 0;
         r < rows;
         ++r) {

        int sample_shape[2] = {
            1,
            cols
        };

        tensor_t* sample =
            tensor_create(
                DTYPE_FLOAT32,
                2,
                sample_shape);

        float* sd =
            (float*)sample->data;

        for (int c = 0;
             c < cols;
             ++c) {

            sd[c] =
                Xd[r * cols + c];
        }

        /*
         * IMPORTANT:
         * Use EXACTLY the same
         * inference path as predict_spam.
         */
        tensor_t* pred =
            tensor_logreg_forward(
                sample,
                W,
                NULL);

        if (!pred) {

            tensor_free(sample);
            continue;
        }

        float probability =
            ((float*)pred->data)[0];

        /*
         * Same threshold as predictor
         */
        int predicted =
            probability >= THRESHOLD
            ? 1
            : 0;

        int actual =
            (int)yd[r];

        if (predicted == 1 &&
            actual == 1)
            TP++;

        else if (predicted == 0 &&
                 actual == 0)
            TN++;

        else if (predicted == 1 &&
                 actual == 0)
            FP++;

        else
            FN++;

        tensor_free(sample);
        tensor_free(pred);
    }

    int total =
        TP + TN + FP + FN;

    float accuracy =
        (float)(TP + TN) /
        (float)total;

    printf(
        "\n========== RESULTS ==========\n\n");

    printf(
        "Samples : %d\n",
        total);

    printf(
        "TP      : %d\n",
        TP);

    printf(
        "TN      : %d\n",
        TN);

    printf(
        "FP      : %d\n",
        FP);

    printf(
        "FN      : %d\n",
        FN);

    printf(
        "\nAccuracy: %.2f%%\n",
        accuracy * 100.0f);

    tensor_free(X);
    tensor_free(y);
    tensor_free(W);

    return 0;
}