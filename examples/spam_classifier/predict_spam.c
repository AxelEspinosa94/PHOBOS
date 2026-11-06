#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model_io.h"
#include "tensor.h"

#define VOCAB_SIZE 200
#define MAX_WORD_LEN 128
#define MAX_TEXT_LEN 4096

static int word_exists(
    const char* text,
    const char* word) {
    return strstr(text, word) != NULL;
}

int main(void) {
    char vocabulary[VOCAB_SIZE][MAX_WORD_LEN];

    FILE* vocab =
        fopen(
            "./examples/spam_classifier/vocabulary.txt",
            "r");

    if (!vocab) {
        printf(
            "ERROR: Could not open vocabulary.txt\n");

        return 1;
    }

    int vocab_count = 0;

    while (
        vocab_count < VOCAB_SIZE &&
        fgets(
            vocabulary[vocab_count],
            MAX_WORD_LEN,
            vocab)) {
        vocabulary[vocab_count]
                  [strcspn(
                      vocabulary[vocab_count],
                      "\r\n")] = '\0';

        vocab_count++;
    }

    fclose(vocab);

    int w_shape[2] = {
        vocab_count,
        1};

    tensor_t* W =
        tensor_create(
            DTYPE_FLOAT32,
            2,
            w_shape);

    float bias = 0.0f;

    if (
        model_load(
            "./examples/spam_classifier/spam_model.bin",
            W,
            &bias) != 0) {
        printf(
            "ERROR: Could not load model\n");

        return 1;
    }

    char message[MAX_TEXT_LEN];

    printf(
        "Enter message:\n");

    fgets(
        message,
        sizeof(message),
        stdin);

    int x_shape[2] = {
        1,
        vocab_count};

    tensor_t* X =
        tensor_create(
            DTYPE_FLOAT32,
            2,
            x_shape);

    float* xd =
        (float*)X->data;

    for (int i = 0;
         i < vocab_count;
         ++i) {
        xd[i] =
            word_exists(
                message,
                vocabulary[i])
                ? 1.0f
                : 0.0f;
    }

    tensor_t* y_hat =
        tensor_logreg_forward(
            X,
            W,
            NULL);

    if (!y_hat) {
        printf(
            "ERROR: Forward pass failed\n");

        return 1;
    }

    float probability =
        ((float*)y_hat->data)[0];

    printf(
        "\nSpam probability: %.4f\n",
        probability);

    if (probability >= 0.5f) {
        printf(
            "Prediction: SPAM\n");

    } else {
        printf(
            "Prediction: HAM\n");
    }

    tensor_free(X);
    tensor_free(W);
    tensor_free(y_hat);

    return 0;
}