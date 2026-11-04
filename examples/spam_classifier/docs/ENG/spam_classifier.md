# Use Case: Spam Classifier

**Location:** `examples/spam_classifier/`
**Modules used:** PHOBOS Core (tensors, data I/O) and PHOBOS ML (logistic regression, training, model I/O)

This example trains a logistic regression model that decides whether an SMS message is **spam** or **ham** (legitimate). It exercises the whole PHOBOS pipeline end to end: loading data from CSV, training, saving the model, loading it back, and running inference.

---

## Pipeline overview

The workflow has four stages. Preprocessing is done in Python; everything else runs on the PHOBOS C engine.

```
                         dataset.csv
                              │
                              ▼
                  ┌───────────────────────┐
                  │   preprocess.py       │  Python
                  └───────────┬───────────┘
              ┌───────────────┼────────────────┐
              ▼               ▼                ▼
       vocabulary.txt       X.csv            y.csv
              │               │                │
              │               └───────┬────────┘
              │                       ▼
              │           ┌───────────────────────┐
              │           │   train_spam          │  C
              │           └───────────┬───────────┘
              │                       ▼
              │                spam_model.bin
              │                       │
              │        ┌──────────────┴──────────────┐
              ▼        ▼                             ▼
   ┌───────────────────────┐            ┌───────────────────────┐
   │   predict_spam        │  C         │   accuracy            │  C
   │   one message         │            │   whole dataset       │
   └───────────────────────┘            └───────────────────────┘
```

| File | Kind | Produced by | Used by |
|---|---|---|---|
| `dataset.csv` | Input data | — | `preprocess.py` |
| `vocabulary.txt` | Text, one word per line | `preprocess.py` | `predict_spam` |
| `X.csv` | Feature matrix (0/1), with header | `preprocess.py` | `train_spam`, `accuracy` |
| `y.csv` | Labels (0/1), with header | `preprocess.py` | `train_spam`, `accuracy` |
| `spam_model.bin` | Trained model (PHOBOS binary format) | `train_spam` | `predict_spam`, `accuracy` |

---

## Dataset

`dataset.csv` contains 5,573 SMS messages in two columns:

| Column | Content |
|---|---|
| `label` | `ham` or `spam` |
| `text` | Raw message text |

| Class | Messages | Share |
|---|---|---|
| ham | 4,826 | 86.6 % |
| spam | 747 | 13.4 % |

The classes are **imbalanced**. A model that always answers "ham" already reaches 86.6 % accuracy, so that figure is the baseline any real model has to beat.

---

## Stage 1 — Preprocessing (`preprocess.py`)

Turns raw text into a numeric representation that the C engine can load.

### Steps

1. **Labels.** `label` is lowercased and trimmed, then mapped to `ham → 0`, `spam → 1`.
2. **Text cleaning.** Each message is lowercased, every character outside `a–z`, `0–9` and whitespace is replaced by a space, and runs of whitespace are collapsed into one.
3. **Tokenization.** The cleaned text is split on spaces.
4. **Vocabulary.** The 200 most frequent tokens across the whole dataset are selected (`VOCAB_SIZE = 200`).
5. **Vectorization.** Each message becomes a binary vector of length 200: position $j$ is 1 if the $j$-th vocabulary word appears **as a token** in the message, 0 otherwise (bag of words, presence only).

### Outputs

| File | Shape | Notes |
|---|---|---|
| `vocabulary.txt` | 200 lines | Ordered from most to least frequent. The order defines the feature index. |
| `X.csv` | `(5573, 200)` | Header row with the vocabulary words. |
| `y.csv` | `(5573, 1)` | Header row `label`. |

Both CSV files have a header, which is why the C programs load them with `skip_header = 1`.

---

## Stage 2 — Training (`train_spam.c`)

Trains the model on `X.csv` and `y.csv` and saves it to `spam_model.bin`.

### Steps

1. Load `X` and `y` with `tensor_load_csv(path, 1)`.
2. Create `W` of shape `(200, 1)` and initialize it to zero. Set `b = 0`.
3. Call `train_loop(X, y, W, &b, LEARNING_RATE, EPOCHS)`, which prints the loss and bias after every epoch.
4. Save the result with `model_save("spam_model.bin", W, b)`.

### Hyperparameters

| Constant | Value |
|---|---|
| `LEARNING_RATE` | `0.03` |
| `EPOCHS` | `2500` |
| Weight initialization | zeros |
| Batch | full dataset every epoch |

---

## Stage 3 — Single prediction (`predict_spam.c`)

Classifies one message typed by the user.

### Steps

1. Read up to 200 words from `vocabulary.txt`. The number of words read defines the shape of `W`.
2. Create `W` with shape `(vocab_count, 1)` and load it with `model_load`. If the model was trained with a different vocabulary size, loading fails with a shape error.
3. Read one line from `stdin` (up to 4,095 characters).
4. Build the feature vector: position $j$ is 1 if the $j$-th vocabulary word occurs in the message, using `strstr`.
5. Run `tensor_logreg_forward(X, W, NULL)` and print the probability.
6. Print `SPAM` if the probability is ≥ 0.5, `HAM` otherwise.

### Example session

```
Enter message:
free entry win a prize now

Spam probability: 0.9561
Prediction: SPAM
```

> ⚠️ Features are built differently here than in training. See [Train/serve skew](#trainserve-skew-in-predict_spam).

---

## Stage 4 — Evaluation (`accuracy.c`)

Runs the model over every row of `X.csv` and compares predictions with `y.csv`.

### Steps

1. Load `X`, `y` and the model.
2. For each row, copy it into a `(1, 200)` tensor, run `tensor_logreg_forward(sample, W, NULL)` and apply the 0.5 threshold.
3. Count the four outcomes of the confusion matrix and print them with the accuracy.

### Confusion matrix

|  | Predicted spam | Predicted ham |
|---|---|---|
| **Actual spam** | TP | FN |
| **Actual ham** | FP | TN |

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

---

## Results

The figures below come from a NumPy replica of the pipeline that follows the C code's exact semantics (same preprocessing, zero initialization, same learning rate and epochs, same gradient formulas). It reproduces the ~96 % accuracy observed with the C build. Exact counts from the C binaries may differ slightly due to floating-point details.

| Metric | Value |
|---|---|
| Accuracy | **96.61 %** (baseline: 86.60 %) |
| TP / FN | 678 / 69 |
| FP / TN | 120 / 4,706 |
| Spam recall $\frac{TP}{TP+FN}$ | 90.8 % |
| Spam precision $\frac{TP}{TP+FP}$ | 85.0 % |

### How to read these numbers

- **This is training accuracy.** `accuracy` evaluates the model on the same rows it was trained on. There is no train/test split, so the figure measures how well the model fits the data it has seen, not how well it generalizes to new messages.
- **Compare against the baseline.** Because 86.6 % of messages are ham, the meaningful gain is from 86.6 % to 96.6 %, and recall and precision on the spam class describe the model better than accuracy alone.

---

## How to run

All programs use paths relative to the repository root, so they must be run **from the root** of the repository.

```bash
# 1. Preprocess (requires pandas and numpy)
python3 examples/spam_classifier/preprocess.py

# 2. Train   — writes examples/spam_classifier/spam_model.bin
./train_spam

# 3. Evaluate on the dataset
./accuracy

# 4. Classify a single message
./predict_spam
```

> The executable names and locations depend on how the examples are built in the Makefile.

---

## Known limitations

### Train/serve skew in `predict_spam`

Training features come from `preprocess.py`, which lowercases the text, strips punctuation and checks for **whole tokens**. `predict_spam` instead searches the raw message with `strstr`, which is **case-sensitive** and matches **substrings**. Short vocabulary words such as `a`, `i`, `u`, `s`, `t` or `m` therefore match inside almost any word, while capitalized words such as `FREE` do not match `free`.

Running the dataset through `predict_spam`'s feature extraction instead of `preprocess.py`'s shows the impact:

| Feature extraction | Accuracy | Spam recall |
|---|---|---|
| Tokens (`preprocess.py`, as in training and `accuracy`) | 96.61 % | 90.8 % |
| Substrings (`strstr`, as in `predict_spam`) | 88.48 % | 29.0 % |

With substring matching, the model misses about 7 in 10 spam messages. To fix it, `predict_spam` should apply the same cleaning and tokenization as `preprocess.py`.

### The bias is not used

Training, prediction and evaluation all call `tensor_logreg_forward` with `NULL` as the bias. The bias is trained and saved (in the replica it converges to about −2.0), but it never affects a prediction. This is consistent across the three programs, but once `train_loop` is fixed to use the bias, `predict_spam` and `accuracy` must pass it too.

### Other limitations

- **Probabilities saturate.** Many predictions print as `1.0000` or `0.0000`. This is partly due to the weight gradient being larger than intended (see the gradients documentation).
- **Vocabulary size appears in two places.** `VOCAB_SIZE = 200` is defined in both `preprocess.py` and `predict_spam.c`. If the Python value grows beyond 200, `predict_spam` reads only the first 200 words and `model_load` rejects the model.
- **Unknown labels.** A label other than `ham` or `spam` becomes `NaN` in `y.csv`, which the CSV loader then rejects.
- **Inefficient evaluation.** `accuracy` allocates a new tensor for every row instead of running a single forward pass over all of `X`.
- **Unchecked input.** `predict_spam` does not check the result of `fgets`. If `stdin` is closed, the message buffer is used uninitialized.
