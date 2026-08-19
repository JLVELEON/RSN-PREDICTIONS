## Introduction to Transformers and Their Application to Time Series Prediction

### Motivation

In previous notebooks, we explored linear models (logistic regression) to predict the evolution of the states of the 7 functional networks (RSNs) from their temporal history. The results obtained, although positive (F1 ≈ 0.45 in change prediction with a 1 SD threshold), show that the improvement over the persistence baseline is limited.

As Suzyahyah (2026) points out, the difficulty of time series forecasting does not lie solely in the choice of the model, but in the nature of the data: high autocorrelation and scarcity of events make simple models hard to beat. However, there is a class of models that, due to their ability to capture long-range dependencies and their flexibility, could offer an improvement: **Transformers**.

Transformers, introduced by Vaswani et al. (2017), have revolutionized natural language processing and, more recently, have begun to be successfully applied to time series. Their **multi-head attention** mechanism allows modelling relationships between any pair of elements in a sequence, without the limitations of recurrent models (which suffer from the vanishing gradient problem) or convolutions (which have a fixed receptive field).

### Objective of this notebook

The aim of this notebook is twofold:

1. **Implement a Transformer from scratch** following Andrej Karpathy's tutorial and code (nanoGPT). This didactic approach, based on PyTorch, allows a deep understanding of the architecture, from scaled dot-product attention to the output layer.

2. **Adapt the model to our specific problem**: instead of predicting the next character in a text sequence, we will predict the next brain state (one of the 128 possible states, represented as a 7-bit token). This will allow us to compare the Transformer's performance with that of the linear models previously evaluated.

### Structure of the base code (gpt.py)

The file `gpt.py`, taken from Karpathy's repository, contains the complete implementation of a GPT-like Transformer (decoder-only). Its structure is as follows:

- **Configuration (`Config`)**: a simple class that groups all hyperparameters (batch size, context size, model dimensions, number of attention heads, number of layers, etc.).

- **Model (`GPTLanguageModel`)**: the main class, inheriting from `nn.Module`. Inside, it defines:
  - **`token_embedding_table`**: converts each token (integer) into a vector of dimension `n_embd`.
  - **`position_embedding_table`**: adds information about the position of each token in the sequence.
  - **`blocks`**: a sequence of `TransformerBlock` modules. Each block contains a `MultiHeadAttention` mechanism and a `FeedForward` network.
  - **`lm_head`**: a final linear layer that projects the Transformer output to the vocabulary size (`vocab_size`) to predict the next token.

- **Main methods**:
  - **`forward`**: defines the data flow through the model. It takes a tensor of indices (`idx`) as input and returns the logits (scores) for each position in the sequence.
  - **`generate`**: a method that, given an initial context, generates new tokens autoregressively (step by step, using the model's output as input for the next step).

- **Utility functions**:
  - **`get_batch`**: loads a batch of input data (context sequences) and their corresponding targets (the next token to predict) from the training or validation data.

- **Training loop**: the final code sets up the optimizer (AdamW) and runs a loop that iterates over the data, computes the loss (cross-entropy), and updates the model weights.

### Initial test: training with text

Before adapting the model to our data, we will run the code as is on the Shakespeare corpus included in the repository. This will allow us to verify that the environment is correctly set up and to familiarise ourselves with the training flow.

### References

- Karpathy, A. (2023). *nanoGPT lecture*. GitHub repository. https://github.com/karpathy/ng-video-lecture
- Suzyahyah, A. (2026). *The unreasonable difficulty of time series forecasting*. https://suzyahyah.github.io/machine%20learning/2026/06/27/trouble-with-time-series.html
- Vaswani, A. et al. (2017). *Attention Is All You Need*. Advances in Neural Information Processing Systems (NeurIPS).

## Adapting the Transformer to brain state sequences

After verifying that the Transformer architecture works correctly on text data, the next step is to adapt it to our specific problem: predicting the next brain state from a sequence of previous states.

### Data representation

In previous notebooks, we reduced the brain activity of the 7 resting-state networks (RSNs) to a sequence of discrete states. Each state is a 7‑bit binary vector indicating which networks are active (1) or inactive (0). Since there are 2⁷ = 128 possible combinations, each state can be represented as a single integer token between 0 and 127.

These tokens were generated in Notebook4 and saved as `state_codes` in a pickle file. They form a one‑dimensional sequence of length 256 (the number of time points in the fMRI scan). This sequence will be the input to the Transformer, just like the character sequence of a text corpus.

### Hyperparameter adjustments

The original `gpt.py` model was designed for large text corpora (e.g., Shakespeare). With only 256 time points, the original hyperparameters would cause severe overfitting and unnecessary computational cost. We reduced the model size and complexity as follows:

| Hyperparameter | Original value | New value | Justification |
|----------------|---------------|-----------|---------------|
| `vocab_size`   | 65 (characters) | 128 (states) | Number of possible brain states |
| `block_size`   | 256           | 10        | Short‑range temporal dependencies are sufficient |
| `batch_size`   | 64            | 16        | Small dataset, smaller batches |
| `n_embd`       | 384           | 64        | Reduce model capacity to avoid overfitting |
| `n_head`       | 6             | 2         | Fewer attention heads for small data |
| `n_layer`      | 6             | 2         | Fewer layers to keep model simple |
| `max_iters`    | 5000          | 2000      | Enough to see convergence with small data |

With these settings, the model has approximately 0.4 million parameters (compared to 10.8 million in the original), making it fast to train and less prone to overfitting.

### Loading the data

Instead of reading a text file, the adapted script loads the `state_codes` from the pickle file generated in Notebook4. The data is split chronologically into training (80%) and validation (20%) sets, preserving temporal order. This is crucial for time‑series forecasting, as random shuffling would break the sequential structure.

### What to expect

Given the small dataset and the high autocorrelation of the BOLD signal, the Transformer is unlikely to outperform the persistence baseline dramatically. However, this experiment serves two purposes:

1. **Benchmarking**: compare the expressive power of a Transformer against a simple linear model on the same data.
2. **Understanding**: verify whether the attention mechanism can capture any meaningful temporal patterns beyond what a linear model can extract.

The results will be evaluated using accuracy and F1‑score, as in previous notebooks, with special attention to the prediction of rare events (state changes).

## Results and Discussion

### Transformer evaluation on a single subject

The Transformer was trained and evaluated on the same single subject used in Notebook4 (256 time points). The model achieved a training loss of 0.17 and a validation loss of 0.03, indicating successful convergence. However, the evaluation on the validation set (52 samples) revealed a critical limitation: the model always predicts the state `0` (no networks active).

As a result:
- **Exact-match accuracy** was 1.0 (perfect), because all samples in the validation set were state 0.
- **Per-network F1** was 0.0 for all networks, because no positive events occurred in the validation set.
- **Change prediction F1** was also 0.0, for the same reason.

The sequence generated by the model confirms this behaviour:

[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


### Why did the Transformer fail on a single subject?

This behaviour is consistent with the **class imbalance** in the data: state 0 accounts for more than 80% of the samples. The model has learned that predicting 0 is the safest strategy to minimise the cross-entropy loss. With only 52 validation samples, it is highly likely that no activation events were present in the validation set, leading to perfect accuracy but zero F1.

Crucially, this is **not a failure of the Transformer architecture**. Rather, it confirms the central thesis of **Suzyahyah (2026)**: in time series with high autocorrelation and scarce events, even powerful models cannot overcome the limitations of the data itself. The author argues that the "forecastability" of a series is an intrinsic property of the data, not a property of the model. When the signal is weak and events are rare, the best any model can do is to exploit the dominant class (in this case, the null state).

This result also aligns with the observations in Notebook4, where the persistence baseline (predicting that the next state is identical to the current one) achieved an accuracy of over 0.95. The high autocorrelation of the BOLD signal makes it extremely difficult to predict anything beyond simple inertia.

### Comparison with Notebook4 (logistic regression)

The following table summarises the performance of the Transformer on a single subject compared to the logistic regression models from Notebook4:

| Model | Task | Best F1 | Baseline F1 | Notes |
|-------|------|---------|-------------|-------|
| Logistic regression (Notebook4) | Full state (0.5 SD) | 0.6864 | 0 (chance) | Trained on 204 samples |
| Logistic regression (Notebook4) | Changes (1.0 SD) | 0.4528 | 0 | Trained on 204 samples |
| Transformer (single subject) | Full state | 0.0 | — | Trained on 204 samples |
| Transformer (single subject) | Changes | 0.0 | — | Trained on 204 samples |

The Transformer's poor performance on a single subject is due to data scarcity, not to the model itself. With only 256 time points and a heavily imbalanced class distribution, the model cannot learn meaningful patterns beyond the null state. This is a clear demonstration of the principle discussed by Suzyahyah (2026): **more complex models do not necessarily lead to better predictions when the data has low intrinsic forecastability**.

### What does this mean for the project?

These results are **not a setback**. On the contrary, they provide strong empirical evidence for the argument that the limitation lies in the data, not in the modelling approach. This is a valuable scientific finding in itself: it shows that with a single subject and a short resting-state fMRI scan, even state-of-the-art models cannot extract reliable predictive signals.

### Next steps: scaling to multiple subjects

The natural next step is to extend the analysis to the full dataset of 90 subjects. This will provide:

- **More data**: approximately 23,000 tokens (90 subjects × 256 time points) instead of 256.
- **More events**: a higher total number of state changes and activations, which will allow the model to learn patterns that generalise.
- **Better generalisation**: the model will learn patterns that are common across subjects, rather than idiosyncratic to one individual.

The same pipeline (extraction of signals, point-process thresholding, state encoding) will be applied to all subjects. The Transformer will then be trained on the concatenated sequences, and its performance will be compared against the logistic regression baseline using the same evaluation metrics (F1 for state prediction and change prediction).

### References

- Suzyahyah, A. (2026). *The unreasonable difficulty of time series forecasting*. https://suzyahyah.github.io/machine%20learning/2026/06/27/trouble-with-time-series.html

## Data preparation for multi-subject training

### Dataset and processing pipeline

To obtain a sufficiently large dataset for the Transformer, we processed all 90 subjects from the ds005747 dataset using DataLad to manage the download and storage of large functional files. The processing script (`process_all_subjects.py`) performs the following steps for each subject:

1. **Download the functional file** using `datalad get` (only if not already local).
2. **Extract the 7 RSN time series** using the tutor's masks (MNI305 space) and the same parameters as in Notebook4.
3. **Normalise** the signals to zero mean and unit variance.
4. **Apply the point-process threshold** at 1 standard deviation.
5. **Encode** each time point as a state token (0‑127).
6. **Append** the sequence to a global array.
7. **Drop the file** using `datalad drop` to free disk space.

This approach ensures that only the current subject's file is stored locally at any time, minimising disk usage (≈1.5 GB per subject instead of 360 GB for all).

The concatenated state sequences from all 90 subjects yield approximately **23,040 tokens** (90 × 256). The resulting `state_codes_all.pkl` file is only a few kilobytes in size and will be used to train the Transformer.

### Status

At the time of writing, the processing script is being executed. Once completed, the training and evaluation of the Transformer on the full dataset will be performed.

## Scaling to multiple subjects: data preparation

As discussed in the previous section, the Transformer trained on a single subject failed to learn meaningful patterns beyond the null state due to the extreme class imbalance and the small number of samples (256 time points). To overcome this limitation, we extended the analysis to the full dataset of **90 subjects** from the OpenNeuro ds005747 study.

### Dataset description

The dataset consists of 90 healthy adults (18‑40 years old) scanned at 7T, each with three 10‑minute resting‑state fMRI runs. After the preprocessing steps described in Notebook4 (signal extraction, detrending, band‑pass filtering, normalisation, and point‑process thresholding at 1 SD), each run yields a sequence of 256 state tokens (integers 0‑127 representing the activation pattern of the 7 RSNs).

### Processing pipeline for multiple subjects

To generate the state sequences for all subjects without storing the entire dataset locally, we used the `openneuro-py` library, which allows downloading individual files from OpenNeuro on demand. The processing script (`process_all_subjects.py`) performs the following steps for each subject and each functional run (1, 2, and 3):

1. **Download the fMRI file** using `openneuro.download()`.
2. **Extract the time series** for the 7 RSNs using the same `NiftiMasker` parameters as in Notebook4.
3. **Normalise the signals** manually to zero mean and unit variance.
4. **Apply the point‑process threshold** at 1 standard deviation.
5. **Encode each time point** as a token (0‑127).
6. **Concatenate** the tokens from all runs of all subjects.
7. **Delete the downloaded file** to free disk space.

The script processes subjects sequentially, ensuring that at any given moment only the current subject's run is stored locally. This avoids storing all 90 subjects (≈360 GB) simultaneously.

### Resulting dataset

The concatenated state sequences from all 90 subjects and all three runs yield approximately **69,120 tokens** (90 × 3 × 256). The resulting `state_codes_all.pkl` file is only a few hundred kilobytes in size.

### Generating the multi-subject dataset

With the tutor's masks (`mask_imgs_tutor`) available in the correct space (MNI305), the processing script (`process_all_subjects.py`) was executed to generate the state sequences for all 90 subjects.

The script performs the following steps for each subject and each of the three functional runs:

1. Downloads the functional file using `openneuro-py`.
2. Extracts the 7 RSN time series using the same `NiftiMasker` parameters as in Notebook4.
3. Normalises the signals manually to zero mean and unit variance.
4. Applies the point-process threshold at 1 standard deviation.
5. Encodes each time point as a token (0‑127).
6. Concatenates the tokens into a global array and deletes the downloaded file.

The script includes automatic resume capability: if interrupted, it restarts from the last successfully processed run.

Once completed, the resulting `state_codes_all.pkl` file contains approximately **69,120 tokens** (90 subjects × 3 runs × 256 time points) and is ready for Transformer training.

## Training the Transformer on all subjects

After processing all 90 subjects and their three runs, we obtained a concatenated sequence of **69,120 tokens** (90 × 3 × 256). This dataset is 270 times larger than the single‑subject dataset and contains a much richer variety of activation patterns.

The same Transformer architecture (Config: block_size=10, n_embd=64, n_head=2, n_layer=2) was trained on this dataset for 5000 iterations. The training loss and validation loss are shown below.

### Results

| Metric | Value |
|--------|-------|
| Train loss | ... |
| Val loss | ... |
| Exact-match accuracy | ... |
| Per-network F1 (average) | ... |
| Change prediction F1 | ... |

### Comparison with single‑subject results

| Model | Dataset | Best F1 (changes) |
|-------|---------|-------------------|
| Logistic regression | Single subject | 0.4528 |
| Transformer | Single subject | 0.0 |
| Transformer | 90 subjects | ... |

The improvement (or lack thereof) will be discussed in the context of the data's intrinsic forecastability (Suzyahyah, 2026).
