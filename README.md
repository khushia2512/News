# Predicting the Source of a News Headline

Given a news headline, predict which publisher wrote it, using only the writing style and vocabulary of the headline.

Dataset: [UCI News Aggregator (Kaggle)](https://www.kaggle.com/datasets/uciml/news-aggregator-dataset), 422,419 headlines. Download `uci-news-aggregator.csv` and place it next to the notebooks.

## Notebooks

| File | What it does |
|---|---|
| `MAIN FILE.ipynb` | First version: Doc2Vec (PV-DBOW) with one vector per publisher, prediction by cosine similarity |
| `News_Source_Evaluation.ipynb` | Proper evaluation: train/test split, Doc2Vec + ANN, baselines, accuracy/precision/recall/F1 |

## Data

Publishers with 2,000 to 3,000 headlines were kept, giving 6 balanced classes and 13,751 headlines:
Huffington Post, Businessweek, Contactmusic.com, Daily Mail, NASDAQ, Examiner.com.

Text is lowercased, punctuation removed and split into words. The data is split 80/20 (stratified): 11,000 train, 2,751 test.

## Methods

1. **Doc2Vec + cosine similarity** (original approach): each headline is tagged with its publisher, so Doc2Vec learns one vector per publisher. A new headline is converted with `infer_vector` and assigned to the most similar publisher vector.
2. **Doc2Vec + Logistic Regression**: each headline gets its own 100-dimension PV-DBOW vector, then a linear classifier.
3. **Doc2Vec + ANN**: same vectors, fed to a neural network (128 → 64 hidden units, ReLU, Adam, early stopping, softmax output over 6 publishers).
4. **TF-IDF + Logistic Regression**: word and word-pair (1–2 gram) TF-IDF features, used as a baseline.

## Results (held-out test set, 2,751 headlines)

| Method | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| Doc2Vec + cosine similarity | 0.526 | 0.543 | 0.527 | 0.497 |
| Doc2Vec + Logistic Regression | 0.501 | 0.502 | 0.499 | 0.498 |
| Doc2Vec + ANN | 0.517 | 0.527 | 0.514 | 0.514 |
| TF-IDF + Logistic Regression | **0.629** | **0.630** | **0.629** | **0.629** |

Random guessing over 6 balanced classes gives about 0.167.

## Findings

- Finance publishers are easiest to identify (NASDAQ F1 0.76, Businessweek 0.63) because of specific vocabulary such as "forex", "stocks" and "fed".
- General news publishers are hardest (Huffington Post 0.40, Examiner.com 0.36) because they cover every topic and overlap with the others.
- On very short texts like headlines, TF-IDF keyword features beat Doc2Vec embeddings: exact words and names matter more than overall meaning.

## Possible improvements

- Fine-tune a transformer model (e.g. DistilBERT) on the headlines
- Use the full article text instead of only the headline
- Tune Doc2Vec (vector size, epochs, PV-DM vs PV-DBOW) and the ANN with cross-validation
- Combine TF-IDF and Doc2Vec features

## Tech

Python, pandas, Gensim (Doc2Vec), scikit-learn (MLPClassifier, LogisticRegression, TF-IDF)
