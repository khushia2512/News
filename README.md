# Predicting the Source of a News Article (Doc2Vec + ANN)

Given a news article's title, predict which publisher wrote it, using only its words and writing style.

## Dataset

[UCI News Aggregator (Kaggle)](https://www.kaggle.com/datasets/uciml/news-aggregator-dataset): over 400,000 news articles collected from many publishers, with the title, URL, publisher, category and timestamp of each article. Publishers with a similar number of articles are selected so the classes are balanced.

## How it works

1. **Data selection:** keep the article title and publisher. The URL is dropped because it contains the publisher name.
2. **Cleaning:** lowercase, remove punctuation, split into words.
3. **Split:** 80% train / 20% test, stratified by publisher.
4. **Doc2Vec (Gensim):** each article title becomes a 100-number vector (PV-DBOW with word training, 40 epochs).
5. **ANN (scikit-learn MLPClassifier):** two hidden layers (256 and 128 neurons), ReLU, Adam optimizer, early stopping, softmax output over the publishers.
6. **Evaluation:** accuracy, precision, recall, F1, false alarm rate, ROC-AUC, PR-AUC, top-2 accuracy and confusion matrix on the unseen test set.

## Results (test set)

| Metric | Score |
|---|---|
| Accuracy | 0.572 |
| Precision (macro) | 0.572 |
| Recall (macro) | 0.570 |
| F1 (macro) | 0.565 |
| False alarm rate (macro) | 0.086 |
| ROC-AUC (macro, one-vs-rest) | 0.860 |
| PR-AUC (macro) | 0.603 |
| Top-2 accuracy | 0.765 |

- Finance publishers are easiest to identify: NASDAQ reaches F1 0.80 and ROC-AUC 0.97, because of specific words like "forex", "stocks" and "fed".
- General news sites are hardest because they write about every topic.

## Run it

```bash
pip install gensim scikit-learn pandas numpy
```
Download `uci-news-aggregator.csv` from Kaggle, put it next to the notebook, and run `News_Source_Doc2Vec_ANN.ipynb`.

## Next steps

- Use the full article text instead of only titles
- Try a transformer model such as DistilBERT
