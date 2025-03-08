# **N-Gram Language Model**

## **Overview**
This project implements an **N-Gram Language Model** with support for **unigram and bigram modeling**, incorporating **smoothing techniques** such as **Laplace smoothing** and **Add-k smoothing** to improve generalization on unseen data. The model is trained on a given text corpus and evaluated using **perplexity** to measure performance.

## **Features**
- Supports **unigram and bigram** models.
- Implements **Laplace and Add-k smoothing** for better generalization.
- Computes **perplexity** to evaluate model performance on training and validation datasets.
- Displays **probability distributions** for the trained model.

## **Perplexity Analysis**
- **Unsmoothed Bigram Model** severely overfits training data, leading to **extremely high perplexity** on validation data.
- **Laplace Smoothing (Unigram)** improves generalization, lowering validation perplexity.
- **Add-k Smoothing (Bigram, k=0.5)** significantly reduces validation perplexity, making bigram models more viable.

## **Usage**
1. Place your training data in `train.txt` and validation data in `val.txt`.
2. Run the script:
   ```bash
   python ngram_model.py
