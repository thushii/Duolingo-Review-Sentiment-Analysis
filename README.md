# Duolingo Review Sentiment Analysis

This project analyzes real user reviews from the Duolingo mobile app to understand how people feel about the platform. The goal was to classify reviews as Positive or Negative using NLP techniques and compare different machine learning models.

## What I Did

1. Cleaned and prepared raw text reviews  
2. Converted ratings into sentiment labels  
3. Applied NLP preprocessing  
4. Trained multiple ML models  
5. Evaluated performance with proper metrics  
6. Visualized results with WordClouds and charts

## Data Processing Steps

- Converted text to lowercase  
- Removed punctuation and numbers  
- Tokenized sentences  
- Removed stopwords  
- Lemmatized words using POS tagging  
- Converted text to TF-IDF vectors

## Models Compared

- Naive Bayes  
- Logistic Regression  
- Support Vector Machine (SVM)

## Results Summary

| Model | Accuracy | Precision | Recall | F1 |
|------|----------|-----------|--------|----|
| Naive Bayes | 0.92 | 0.91 | 0.54 | 0.55 |
| Logistic Regression | 0.94 | 0.90 | 0.68 | 0.74 |
| SVM | 0.94 | 0.93 | 0.72 | 0.78 |

SVM gave the most balanced performance and handled negative reviews better than the other models.

## What I Learned

- Real app reviews are highly imbalanced toward positive ratings  
- Simple models like Naive Bayes struggle with minority negative class  
- Preprocessing quality directly affects performance  
- TF-IDF with SVM works well for short review text

## Technologies

- Python  
- NLTK  
- Scikit-Learn  
- Pandas & NumPy  
- Matplotlib & Seaborn  
- WordCloud

## How to Run

pip install -r requirements.txt  
python sentiment_analysis.py

---

Created as part of my learning in NLP and Machine Learning.
