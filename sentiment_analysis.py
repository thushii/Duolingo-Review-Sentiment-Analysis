import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from collections import Counter

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab')


# Load dataset
df = pd.read_csv('duolingoreviews.csv', low_memory=False)

df = df[['review_description', 'rating']].dropna()

df['label'] = df['rating'].apply(lambda x: 1 if x >= 4 else (0 if x <= 2 else None))
df.dropna(subset=['label'], inplace=True)

df.rename(columns={'review_description': 'review'}, inplace=True)


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_tokens(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))

    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]

    pos_tags = nltk.pos_tag(tokens)

    lemmatized = []
    for word, tag in pos_tags:
        pos = tag[0].lower()
        pos = {'j': 'a', 'n': 'n', 'v': 'v', 'r': 'r'}.get(pos, 'n')
        lemma = lemmatizer.lemmatize(word, pos)
        lemmatized.append(lemma)

    return lemmatized


df['clean_tokens'] = df['review'].apply(preprocess_tokens)
df['clean_review'] = df['clean_tokens'].apply(lambda tokens: " ".join(tokens))


X_train, X_test, y_train, y_test = train_test_split(
    df['clean_review'],
    df['label'],
    test_size=0.3,
    random_state=42,
    stratify=df['label']
)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": LinearSVC()
}

results = []

for name, model in models.items():
    print(f"\n{name} Results:")

    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])

    plt.title(f'{name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-score": f1_score(y_test, y_pred, zero_division=0)
    })


positive_text = " ".join(df[df['label'] == 1]['clean_review'])
negative_text = " ".join(df[df['label'] == 0]['clean_review'])

positive_wc = WordCloud(width=800, height=400, background_color='white').generate(positive_text)
negative_wc = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(negative_text)

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.imshow(positive_wc, interpolation='bilinear')
plt.axis('off')
plt.title('Positive Reviews WordCloud')

plt.subplot(1, 2, 2)
plt.imshow(negative_wc, interpolation='bilinear')
plt.axis('off')
plt.title('Negative Reviews WordCloud')

plt.tight_layout()
plt.show()


positive_words = [word for tokens in df[df['label'] == 1]['clean_tokens'] for word in tokens]
negative_words = [word for tokens in df[df['label'] == 0]['clean_tokens'] for word in tokens]

positive_counts = Counter(positive_words).most_common(20)
negative_counts = Counter(negative_words).most_common(20)

top_words = list(set([word for word, _ in positive_counts] +
                     [word for word, _ in negative_counts]))

pos_dict = dict(positive_counts)
neg_dict = dict(negative_counts)

comparison_data = {
    'Word': [],
    'Positive': [],
    'Negative': []
}

for word in top_words:
    comparison_data['Word'].append(word)
    comparison_data['Positive'].append(pos_dict.get(word, 0))
    comparison_data['Negative'].append(neg_dict.get(word, 0))

comparison_df = pd.DataFrame(comparison_data)
comparison_df['Total'] = comparison_df['Positive'] + comparison_df['Negative']
comparison_df = comparison_df.sort_values(by='Total', ascending=False).drop(columns='Total')


plt.figure(figsize=(14, 7))
x = np.arange(len(comparison_df['Word']))
width = 0.4

plt.bar(x - width/2, comparison_df['Positive'], width=width, label='Positive', color='green')
plt.bar(x + width/2, comparison_df['Negative'], width=width, label='Negative', color='red')

plt.xticks(ticks=x, labels=comparison_df['Word'], rotation=45, ha='right')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Word Frequency in Positive vs Negative Reviews')
plt.legend()
plt.tight_layout()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()


results_df = pd.DataFrame(results)
results_df.set_index('Model', inplace=True)

results_df.plot(kind='bar', figsize=(10, 6))
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
