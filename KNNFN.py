import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, \
    f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tfidf_matrix):
        self.texts = texts
        self.labels = labels
        self.tfidf_matrix = tfidf_matrix

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        tfidf_vector = self.tfidf_matrix[idx].toarray().flatten()
        return {'text': text, 'label': label, 'tfidf_vector': torch.tensor(tfidf_vector, dtype=torch.float32)}


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def preprocess_data(df):

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()


    def clean_text(text):

        text = text.lower()

        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text


    df['cleaned_text'] = df['text'].apply(clean_text)


    vectorizer = TfidfVectorizer(max_features=5000)
    X_tfidf = vectorizer.fit_transform(df['cleaned_text'])

    return X_tfidf


df = pd.read_csv('fake_news_dataset.csv')
X_tfidf = preprocess_data(df)
y = df['label'].values


X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)


train_dataset = FakeNewsDataset(X_train, y_train, X_train)
test_dataset = FakeNewsDataset(X_test, y_test, X_test)


class FakeNewsKNN:
    def __init__(self, n_neighbors=5):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]



knn_model = FakeNewsKNN(n_neighbors=5)
knn_model.train(X_train, y_train)


all_preds = knn_model.predict(X_test)
all_probs = knn_model.predict_proba(X_test)


def calculate_metrics(labels, preds):
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    auc = roc_auc_score(labels, all_probs)

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-Score: {f1:.4f}')
    print(f'AUC: {auc:.4f}')


def plot_confusion_matrix(labels, preds):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()




calculate_metrics(y_test, all_preds)
plot_confusion_matrix(y_test, all_preds)
