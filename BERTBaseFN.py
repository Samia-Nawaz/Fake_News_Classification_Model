import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, \
    f1_score, roc_auc_score
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import BertTokenizer, BertModel


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

    # Function to clean text
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


class BERTModel(nn.Module):
    def __init__(self):
        super(BERTModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state[:, -1, :]
        out = self.fc(hidden_state)
        return self.sigmoid(out)



model = BERTModel()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train_model(model, train_dataset, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data in train_dataset:
            input_ids = data['tfidf_vector']
            labels = data['label'].float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {total_loss:.4f}')


def evaluate_model(model, test_dataset):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in test_dataset:
            input_ids = data['tfidf_vector']
            labels = data['label']
            outputs = model(input_ids).squeeze()
            preds = (outputs > 0.5).int()
            all_preds.append(preds)
            all_labels.append(labels)

    return np.concatenate(all_labels), np.concatenate(all_preds)


def calculate_metrics(labels, preds):
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    auc = roc_auc_score(labels, preds)

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



train_model(model, train_dataset)


all_labels, all_preds = evaluate_model(model, test_dataset)


calculate_metrics(all_labels, all_preds)
plot_confusion_matrix(all_labels, all_preds)
