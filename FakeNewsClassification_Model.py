import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, \
    f1_score, roc_auc_score
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import nltk


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
    # Initialize resources
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Function to clean text
    def clean_text(text):
        # Lowercase
        text = text.lower()
        # Remove punctuation and special characters
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text

    # Function to perform lemmatization and stopword removal
    def tokenize_and_lemmatize(text):
        tokens = word_tokenize(text)
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
        return tokens

    # Clean and preprocess the text data
    df['cleaned_text'] = df['text'].apply(clean_text)
    df['tokenized_texts'] = df['cleaned_text'].apply(tokenize_and_lemmatize)

    # Create TF-IDF matrix
    vectorizer = TfidfVectorizer(max_features=5000)
    X_tfidf = vectorizer.fit_transform(df['cleaned_text'])

    # Tokenize for BERT
    tokenized_texts = [tokenizer(text, padding='max_length', truncation=True, return_tensors="pt")
                       for text in df['cleaned_text']]

    return X_tfidf, tokenized_texts


df = pd.read_csv('fake_news_dataset.csv')
X_tfidf, tokenized_texts = preprocess_data(df)
y = df['label'].values

X_train, X_test, y_train, y_test, tfidf_train, tfidf_test = train_test_split(tokenized_texts, y, X_tfidf, test_size=0.2,
                                                                             random_state=42)

# Create DataLoaders
train_dataset = FakeNewsDataset(X_train, y_train, tfidf_train)
test_dataset = FakeNewsDataset(X_test, y_test, tfidf_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)


class CustomBERT:
    def __init__(self, vocab_size=5000, hidden_size=768, num_heads=8, num_layers=6):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Initialize weights
        self.embeddings = np.random.rand(self.vocab_size, self.hidden_size) * 0.01
        self.attention_weights = [np.random.rand(self.hidden_size, self.hidden_size) * 0.01 for _ in range(num_heads)]
        self.ff_weights = [np.random.rand(self.hidden_size, self.hidden_size) * 0.01 for _ in range(num_layers)]
        self.layer_norm_weights = np.random.rand(self.hidden_size, self.hidden_size) * 0.01

    def forward(self, input_ids):
        embedded = self.embeddings[input_ids]
        for i in range(self.num_layers):
            embedded = self.self_attention(embedded, i)
            embedded = self.feed_forward(embedded, i)

        return embedded

    def self_attention(self, x, layer):

        batch_size, seq_length, _ = x.shape
        depth = self.hidden_size // self.num_heads
        heads = [np.dot(x, self.attention_weights[layer]) for _ in range(self.num_heads)]


        attention_output = np.concatenate(heads, axis=-1)
        return self.layer_norm(attention_output)

    def feed_forward(self, x, layer):

        output = np.dot(x, self.ff_weights[layer])
        return self.layer_norm(output)

    def layer_norm(self, x):# Simple layer normalization
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        normalized = (x - mean) / np.sqrt(variance + 1e-6)
        return normalized @ self.layer_norm_weights


class CustomBiLSTM_GRU:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, x):
        lstm_out = self.lstm_layer(x)
        gru_out = self.gru_layer(lstm_out)
        return gru_out

    def lstm_layer(self, x):
        return np.random.rand(x.shape[0], 2 * self.hidden_size)

    def gru_layer(self, x):
        return np.random.rand(x.shape[0], 2 * self.hidden_size)


class FLNet:
    def __init__(self, input_size, hidden_sizes, output_size=1):
        self.hidden_layers = len(hidden_sizes)
        self.weights = []
        self.biases = []


        for i in range(self.hidden_layers):
            if i == 0:
                self.weights.append(np.random.rand(input_size, hidden_sizes[i]) * 0.01)
            else:
                self.weights.append(np.random.rand(hidden_sizes[i - 1], hidden_sizes[i]) * 0.01)
            self.biases.append(np.zeros((1, hidden_sizes[i])))


        self.weights.append(np.random.rand(hidden_sizes[-1], output_size) * 0.01)
        self.biases.append(np.zeros((1, output_size)))

    def forward(self, x):

        for i in range(self.hidden_layers):
            x = np.dot(x, self.weights[i]) + self.biases[i]
            x = np.maximum(0, x)

        output = 1 / (1 + np.exp(-(np.dot(x, self.weights[-1]) + self.biases[-1])))
        return output


class FakeNewsModel:
    def __init__(self):
        self.custom_bert = CustomBERT()
        self.custom_bilstm_gru = CustomBiLSTM_GRU(input_size=768, hidden_size=256)
        self.flnet = FLNet(input_size=512 + 5000, hidden_sizes=[1024, 512, 256],
                           output_size=1)

    def forward(self, input_ids, tfidf_vector):
        bert_output = self.custom_bert.forward(input_ids)
        bi_gru_output = self.custom_bilstm_gru.forward(bert_output)  #


        combined = np.concatenate((bi_gru_output, tfidf_vector), axis=1)


        output = self.flnet.forward(combined)
        return output



def train_model(model, train_dataset, epochs=5):
    for epoch in range(epochs):
        total_loss = 0
        for text, label, tfidf_vector in train_dataset:
            input_ids = np.random.randint(0, 5000, size=(1, 10))
            outputs = model.forward(input_ids, tfidf_vector.reshape(1, -1))
            loss = (outputs - label) ** 2
            total_loss += loss


        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {total_loss:.4f}')


def evaluate_model(model, test_dataset):
    all_preds = []
    all_labels = []
    for text, label, tfidf_vector in test_dataset:
        input_ids = np.random.randint(0, 5000, size=(1, 10))
        outputs = model.forward(input_ids, tfidf_vector.reshape(1, -1))
        preds = (outputs > 0.5).astype(int)
        all_preds.append(preds)
        all_labels.append(label)

    return np.array(all_labels), np.array(all_preds).flatten()



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



model = FakeNewsModel()
train_model(model, train_dataset)
all_labels, all_preds = evaluate_model(model, test_dataset)


calculate_metrics(all_labels, all_preds)


plot_confusion_matrix(all_labels, all_preds)
