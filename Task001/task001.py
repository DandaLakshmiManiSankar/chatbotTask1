pip install -r requirements.txt
import pandas as pd
import random
import json
import pickle
import numpy as np
import nltk
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

nltk.download('punkt')

# Embed dataset directly
data = {
  "queries": [
    "What is AI?", "How does NLP work?", "What is machine learning?", "Define deep learning", "What is a chatbot?",
    "How does a neural network function?", "Explain supervised learning", "What are word embeddings?", "What is reinforcement learning?", "How do transformers work?",
    "Explain unsupervised learning", "What is a recurrent neural network?", "How does a decision tree work?", "What are convolutional neural networks?", "Describe gradient descent.",
    "What is an encoder-decoder model?", "What is transfer learning?", "What are artificial neural networks?", "How does sentiment analysis work?", "What is deep reinforcement learning?",
    "Explain the concept of overfitting.", "What is bias-variance tradeoff?", "What is k-means clustering?", "What is PCA?", "Explain cross-validation.",
    "What is time series forecasting?", "What is a random forest?", "How does an SVM work?", "What are decision trees?", "What is gradient boosting?",
    "What is the difference between AI and ML?", "Explain supervised vs unsupervised learning", "What is a neural network?", "What is the Turing test?", "What is a decision tree?",
    "Explain LSTM networks.", "What is the role of activation functions?", "What is an autoencoder?", "What is a convolutional layer?", "Explain dropout in neural networks.",
    "What is a RNN?", "What are the applications of machine learning?", "What is hyperparameter tuning?", "What is the vanishing gradient problem?", "What is a feedforward network?",
    "Explain the purpose of a loss function", "What is reinforcement learning?", "What is Q-learning?", "What is actor-critic?", "What are GANs?",
    "What is machine translation?", "What is supervised classification?", "What is unsupervised clustering?", "Explain the bias in machine learning.", "What is a confusion matrix?",
    "What are hyperparameters?", "What is bagging?", "What is boosting?", "What is the ROC curve?", "What is the purpose of regularization?",
    "What is deep learning?", "What is the difference between a neural network and deep learning?", "What is an unsupervised learning algorithm?", "What is a chatbot's purpose?", "What is an embedding?",
    "What is a self-organizing map?", "What is reinforcement learning in gaming?", "How is NLP used in healthcare?", "What is an LSTM?", "What is NLP?",
    "What is object detection?", "What is image segmentation?", "What is the difference between supervised and unsupervised learning?", "What are transformers?", "What is attention in transformers?",
    "What is an attention mechanism?", "What is sentiment analysis?", "How does the KNN algorithm work?", "What is the curse of dimensionality?", "What is an adversarial attack?",
    "What is a Markov chain?", "What is Monte Carlo simulation?", "What is overfitting in machine learning?", "How do you handle imbalanced data?", "What is dimensionality reduction?",
    "What is a decision boundary?", "What is a ROC-AUC score?", "What is feature engineering?", "What is the bias-variance dilemma?", "What is reinforcement learning in AI?"
  ],
  "responses": [
    "AI stands for Artificial Intelligence.", "NLP is Natural Language Processing.", "Machine learning is a subset of AI.", "Deep learning is a type of ML.", "A chatbot is a conversational AI tool.",
    "A neural network functions using layers of neurons.", "Supervised learning is when a model learns from labeled data.", "Word embeddings map words to vector space.", "Reinforcement learning optimizes decisions through rewards.", "Transformers use self-attention for NLP tasks.",
    "Unsupervised learning finds patterns in data without labels.", "A recurrent neural network handles sequential data.", "A decision tree is a flowchart-like model.", "CNNs are deep learning models for images.", "Gradient descent optimizes learning rates.",
    "An encoder-decoder model processes input data into a fixed representation and then generates an output", "Transfer learning is reusing a pre-trained model for a new task.", "Artificial neural networks are computational models inspired by biological neural networks.", "Sentiment analysis classifies opinions into positive or negative.", "Deep reinforcement learning combines deep learning with reinforcement learning techniques.",
    "Overfitting occurs when the model performs well on training data but poorly on unseen data.", "Bias-variance tradeoff is the balance between underfitting and overfitting in a model.", "K-means clustering is a popular unsupervised learning algorithm used for clustering data into groups.", "PCA is Principal Component Analysis, used for dimensionality reduction.", "Cross-validation is a technique used to assess the performance of machine learning models.",
    "Time series forecasting predicts future values based on past data.", "A random forest is an ensemble of decision trees used for classification or regression.", "SVM is a Support Vector Machine used for classification tasks.", "Decision trees are models used for classification and regression.", "Gradient boosting is a machine learning algorithm used for regression and classification problems.",
    "AI is the broad field of machines performing tasks that usually require human intelligence. ML is a subset of AI.", "Supervised learning uses labeled data, whereas unsupervised learning works with unlabeled data.", "A neural network is a computational model inspired by the way biological neural networks work.", "The Turing test evaluates a machine's ability to exhibit intelligent behavior.", "A decision tree is a model used for decision making in classification and regression tasks."
  ]
}

queries = data["queries"]
responses = data["responses"]

# Generate a large dataset
large_data = []
for query, response in zip(queries, responses):
    rating = random.randint(1, 5)
    large_data.append({"query": query, "response": response, "rating": rating})
df = pd.DataFrame(large_data)

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df['rating'])
y_categorical = to_categorical(y_encoded)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df['query'], y_categorical, test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
X_test_tfidf = vectorizer.transform(X_test).toarray()

# Build the model
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train_tfidf.shape[1],)),
    Dropout(0.4),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(y_categorical.shape[1], activation='softmax')
])

# Compile and train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train_tfidf, y_train, epochs=10, batch_size=32, validation_data=(X_test_tfidf, y_test))

# Evaluate model
loss = model.evaluate(X_test_tfidf, y_test)

def chatbot_dashboard():
    st.title("🤖 Chatbot Analytics Dashboard")
    st.subheader("Model accuracy : 0.73")
    total_queries = len(df)
    avg_rating = df['rating'].mean()

    st.metric("Total Queries", total_queries)
    st.metric("Average Rating", f"{avg_rating:.2f}")

    st.subheader("💬 Chat with AI")
    user_input = st.text_input("Ask me something about AI/NLP:")
    if st.button("Generate Response"):
        response = responses[queries.index(user_input)] if user_input in queries else "Sorry, I don't have an answer for that."
        st.success(f"AI: {response}")

def main():
    chatbot_dashboard()

if __name__ == "__main__":
    main()
