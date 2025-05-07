import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Dataset
data = {
    'question': [
        'hello', 'hi', 'how are you', 'what is your name', 'bye', 'thank you',
        'how can you help me', 'tell me a joke', 'what is the time',
        'what is today\'s date', 'what is AI', 'explain data science',
        'I want to book a ticket', 'it\'s raining heavily', 'Tell me a funny joke',
        'What is Machine Learning?', 'good morning', 'good night', 'who created you',
        'can you help me', 'tell me weather', 'book hotel room', 'explain ML', 
        'what\'s the meaning of AI', 'give me a joke', 'thank you so much'
    ],
    'intent': [
        'greet', 'greet', 'feeling', 'identity', 'bye', 'thanks',
        'help', 'joke', 'time',
        'date', 'ai', 'datascience',
        'booking', 'weather', 'joke',
        'machinelearning', 'greet', 'bye', 'identity',
        'help', 'weather', 'booking', 'machinelearning',
        'ai', 'joke', 'thanks'
    ]
}

df = pd.DataFrame(data)

# TF-IDF vectorizer
vectorizer = TfidfVectorizer(lowercase=True, stop_words=None)
X = vectorizer.fit_transform(df['question'])

# Encode the labels
le = LabelEncoder()
y = le.fit_transform(df['intent'])

# Train model
model = LogisticRegression()
model.fit(X, y)

# Response mapping
responses = {
    'greet': "Hello! 👋",
    'feeling': "I'm doing well, thank you!",
    'identity': "I'm your AI chatbot, built using data science!",
    'bye': "Goodbye! 👋",
    'thanks': "You're welcome! 🙏",
    'help': "I'm here to help you. Ask me anything.",
    'joke': "Why did the computer go to therapy? It had too many bugs! 😂",
    'time': "Sorry, I can't tell the time yet!",
    'date': "I'm not connected to a calendar right now.",
    'ai': "AI stands for Artificial Intelligence.",
    'datascience': "Data Science involves extracting insights from data.",
    'booking': "Sure, I can help with bookings!",
    'weather': "I can't give live weather updates yet, sorry!",
    'machinelearning': "Machine Learning is a subset of AI that learns from data."
}

# Context memory
class ContextMemory:
    def __init__(self):
        self.history = []

    def add(self, msg):
        self.history.append(msg)
        if len(self.history) > 5:
            self.history.pop(0)

    def get_last(self):
        return self.history[-1] if self.history else None

context_memory = ContextMemory()

# Chat function
def chat():
    print("Chatbot: Hello! How can I help you? (Type 'bye' to exit)")
    while True:
        user_input = input("You: ").strip().lower()
        context_memory.add(user_input)

        if 'bye' in user_input:
            print("Chatbot:", responses['bye'])
            break

        # Vectorize user input
        X_test = vectorizer.transform([user_input])

        # Predict intent
        try:
            y_pred = model.predict(X_test)
            intent = le.inverse_transform(y_pred)[0]
            print("Chatbot:", responses.get(intent, "Sorry, I didn’t understand that."))
        except Exception:
            print("Chatbot: Sorry, I didn’t understand that.")

# Run the chatbot
if __name__ == "__main__":
    chat()

