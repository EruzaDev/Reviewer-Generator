import joblib
import pandas as pd

# Load model and vectorizer
model = joblib.load("quiz_classifier_model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

# Test sentences
test_sentences = [
    "The OSI model consists of seven layers.",
    "This sentence is just a random thought.",
    "TCP and UDP are transport layer protocols.",
    "How are you doing",
    "The sun is really hot"
]

# Transform and predict
X_test = vectorizer.transform(test_sentences)
predictions = model.predict(X_test)

# Show results
for sentence, pred in zip(test_sentences, predictions):
    label = "Quiz-Worthy ✅" if pred == 1 else "Not Quiz-Worthy ❌"
    print(f"{label}: {sentence}")
