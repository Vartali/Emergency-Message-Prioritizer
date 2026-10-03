import joblib

# Load the trained model and TF-IDF vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Take a new message from the user
message = input("Enter an emergency message: ")

# Convert the message into TF-IDF features
message_vector = vectorizer.transform([message])

# Predict the category
prediction = model.predict(message_vector)[0]

print("\nPredicted Category:", prediction)