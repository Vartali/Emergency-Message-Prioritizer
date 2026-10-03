from datasets import load_dataset
from collections import Counter

# Load HumAID dataset
dataset = load_dataset(
    "parquet",
    data_files={
        "train": "https://huggingface.co/datasets/QCRI/HumAID-all/resolve/main/data/train-00000-of-00001.parquet",
        "validation": "https://huggingface.co/datasets/QCRI/HumAID-all/resolve/main/data/validation-00000-of-00001.parquet",
        "test": "https://huggingface.co/datasets/QCRI/HumAID-all/resolve/main/data/test-00000-of-00001.parquet"
    }
)

# Our four target categories
mapping = {
    "injured_or_dead_people": "Casualties",
    "infrastructure_and_utility_damage": "Infrastructure Damage",
    "requests_or_urgent_needs": "Rescue Request",
    "not_humanitarian": "Not Relevant"
}


def prepare_data(split):
    data = []

    for row in dataset[split]:

        original_label = row["class_label"]

        if original_label in mapping:
            data.append({
                "text": row["tweet_text"],
                "label": mapping[original_label]
            })

    return data


# Prepare all three splits
train_data = prepare_data("train")
validation_data = prepare_data("validation")
test_data = prepare_data("test")


# Display sizes
print("Training samples:", len(train_data))
print("Validation samples:", len(validation_data))
print("Test samples:", len(test_data))


# Display category counts
print("\nTraining category counts:")

counts = Counter(row["label"] for row in train_data)

for label, count in counts.items():
    print(label, ":", count)

from sklearn.feature_extraction.text import TfidfVectorizer

# Separate text and labels
X_train_text = [row["text"] for row in train_data]
y_train = [row["label"] for row in train_data]

X_validation_text = [row["text"] for row in validation_data]
y_validation = [row["label"] for row in validation_data]

X_test_text = [row["text"] for row in test_data]
y_test = [row["label"] for row in test_data]


# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2)
)


# Learn vocabulary ONLY from training data
X_train = vectorizer.fit_transform(X_train_text)

# Transform validation and test using the same vocabulary
X_validation = vectorizer.transform(X_validation_text)
X_test = vectorizer.transform(X_test_text)


print("\nTF-IDF shapes:")
print("Training:", X_train.shape)
print("Validation:", X_validation.shape)
print("Test:", X_test.shape)

from sklearn.linear_model import LogisticRegression

# Create the model
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

# Train the model
model.fit(X_train, y_train)

print("\nModel training completed!")

from sklearn.metrics import accuracy_score, classification_report

# Make predictions on validation data
y_validation_pred = model.predict(X_validation)

# Calculate accuracy
accuracy = accuracy_score(y_validation, y_validation_pred)

print("\nValidation Accuracy:", accuracy)

# Detailed evaluation
print("\nClassification Report:")
print(classification_report(y_validation, y_validation_pred))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(
    y_validation,
    y_validation_pred,
    labels=[
        "Casualties",
        "Infrastructure Damage",
        "Rescue Request",
        "Not Relevant"
    ]
)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Casualties",
        "Infrastructure Damage",
        "Rescue Request",
        "Not Relevant"
    ]
)

disp.plot(xticks_rotation=45)
plt.tight_layout()
plt.show()

from sklearn.preprocessing import LabelEncoder

# Convert category names into numbers
label_encoder = LabelEncoder()

y_train_encoded = label_encoder.fit_transform(y_train)
y_validation_encoded = label_encoder.transform(y_validation)
y_test_encoded = label_encoder.transform(y_test)

print("\nLabel mapping:")

for number, label in enumerate(label_encoder.classes_):
    print(number, "=", label)

from sklearn.decomposition import TruncatedSVD

# Reduce TF-IDF features from 10,000 to 300
svd = TruncatedSVD(n_components=300, random_state=42)

X_train_nn = svd.fit_transform(X_train)
X_validation_nn = svd.transform(X_validation)
X_test_nn = svd.transform(X_test)

print("\nNeural Network input shapes:")
print("Training:", X_train_nn.shape)
print("Validation:", X_validation_nn.shape)
print("Test:", X_test_nn.shape)

import tensorflow as tf

Sequential = tf.keras.Sequential
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout

# Create neural network
nn_model = Sequential([
    Dense(128, activation="relu", input_shape=(300,)),
    Dropout(0.3),

    Dense(64, activation="relu"),
    Dropout(0.3),

    Dense(4, activation="softmax")
])

# Configure the model
nn_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Display architecture
nn_model.summary()

# Display architecture
nn_model.summary()


# Train the neural network
history = nn_model.fit(
    X_train_nn,
    y_train_encoded,
    validation_data=(X_validation_nn, y_validation_encoded),
    epochs=15,
    batch_size=32,
    verbose=1
)

from sklearn.metrics import accuracy_score, classification_report

# Predict validation data
y_nn_prob = nn_model.predict(X_validation_nn)

# Convert probabilities to predicted class numbers
y_nn_pred = y_nn_prob.argmax(axis=1)

# Accuracy
nn_accuracy = accuracy_score(
    y_validation_encoded,
    y_nn_pred
)

print("\nNeural Network Validation Accuracy:", nn_accuracy)

# Classification report
print("\nNeural Network Classification Report:")
print(
    classification_report(
        y_validation_encoded,
        y_nn_pred,
        target_names=label_encoder.classes_
    )
)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

nn_cm = confusion_matrix(
    y_validation_encoded,
    y_nn_pred
)

print("\nNeural Network Confusion Matrix:")
print(nn_cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=nn_cm,
    display_labels=label_encoder.classes_
)

disp.plot(xticks_rotation=45)
plt.tight_layout()
plt.show()

# ==============================
# FINAL TEST EVALUATION
# ==============================

from sklearn.metrics import accuracy_score, classification_report

# Make predictions on the test set
y_test_pred = model.predict(X_test)

# Calculate test accuracy
test_accuracy = accuracy_score(y_test, y_test_pred)

print("\n==============================")
print("FINAL TEST RESULTS")
print("==============================")

print("\nTest Accuracy:", test_accuracy)

print("\nTest Classification Report:")
print(
    classification_report(
        y_test,
        y_test_pred
    )
)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Final test confusion matrix
test_cm = confusion_matrix(
    y_test,
    y_test_pred,
    labels=[
        "Casualties",
        "Infrastructure Damage",
        "Rescue Request",
        "Not Relevant"
    ]
)

print("\nFinal Test Confusion Matrix:")
print(test_cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=test_cm,
    display_labels=[
        "Casualties",
        "Infrastructure Damage",
        "Rescue Request",
        "Not Relevant"
    ]
)

disp.plot(xticks_rotation=45)
plt.tight_layout()
plt.show()

import joblib

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\nModel and vectorizer saved successfully!")