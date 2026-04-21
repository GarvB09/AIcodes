import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load dataset
data = pd.read_csv("credit_data.csv")

# 2. Preprocess data
data = data.fillna(method='ffill')   # handle missing values
data = pd.get_dummies(data)          # convert categorical to numeric

# 3. Separate features and label
X = data.drop("loan_approved", axis=1)
y = data["loan_approved"]

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Normalize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Model creation
model = MLPClassifier(
    hidden_layer_sizes=(10, 10),
    activation='relu',
    max_iter=200
)

# 7. Training
model.fit(X_train, y_train)

# 8. Prediction
y_pred = model.predict(X_test)

# 9. Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
