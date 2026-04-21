import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Dummy dataset
data = pd.DataFrame({
    "income": [50, 60, 70, 80, 90],
    "age": [25, 35, 45, 20, 30],
    "loan_approved": [0, 1, 1, 0, 1]
})

# Features & label
X = data.drop("loan_approved", axis=1)
y = data["loan_approved"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
model = MLPClassifier(max_iter=200)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Output
print("Accuracy:", accuracy_score(y_test, y_pred))
