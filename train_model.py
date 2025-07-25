# train_model_fixed.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# ---- Step 1: Generate a balanced dataset ----

np.random.seed(42)

# 500 Safe (burn_risk = 0)
safe = pd.DataFrame({
    'temperature': np.random.normal(35, 3, 500),
    'voltage': np.random.normal(48, 1, 500),
    'current': np.random.normal(2.5, 0.5, 500),
    'charge_cycles': np.random.randint(100, 800, 500),
    'burn_risk': 0
})

# 500 Unsafe (burn_risk = 1)
unsafe = pd.DataFrame({
    'temperature': np.random.normal(60, 5, 500),
    'voltage': np.random.normal(45, 1.5, 500),
    'current': np.random.normal(5.5, 1.0, 500),
    'charge_cycles': np.random.randint(900, 1500, 500),
    'burn_risk': 1
})

# Combine and shuffle
df = pd.concat([safe, unsafe]).sample(frac=1).reset_index(drop=True)

# ---- Step 2: Train the model ----
X = df.drop("burn_risk", axis=1)
y = df["burn_risk"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# ---- Step 3: Evaluate ----
y_pred = model.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ---- Step 4: Save the model ----
joblib.dump(model, "battery_model.pkl")
print("✅ Model saved as battery_model.pkl")
