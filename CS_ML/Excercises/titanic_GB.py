import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ================
#     1 : Load 
# ================

df = pd.read_csv("./Inputs/titanic_train.csv")

# =============================
#     2 : Data Preposessing
# =============================

# --- 2.1 : Data Cleaning ---
df = df.drop(["Name", "Cabin",  "PassengerId"], axis=1)

# --- 2.2 : Feature Eng ---
# df["FamSize"] = df["SibSp"] + df["Parch"] + 1

# --- 2.3 : Split ---
X = df.drop("Survived", axis=1)
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# --- 2.4 : Fill missing values ---
X_train["Age"] = X_train["Age"].fillna(X_train["Age"].median())
X_test["Age"] = X_test["Age"].fillna(X_train["Age"].median())

# --- 2.5 : Encoding ---
X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# --- 2.6 : Scaling ---
# for logistic regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================
#     3 : Model Training 
# ==========================

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# ============================
#     4 : Model Evaluating
# ============================

y_pred = model.predict(X_test_scaled)
print(f"Accuracy = {accuracy_score(y_test, y_pred)}")

