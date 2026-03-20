import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# === Step 01 : Input & Inspection ===

df = pd.read_csv("./Inputs/titanic_train.csv")

# print(df.info())
# print(df.describe())
# print(df.isnull().sum())

# === Step 02 : Data Preporcessing ===

# --- Step 02.1 : Data Cleaning ---
# df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
df = df.drop(["PassengerId", "Ticket","Cabin"], axis=1)

print(df.head())

# --- Step 02.2 : Data splitting ---
X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# X_train["Age"] = X_train["Age"].fillna(X_train["Age"].median())
# X_val = X_val.fillna(X_train["Age"].median())

X_train = pd.get_dummies(X_train, drop_first=True)
X_val = pd.get_dummies(X_val, drop_first=True)

X_val = X_val.reindex(columns=X_train.columns, fill_value=0)


# === Step 03 : Training ===

model = RandomForestClassifier()
model.fit(X_train, y_train)

# === Step 04 : Evalution ===

y_pred = model.predict(X_val)
print(f"Accuracy = {accuracy_score(y_val, y_pred)}")

# X = df.drop('')
