import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- Read CSV and Make ---
df = pd.read_csv("./Inputs/Iris.csv")

# df.info()
# print(df.head(122))
# print(df.shape)
# print(df["Species"].unique())

df["Species"] = df["Species"].map({
    "Iris-setosa" : 0,
    "Iris-versicolor" : 1,
    "Iris-virginica" : 2
})

# --- Cleaning Data ---
# print(df.isnull().sum())
# print(df.shape)


# --- Split Data ---
X = df.drop(["Id", "Species"], axis=1)
y = df["Species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model ---
model = RandomForestClassifier()
model.fit(X_train, y_train)

# --- Test ---
y_pred = model.predict(X_test)
print("Accuracy", accuracy_score(y_test, y_pred))
# print(y_pred)