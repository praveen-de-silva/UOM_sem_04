import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("./Inputs/titanic_train.csv")

# --- Basic Checks ---
# print(df.head())
print(df.info())
# print(df.isnull().sum())
# print(df.shape)

# --- Simple Cleaning ---
df = df.dropna() 
# print(df.shape)

# --- Split ---
X = df.drop(["Survived", "Name", "Sex", "Ticket", "Cabin", "Embarked", "Pclass", "Parch"], axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Model ---
# print(X, )
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# --- Test ---
y_pred = model.predict(X_test)
print("Accuracy : ", accuracy_score(y_test, y_pred))