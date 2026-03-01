import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("./Inputs/titanic_train.csv")

# === Basic Checks ===
print(df["Name"].head())
# print(df.info())
# print(df.isnull().sum())
# print(df.shape)


# === Simple Cleaning ===
# df = df.dropna() # Accuracy : 0.71508 -> 0.75676
df = df.drop(["Name", "PassengerId", "Ticket", "Cabin"], axis=1)
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

# === Split ===
X = df.drop(["Survived"], axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === Fill missing Values ===
median_Age = X_train["Age"].median()
mode_Embarked = X_train["Embarked"].mode()

X_train["Age"].fillna(median_Age, inplace=True)
X_train["Embarked"].fillna(mode_Embarked, inplace=True)

X_test["Age"].fillna(median_Age, inplace=True)
X_test["Embarked"].fillna(mode_Embarked, inplace=True)

# X_train = pd.get_dummies(X_train, columns=["Sex", "Embarked", "Pclass"], drop_first=True)
# X_test = pd.get_dummies(X_test, columns=["Sex", "Embarked", "Pclass"], drop_first=True)

X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

# === Model ===
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42) # Accuracy : 0.7567567567567568
model.fit(X_train, y_train)

# === Test ===
y_pred = model.predict(X_test)
print("Accuracy : ", accuracy_score(y_test, y_pred))