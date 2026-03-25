import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


# --------------------
#     Data Loading
# --------------------

df = pd.read_csv("./Inputs/titanic_train.csv")

# --------------------------
#     Data Preprocessing
# --------------------------

# print(df.shape)
# print(df.info())

# corr = df.corr()
# sns.heatmap(corr, annot=True)

df = df.drop(["Cabin"], axis=1)
df = pd.get_dummies(df, drop_first=True)

# print(df.shape)

X = df.drop(["Survived"], axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# ----------------------
#     Model Training
# ----------------------

model = XGBClassifier(
    n_estimators = 100,
    max_depth = 5,
    learning_rate = 0.01,
    subsample = 0.8,
    use_lable_encoder = False,
    eval_metric = 'logloss'
)
model.fit(X_train, y_train)

# ------------------------
#     Model Evaluating
# ------------------------

y_pred = model.predict(X_test)
print(f"Accuracy : {accuracy_score(y_test, y_pred)}")

