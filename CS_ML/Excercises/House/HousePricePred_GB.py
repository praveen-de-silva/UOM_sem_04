import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -----------------
#     Load Data
# -----------------

df = pd.read_csv("./Inputs/Housing.csv")

# --------------------------
#     Data Preprocessing
# --------------------------

# print(df.shape)
# print(df.info())
# print(df.head())


# --- check imbalanceness ---
print(f"Target Var count = {df['price'].value_counts()}")
print(f"Target Var count - Norm = {df['price'].value_counts(normalize=True)*100}")

df = pd.get_dummies(df, drop_first=True)

print(type(df.corr()))
corr = df.corr()["price"].abs().sort_values(ascending=False)
selected_features = corr[corr > 0.3].index()
print(selected_features)

X = df.drop(["price"], axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# ----------------------
#     Model Training
# ----------------------

model = GradientBoostingRegressor()
model.fit(X_train, y_train)

# -----------------------
#     Model Valdating
# -----------------------

y_pred = model.predict(X_test)
print(f"RMSE = {np.sqrt(mean_squared_error(y_test, y_pred))}")
print(f"r2-score = {np.sqrt(r2_score(y_test, y_pred))}")

# RMSE = 1338077.40570275
# r2-score = 0.7819642424021003