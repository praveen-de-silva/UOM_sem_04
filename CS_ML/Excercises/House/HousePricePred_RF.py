import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
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

df = pd.get_dummies(df, drop_first=True)

X = df.drop(["price"], axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# ----------------------
#     Model Training
# ----------------------

model = RandomForestRegressor()
model.fit(X_train, y_train)

# -----------------------
#     Model Valdating
# -----------------------

y_pred = model.predict(X_test)
print(f"RMSE = {np.sqrt(mean_squared_error(y_test, y_pred))}")
print(f"r2-score = {np.sqrt(r2_score(y_test, y_pred))}")

# RMSE = 1406637.1813562706
# r2-score = 0.7554027010670709