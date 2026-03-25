import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

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

X = df.drop(["price"], axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# --------------------------
#     Data 
# --------------------------

model = RandomForestRegressor()
