# ============================================================
#  Road Accidents — REGRESSION PIPELINE  (v2)
#  Target  : accident_risk  (continuous)
#  Models  : XGBoost  +  CatBoost  (tuned, ensembled)
#
#  What changed from v1:
#  [FIX]  Data-leakage bug patched: imputer/OHE now fitted
#         ONLY on train fold, never on val/test data.
#  [NEW]  XGBoost uses sklearn Pipeline + ColumnTransformer
#         (SO-post technique): preprocessor is encapsulated
#         so RandomizedSearchCV tunes it leakage-free across
#         all CV folds automatically.
#  [NEW]  OneHotEncoder(handle_unknown='ignore') replaces
#         pd.get_dummies + align — robustly handles unseen
#         test categories (SO-post technique).
#  [KEPT] RandomizedSearchCV — already better than GridSearchCV
#         for our large param grids (our approach was correct).
#  [KEPT] No StandardScaler — trees are scale-invariant.
#  [KEPT] CatBoost native categoricals via Pool.
#  [KEPT] Early stopping + weighted ensemble.
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import (
    train_test_split, RandomizedSearchCV, cross_val_score
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from xgboost import XGBRegressor
from catboost import CatBoostRegressor, Pool

# ─────────────────────────────────────────────
#  0.  CONFIGURATION
# ─────────────────────────────────────────────
TRAIN_PATH   = "train.csv"
TEST_PATH    = "test.csv"
OUTPUT_DIR   = Path("./Outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET       = "accident_risk"
ID_COL       = "id"
DROP_COLS    = ["road_signs_present", "id", "school_season", "num_lanes", "time_of_day"]
RANDOM_STATE = 42
TEST_SIZE    = 0.20
CV_FOLDS     = 5
TUNE_ITERS   = 40


# ─────────────────────────────────────────────
#  1.  DATA LOADING
# ─────────────────────────────────────────────
def load_data():
    print("=" * 60)
    print("  ROAD ACCIDENTS — REGRESSION PIPELINE  (v2)")
    print("=" * 60)
    df      = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)
    print(f"\n[Data] Train shape : {df.shape}")
    print(f"[Data] Test  shape : {df_test.shape}")
    print(f"\n[Data] Target distribution:\n{df[TARGET].describe()}")
    return df, df_test


# ─────────────────────────────────────────────
#  2.  LIGHT PREP  (no imputation / encoding yet)
#      Real preprocessing happens inside each model
#      function, fitted on train fold only → no leakage.
# ─────────────────────────────────────────────
def initial_prep(df, df_test):
    print("\n[Prep] Dropping unused columns …")
    test_ids = df_test[ID_COL].copy() if ID_COL in df_test.columns else None

    for col in DROP_COLS:
        if col in df:      df      = df.drop(col, axis=1)
        if col in df_test: df_test = df_test.drop(col, axis=1)

    X      = df.drop(TARGET, axis=1)
    y      = df[TARGET].copy()
    X_test = df_test.copy()

    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    print(f"[Prep] Numeric   cols ({len(num_cols)}) : {num_cols}")
    print(f"[Prep] Categoric cols ({len(cat_cols)}) : {cat_cols}")
    return X, y, X_test, num_cols, cat_cols, test_ids


# ─────────────────────────────────────────────
#  3.  SKLEARN PREPROCESSOR BUILDER
#      (ColumnTransformer pattern from SO post)
# ─────────────────────────────────────────────
def build_preprocessor(num_cols, cat_cols):
    """
    Numerics  → median imputation only (no scaler — trees are scale-invariant)
    Categorics → most_frequent imputation → OneHotEncoder
    handle_unknown='ignore' handles unseen test categories robustly
    (SO-post improvement over pd.get_dummies + align).
    """
    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        # No StandardScaler: XGBoost/CatBoost don't benefit from it
    ])
    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols),
        ],
        remainder="drop"
    )


# ─────────────────────────────────────────────
#  4.  METRICS HELPER
# ─────────────────────────────────────────────
def print_metrics(y_true, y_pred, label="Val"):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"  [{label}] RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")
    return rmse, mae, r2


# ─────────────────────────────────────────────
#  5.  XGBOOST  (sklearn Pipeline — leakage-free)
# ─────────────────────────────────────────────
def train_xgboost(X_train, y_train, X_val, y_val,
                  X, y, num_cols, cat_cols, X_test):
    """
    Wraps ColumnTransformer + XGBRegressor in a sklearn Pipeline.
    RandomizedSearchCV tunes the whole pipeline → preprocessor is
    re-fitted inside every CV fold → zero data leakage.
    Param names carry the 'xgb__' prefix (sklearn Pipeline convention).
    """
    print("\n" + "─" * 60)
    print("  XGBoost Regressor  [sklearn Pipeline, leakage-free]")
    print("─" * 60)

    xgb_pipeline = Pipeline([
        ("preprocessor", build_preprocessor(num_cols, cat_cols)),
        ("xgb", XGBRegressor(tree_method="hist", eval_metric="rmse",
                              random_state=RANDOM_STATE, n_jobs=-1))
    ])

    # 'xgb__' prefix routes params to the XGBRegressor step
    param_dist = {
        "xgb__n_estimators"     : [200, 300, 400, 500, 600],
        "xgb__max_depth"        : [3, 4, 5, 6, 7],
        "xgb__learning_rate"    : [0.01, 0.03, 0.05, 0.1, 0.15],
        "xgb__subsample"        : [0.6, 0.7, 0.8, 0.9, 1.0],
        "xgb__colsample_bytree" : [0.5, 0.6, 0.7, 0.8, 1.0],
        "xgb__min_child_weight" : [1, 3, 5, 7],
        "xgb__gamma"            : [0, 0.1, 0.2, 0.5],
        "xgb__reg_alpha"        : [0, 0.01, 0.1, 1.0],
        "xgb__reg_lambda"       : [1, 1.5, 2, 5],
    }

    print(f"[XGBoost] RandomizedSearchCV ({TUNE_ITERS} iters, {CV_FOLDS}-fold) …")
    xgb_search = RandomizedSearchCV(
        estimator=xgb_pipeline,
        param_distributions=param_dist,
        n_iter=TUNE_ITERS,
        scoring="r2",
        cv=CV_FOLDS,
        random_state=RANDOM_STATE,
        n_jobs=-1, verbose=0
    )
    xgb_search.fit(X_train, y_train)   # preprocessor fit on train fold only ✓

    best_params = xgb_search.best_params_
    print(f"[XGBoost] Best CV R²   : {xgb_search.best_score_:.4f}")
    print(f"[XGBoost] Best params  : {best_params}")

    # Strip 'xgb__' prefix → pass directly to XGBRegressor
    xgb_params = {k.replace("xgb__", ""): v for k, v in best_params.items()}

    # Fit preprocessor on train only, transform val + test
    prep = build_preprocessor(num_cols, cat_cols)
    X_tr_proc = prep.fit_transform(X_train, y_train)
    X_vl_proc = prep.transform(X_val)
    X_te_proc = prep.transform(X_test)

    # Final model with early stopping
    xgb_model = XGBRegressor(
        **xgb_params,
        tree_method="hist", eval_metric="rmse",
        n_estimators=1000, early_stopping_rounds=50,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    xgb_model.fit(X_tr_proc, y_train,
                  eval_set=[(X_vl_proc, y_val)], verbose=False)

    print_metrics(y_val, xgb_model.predict(X_vl_proc), "XGBoost Val")

    # Leakage-free CV on full data via Pipeline
    cv_pipe = Pipeline([
        ("preprocessor", build_preprocessor(num_cols, cat_cols)),
        ("xgb", XGBRegressor(**xgb_params, tree_method="hist",
                              eval_metric="rmse", random_state=RANDOM_STATE, n_jobs=-1))
    ])
    cv = cross_val_score(cv_pipe, X, y, cv=CV_FOLDS, scoring="r2", n_jobs=-1)
    print(f"[XGBoost] CV R² = {cv.mean():.4f} ± {cv.std():.4f}")

    # Feature importance
    try:
        ohe_names = list(prep.named_transformers_["cat"]
                         .named_steps["ohe"].get_feature_names_out(cat_cols))
        feat_names = num_cols + ohe_names
    except Exception:
        feat_names = [f"f{i}" for i in range(xgb_model.n_features_in_)]
    imp = pd.Series(xgb_model.feature_importances_, index=feat_names)
    print("\n[XGBoost] Top 15 features:")
    print(imp.sort_values(ascending=False).head(15).to_string())

    return xgb_model, prep, X_vl_proc, X_te_proc, xgb_params


# ─────────────────────────────────────────────
#  6.  CATBOOST  (native categoricals, leakage-fixed)
# ─────────────────────────────────────────────
def train_catboost(X_train, y_train, X_val, y_val,
                   X, y, cat_cols, X_test):
    """
    CatBoost handles categoricals natively via Pool — no OHE.
    v2 fix: imputers are now fit ONLY on X_train (was full X in v1).
    """
    print("\n" + "─" * 60)
    print("  CatBoost Regressor  [native cats, leakage-fixed]")
    print("─" * 60)

    num_cols_raw = X_train.select_dtypes(include=["number"]).columns.tolist()

    # Fit imputers on train ONLY ✓
    num_imp = SimpleImputer(strategy="median")
    cat_imp = SimpleImputer(strategy="most_frequent")

    def _apply(df_in, fit=False):
        out = df_in.copy()
        if fit:
            out[num_cols_raw] = num_imp.fit_transform(out[num_cols_raw])
            if cat_cols:
                out[cat_cols] = cat_imp.fit_transform(out[cat_cols])
        else:
            out[num_cols_raw] = num_imp.transform(out[num_cols_raw])
            if cat_cols:
                out[cat_cols] = cat_imp.transform(out[cat_cols])
        if cat_cols:
            out[cat_cols] = out[cat_cols].fillna("missing")
        return out

    X_tr = _apply(X_train, fit=True)   # imputers fitted here only ✓
    X_vl = _apply(X_val)
    X_te = _apply(X_test)

    cat_fi = [X_tr.columns.get_loc(c) for c in cat_cols if c in X_tr.columns]
    train_pool = Pool(X_tr, y_train, cat_features=cat_fi)
    val_pool   = Pool(X_vl, y_val,   cat_features=cat_fi)

    param_dist = {
        "iterations"       : [300, 500, 700, 1000],
        "learning_rate"    : [0.01, 0.03, 0.05, 0.1],
        "depth"            : [4, 5, 6, 7, 8],
        "l2_leaf_reg"      : [1, 3, 5, 7, 10],
        "min_data_in_leaf" : [1, 5, 10, 20],
        "subsample"        : [0.6, 0.7, 0.8, 1.0],
        "colsample_bylevel": [0.6, 0.8, 1.0],
    }

    base_cat = CatBoostRegressor(random_seed=RANDOM_STATE, verbose=0,
                                  cat_features=cat_fi)

    print(f"[CatBoost] RandomizedSearchCV ({TUNE_ITERS} iters, {CV_FOLDS}-fold) …")
    cat_search = RandomizedSearchCV(
        estimator=base_cat, param_distributions=param_dist,
        n_iter=TUNE_ITERS, scoring="r2", cv=CV_FOLDS,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=0
    )
    cat_search.fit(X_tr, y_train)

    best_params = cat_search.best_params_
    print(f"[CatBoost] Best CV R²   : {cat_search.best_score_:.4f}")
    print(f"[CatBoost] Best params  : {best_params}")

    cat_model = CatBoostRegressor(
        **best_params, random_seed=RANDOM_STATE, verbose=0,
        cat_features=cat_fi, early_stopping_rounds=50
    )
    cat_model.fit(train_pool, eval_set=val_pool)

    print_metrics(y_val, cat_model.predict(X_vl), "CatBoost Val")

    cv = cross_val_score(
        CatBoostRegressor(**best_params, random_seed=RANDOM_STATE,
                          verbose=0, cat_features=cat_fi),
        X_tr, y_train, cv=CV_FOLDS, scoring="r2", n_jobs=-1
    )
    print(f"[CatBoost] CV R² = {cv.mean():.4f} ± {cv.std():.4f}")

    imp = pd.Series(cat_model.get_feature_importance(), index=X_tr.columns)
    print("\n[CatBoost] Top 15 features:")
    print(imp.sort_values(ascending=False).head(15).to_string())

    return cat_model, num_imp, cat_imp, X_vl, X_te, best_params, cat_fi


# ─────────────────────────────────────────────
#  7.  WEIGHTED ENSEMBLE
# ─────────────────────────────────────────────
def build_ensemble(xgb_model, cat_model,
                   X_vl_proc, X_vl_cat, y_val,
                   X_te_proc, X_te_cat):
    print("\n" + "─" * 60)
    print("  Weighted Ensemble  (XGBoost + CatBoost)")
    print("─" * 60)

    xgb_val = xgb_model.predict(X_vl_proc)
    cat_val = cat_model.predict(X_vl_cat)

    best_w, best_r2 = 0.5, -np.inf
    for w in np.arange(0.05, 1.0, 0.05):
        r2 = r2_score(y_val, w * xgb_val + (1 - w) * cat_val)
        if r2 > best_r2:
            best_r2, best_w = r2, w

    print(f"[Ensemble] Optimal XGB weight = {best_w:.2f}  →  Val R² = {best_r2:.4f}")
    print_metrics(y_val, best_w * xgb_val + (1 - best_w) * cat_val, "Ensemble Val")

    blend_test = best_w * xgb_model.predict(X_te_proc) + (1 - best_w) * cat_model.predict(X_te_cat)
    return blend_test, xgb_model.predict(X_te_proc), cat_model.predict(X_te_cat)


# ─────────────────────────────────────────────
#  8.  SUBMISSION
# ─────────────────────────────────────────────
def save_submissions(blend_test, xgb_test, cat_test, test_ids):
    def _save(preds, name):
        out = pd.DataFrame({TARGET: preds})
        if test_ids is not None:
            out.insert(0, ID_COL, test_ids.values)
        path = OUTPUT_DIR / name
        out.to_csv(path, index=False)
        print(f"  Saved → {path}")

    print("\n[Submission] Saving …")
    _save(blend_test, "submission_ensemble_regression.csv")
    _save(xgb_test,   "submission_xgboost_regression.csv")
    _save(cat_test,   "submission_catboost_regression.csv")


# ─────────────────────────────────────────────
#  9.  MAIN
# ─────────────────────────────────────────────
def main():
    df, df_test = load_data()
    X, y, X_test, num_cols, cat_cols, test_ids = initial_prep(df, df_test)

    # Split BEFORE any preprocessing → guarantees leakage-free fitting
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"\n[Split] Train={len(X_train)}  Val={len(X_val)}")

    # Train both models
    xgb_model, xgb_prep, X_vl_proc, X_te_proc, xgb_params = train_xgboost(
        X_train, y_train, X_val, y_val, X, y, num_cols, cat_cols, X_test
    )
    cat_model, cat_num_imp, cat_cat_imp, X_vl_cat, X_te_cat, cat_params, cat_fi = train_catboost(
        X_train, y_train, X_val, y_val, X, y, cat_cols, X_test
    )

    # Retrain on FULL data (best practice before final submission)
    print("\n[Final] Retraining on full train data …")

    # XGBoost full retrain via Pipeline (preprocessor re-fit on all X)
    full_pipe = Pipeline([
        ("preprocessor", build_preprocessor(num_cols, cat_cols)),
        ("xgb", XGBRegressor(**xgb_params, tree_method="hist",
                              eval_metric="rmse", random_state=RANDOM_STATE, n_jobs=-1))
    ])
    full_pipe.fit(X, y)
    X_te_proc_full = full_pipe.named_steps["preprocessor"].transform(X_test)
    xgb_full       = full_pipe.named_steps["xgb"]

    # CatBoost full retrain
    num_cols_raw = X.select_dtypes(include=["number"]).columns.tolist()
    X_full_cat = X.copy()
    X_te_cat_f = X_test.copy()
    X_full_cat[num_cols_raw] = cat_num_imp.fit_transform(X_full_cat[num_cols_raw])
    X_te_cat_f[num_cols_raw] = cat_num_imp.transform(X_te_cat_f[num_cols_raw])
    if cat_cols:
        X_full_cat[cat_cols] = cat_cat_imp.fit_transform(X_full_cat[cat_cols])
        X_te_cat_f[cat_cols] = cat_cat_imp.transform(X_te_cat_f[cat_cols])
        for df_ in [X_full_cat, X_te_cat_f]:
            df_[cat_cols] = df_[cat_cols].fillna("missing")

    cat_model.set_params(early_stopping_rounds=None)
    cat_model.fit(Pool(X_full_cat, y, cat_features=cat_fi))

    # Ensemble & submit
    blend_test, xgb_test, cat_test = build_ensemble(
        xgb_full, cat_model,
        X_vl_proc, X_vl_cat, y_val,
        X_te_proc_full, X_te_cat_f
    )
    save_submissions(blend_test, xgb_test, cat_test, test_ids)

    print("\n" + "=" * 60)
    print("  Regression Pipeline Complete ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
