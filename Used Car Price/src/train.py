import pandas as pd
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

from preprocess import preprocess_data

# CONFIG
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "used_cars.csv"
MODEL_DIR = BASE_DIR / "model"

RF_MODEL_PATH = MODEL_DIR / "car_price_rf.pkl"
LR_MODEL_PATH = MODEL_DIR / "car_price_lr.pkl"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# LOAD & PREPROCESS DATA
df = pd.read_csv(r"E:\Projects\Used Car Price\data\used_cars.csv")
df = preprocess_data(df)

X = df.drop("price", axis=1)
y = df["price"]

# IDENTIFY COLUMN TYPES
cat_cols = X.select_dtypes(include="object").columns
num_cols = X.select_dtypes(exclude="object").columns

# PREPROCESSING
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", "passthrough", num_cols)
])

# TRAIN–TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# RANDOM FOREST MODEL

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=12,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", rf_model)
])

rf_pipeline.fit(X_train, y_train)
rf_pred = rf_pipeline.predict(X_test)

rf_r2 = r2_score(y_test, rf_pred)
rf_mae = mean_absolute_error(y_test, rf_pred)


# LINEAR REGRESSION MODEL

lr_model = LinearRegression()

lr_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", lr_model)
])

lr_pipeline.fit(X_train, y_train)
lr_pred = lr_pipeline.predict(X_test)

lr_r2 = r2_score(y_test, lr_pred)
lr_mae = mean_absolute_error(y_test, lr_pred)


# COMPARISON OUTPUT

print("\n📊 MODEL COMPARISON")
print("-" * 40)
print(f"Random Forest → R²: {rf_r2:.3f} | MAE: ₹{rf_mae:,.0f}")
print(f"Linear Reg.   → R²: {lr_r2:.3f} | MAE: ₹{lr_mae:,.0f}")


# FEATURE IMPORTANCE (RF ONLY)

feature_names = rf_pipeline.named_steps["preprocessor"].get_feature_names_out()
importances = rf_pipeline.named_steps["model"].feature_importances_

feature_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False).head(15)

plt.figure(figsize=(10, 6))
plt.barh(feature_df["feature"], feature_df["importance"])
plt.gca().invert_yaxis()
plt.title("Top 15 Feature Importances (Random Forest)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()


# SAVE MODELS

with open(RF_MODEL_PATH, "wb") as f:
    pickle.dump(rf_pipeline, f)

with open(LR_MODEL_PATH, "wb") as f:
    pickle.dump(lr_pipeline, f)

print("\n✅ Both models trained, evaluated, and saved successfully")
