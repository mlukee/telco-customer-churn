# feature_selection_tenure.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

# --- 1. PRIPRAVA PODATKOV ---
df = pd.read_csv("Telco-Customer-Churn-Cleaned.csv")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna(subset=['TotalCharges', 'tenure'])

df_model = df.copy()

# Odstranimo nepotrebne stolpce
drop_cols = ["customerID", "TotalCharges"]  # TotalCharges je skoraj linearno odvisen od tenure
df_model = df_model.drop(columns=[c for c in drop_cols if c in df_model.columns])

# Oznacimo target
y = df_model["tenure"]
X = df_model.drop(columns=["tenure"])

# Pretvorimo kategorije v številske vrednosti (One-Hot Encoding)
X_encoded = pd.get_dummies(X, drop_first=True)

# Standardizacija (potrebna za LASSO)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Train/test split (naključno 80/20)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- 2. RANDOM FOREST FEATURE IMPORTANCE ---

rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
rf_importance = pd.DataFrame({
    "Feature": X_encoded.columns,
    "RF_Importance": np.round(rf.feature_importances_, 4)
}).sort_values("RF_Importance", ascending=False)

# --- 3. LASSO REGRESSION FEATURE SELECTION ---

lasso = LassoCV(cv=5, random_state=42, max_iter=5000)
lasso.fit(X_train, y_train)

lasso_importance = pd.DataFrame({
    "Feature": X_encoded.columns,
    "LASSO_Coeff": np.round(lasso.coef_, 4)
})
lasso_importance["LASSO_Abs"] = abs(lasso_importance["LASSO_Coeff"])
lasso_importance = lasso_importance.sort_values("LASSO_Abs", ascending=False)

# --- 4. RFE (Recursive Feature Elimination) ---

lr = LinearRegression()
rfe = RFE(lr, n_features_to_select=10)
rfe.fit(X_train, y_train)

rfe_results = pd.DataFrame({
    "Feature": X_encoded.columns,
    "RFE_Rank": rfe.ranking_
}).sort_values("RFE_Rank", ascending=True)

# --- 5. ZDRUŽIMO REZULTATE ---

feature_summary = rf_importance.merge(
    lasso_importance[["Feature", "LASSO_Coeff"]],
    on="Feature", how="outer"
).merge(
    rfe_results[["Feature", "RFE_Rank"]],
    on="Feature", how="outer"
)

feature_summary = feature_summary.sort_values(
    ["RF_Importance", "LASSO_Coeff"], ascending=False
).reset_index(drop=True)

# Zaokrožimo in prikažemo rezultate
feature_summary = feature_summary.round(3)
print("\n=== Povzetek pomembnosti značilk (Feature Selection Summary) ===")
print(feature_summary)