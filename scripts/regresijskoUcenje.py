import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Uvoz modelov
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.linear_model import Ridge  # Ridge je Linearna regresija s parametrom alpha
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import make_scorer, r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# --- 2.1 PRIPRAVA PODATKOV ---
df = pd.read_csv("../Telco-Customer-Churn-Cleaned.csv")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna(subset=['TotalCharges', 'tenure'])

df_model = df.copy()

drop_cols = ["customerID", "TotalCharges"] 
df_model = df_model.drop(columns=[c for c in drop_cols if c in df_model.columns])

y = df_model["tenure"]
X = df_model.drop(columns=["tenure"])

# Pretvorimo kategorije v številske vrednosti (One-Hot Encoding)
X_encoded = pd.get_dummies(X, drop_first=True)
selected_features = [
    "Contract_Two year", "Contract_One year", "MonthlyCharges", 
    "MultipleLines_Yes", "Partner_Yes", "OnlineBackup_Yes", 
    "PaymentMethod_Mailed check", "Churn_Yes", "OnlineSecurity_Yes", 
    "DeviceProtection_Yes", "PaymentMethod_Electronic check"
]

# Predpostavka: X_encoded in y ("tenure") sta že definirana iz prejšnjih korakov
X_final = X_encoded[selected_features]
y_final = df["tenure"]  # Prepričaj se, da se indexi ujemajo

# 2. Train / Test split (80% / 20%)
# Ker gre za "snapshot" dataset (ni časovne vrste v smislu datumov dogodkov),
# uporabimo random split s fiksnim semenom.
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y_final, test_size=0.2, random_state=42
)

# --- 2.2 GRADNJA MODELOV (Definicija 5 modelov z ne-privzetimi parametri) ---

models = {
    "Ridge Regression": Ridge(alpha=1.0, random_state=42), # Linearni model z regularizacijo
    "Decision Tree": DecisionTreeRegressor(max_depth=8, min_samples_split=10, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42),
    "KNN Regressor": KNeighborsRegressor(n_neighbors=10, weights='distance')
}

# --- 2.3 NAVZKRIŽNA VALIDACIJA (10-fold CV) in METRIKE ---

# Pripravimo K-Fold (z shuffle za dobro premešanje)
kf = KFold(n_splits=10, shuffle=True, random_state=42)

results_list = []

print("Zaganjam 10-fold Cross-Validation za 5 modelov...")

for name, model in models.items():
    # Definiramo metrike, ki jih želimo spremljati
    scoring = {
        'r2': 'r2',
        'neg_rmse': 'neg_root_mean_squared_error',
        'neg_mae': 'neg_mean_absolute_error',
        'neg_mape': 'neg_mean_absolute_percentage_error'
    }
    
    # Izvedba CV
    cv_results = cross_validate(model, X_train, y_train, cv=kf, scoring=scoring, n_jobs=-1)
    
    # Izračun povprečij in SD (za negativne metrike obrnemo znak)
    r2_mean = cv_results['test_r2'].mean()
    r2_sd = cv_results['test_r2'].std()
    
    rmse_mean = -cv_results['test_neg_rmse'].mean()
    rmse_sd = cv_results['test_neg_rmse'].std()
    
    mae_mean = -cv_results['test_neg_mae'].mean()
    mae_sd = cv_results['test_neg_mae'].std()
    
    mape_mean = -cv_results['test_neg_mape'].mean()
    mape_sd = cv_results['test_neg_mape'].std()
    
    # --- AIC / BIC Izračun ---
    # AIC/BIC sta primarna za linearne/probabilistične modele. 
    # Za neparametrske modele (RF, KNN) se pogosto ne računata ali pa se ocenita aproksimativno.
    # Tukaj bomo izračunali na podlagi RSS (Residual Sum of Squares) celotnega X_train.
    
    model.fit(X_train, y_train) # Treniramo na celotnem train setu za AIC izračun
    y_train_pred = model.predict(X_train)
    rss = np.sum((y_train - y_train_pred) ** 2)
    n = len(y_train)
    
    # Število parametrov (k)
    if hasattr(model, 'coef_'):
        k = len(model.coef_) + 1 # +1 za intercept
    else:
        # Aproksimacija za nelinearne modele (ni povsem natančna, a služi za primerjavo kompleksnosti)
        k = 0 # Označimo kot N/A ali 0, če ni linearni model
        
    if k > 0:
        aic = n * np.log(rss / n) + 2 * k
        bic = n * np.log(rss / n) + k * np.log(n)
        aic_str = f"{aic:.1f}"
        bic_str = f"{bic:.1f}"
    else:
        aic_str = "N/A"
        bic_str = "N/A"
        
    # Zapis parametrov (kratek opis)
    params_str = str(model.get_params()).replace('{', '').replace('}', '')[:50] + "..."

    results_list.append({
        "Model": name,
        "Parametri": params_str,
        "R2": f"{r2_mean:.3f} ± {r2_sd:.3f}",
        "RMSE": f"{rmse_mean:.3f} ± {rmse_sd:.3f}",
        "MAE": f"{mae_mean:.3f} ± {mae_sd:.3f}",
        "MAPE": f"{mape_mean:.3f} ± {mape_sd:.3f}",
        "AIC/BIC": f"{aic_str} / {bic_str}",
        "Raw_R2": cv_results['test_r2'] # Shranimo za boxplot
    })

# --- IZPIS TABELE ---
results_df = pd.DataFrame(results_list)
# Odstranimo raw podatke za lep izpis
display_df = results_df.drop(columns=["Raw_R2"])

print("\n=== Primerjava modelov (Validacijska množica: 10-fold CV) ===")
print(display_df.to_string(index=False))

# Shranimo v Excel
display_df.to_excel("model_comparison.xlsx", index=False)

# --- GRAFIČNI PRIKAZ ---
plt.figure(figsize=(10, 6))
# Priprava podatkov za boxplot
plot_data = [res["Raw_R2"] for res in results_list]
model_names = [res["Model"] for res in results_list]

sns.boxplot(data=plot_data, palette="viridis")
plt.xticks(ticks=range(len(model_names)), labels=model_names, rotation=15)
plt.ylabel("R² Score (Validacija)")
plt.title("Primerjava uspešnosti modelov (R² čez 10-fold CV)")
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()