import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

# --- 1. PRIPRAVA PODATKOV (Kopirano in prilagojeno iz regresijskoUcenje.py) ---
try:
    df = pd.read_csv("Telco-Customer-Churn-Cleaned.csv")
except FileNotFoundError:
    try:
        df = pd.read_csv("../Telco-Customer-Churn-Cleaned.csv")
    except FileNotFoundError:
        print("Napaka: Datoteke s podatki ni mogoče najti.")
        exit()

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna(subset=['TotalCharges', 'tenure'])

df_model = df.copy()
drop_cols = ["customerID", "TotalCharges"] 
df_model = df_model.drop(columns=[c for c in drop_cols if c in df_model.columns])

y = df_model["tenure"]
X = df_model.drop(columns=["tenure"])

# One-Hot Encoding
X_encoded = pd.get_dummies(X, drop_first=True)

# Izbor značilk (mora biti enak kot pri učenju modela)
selected_features = [
    "Contract_Two year", "Contract_One year", "MonthlyCharges", 
    "MultipleLines_Yes", "Partner_Yes", "OnlineBackup_Yes", 
    "PaymentMethod_Mailed check", "Churn_Yes", "OnlineSecurity_Yes", 
    "DeviceProtection_Yes", "PaymentMethod_Electronic check"
]

# Preverimo, če katere značilke manjkajo in jih dodamo z 0 (redkost, a previdnost ne škodi)
for feat in selected_features:
    if feat not in X_encoded.columns:
        X_encoded[feat] = 0

X_final = X_encoded[selected_features]
y_final = df["tenure"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y_final, test_size=0.2, random_state=42
)

# --- 2. UČENJE MODELA ---
# Uporabimo Gradient Boosting kot "izbran model" za simulacijo
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
model.fit(X_train, y_train)

print(f"Model: Gradient Boosting Regressor treniran. R2 na testni množici: {model.score(X_test, y_test):.3f}")

# --- 3. DEFINICIJA SIMULACIJ ---

LSL = 15 # Lower Specification Limit (Meseci) - Cilj je zadržati kupca vsaj 15 mesecev

# Funkcija za izračun in izpis rezultatov
def evaluate_simulation(model, X_original, X_simulated, scenario_name):
    pred_original = model.predict(X_original)
    pred_simulated = model.predict(X_simulated)
    
    avg_original = np.mean(pred_original)
    avg_simulated = np.mean(pred_simulated)
    diff = avg_simulated - avg_original
    pct_change = (diff / avg_original) * 100
    
    # Six Sigma Calculations
    def calc_sigma(data, limit):
        defects = np.sum(data < limit)
        n = len(data)
        if n == 0: return 0, 0
        dpmo = (defects / n) * 1_000_000
        yield_rate = 1 - (defects / n)
        
        if yield_rate >= 0.999997: # Cap at ~6 sigma
             sigma = 6.0
        elif yield_rate <= 0.000001: 
             sigma = 0.0
        else:
             sigma = stats.norm.ppf(yield_rate) + 1.5
        return dpmo, sigma

    dpmo_prej, sigma_prej = calc_sigma(pred_original, LSL)
    dpmo_potem, sigma_potem = calc_sigma(pred_simulated, LSL)
    imp_sigma = sigma_potem - sigma_prej
    
    return {
        "Model": "Gradient Boosting",
        "Scenario": scenario_name,
        "Avg Tenure (Pred)": round(avg_original, 3),
        "Avg Tenure (Po)": round(avg_simulated, 3),
        "Razlika": round(diff, 3),
        "Sprememba (%)": f"{round(pct_change, 2)}%",
        "DPMO PREJ": int(dpmo_prej),
        "Sigma PREJ": round(sigma_prej, 2),
        "DPMO POTEM": int(dpmo_potem),
        "Sigma POTEM": round(sigma_potem, 2),
        "Izboljšava Sigma": round(imp_sigma, 2)
    }

results = []

print("--- Zaganjam simulacije za VSE značilke ---")

for feature in selected_features:
    X_sim = X_test.copy()
    
    # 1. Numerične značilke (MonthlyCharges)
    if feature == "MonthlyCharges":
        X_sim[feature] = X_sim[feature] * 0.8
        scenario_name = "Znižanje MonthlyCharges (-20%)"
    
    # 2. Pogodbene značilke (Contract) - medsebojno izključevanje
    elif "Contract" in feature:
        X_sim[feature] = 1
        # Nastavimo ostale Contract značilke na 0
        for col in X_test.columns:
            if "Contract" in col and col != feature:
                X_sim[col] = 0
        scenario_name = f"Vsi na: {feature}"
        
    # 3. Plačilne metode (PaymentMethod) - medsebojno izključevanje
    elif "PaymentMethod" in feature:
        X_sim[feature] = 1
        for col in X_test.columns:
            if "PaymentMethod" in col and col != feature:
                X_sim[col] = 0
        scenario_name = f"Vsi na: {feature}"
        
    # 4. Binarne značilke (vse ostalo, npr. _Yes)
    else:
        # Če je znakilnost že binarna (0/1), jo nastavimo na 1 (aktivacija)
        X_sim[feature] = 1
        scenario_name = f"Aktivacija: {feature}"
        
    results.append(evaluate_simulation(model, X_test, X_sim, scenario_name))


# --- 4. PRIKAZ REZULTATOV ---
results_df = pd.DataFrame(results)

print("\n=== 6. Testiranje globalnih sprememb (Simulacije) ===")
# Izberemo kolomne za osnovni prikaz
cols_basic = ["Scenario", "Avg Tenure (Pred)", "Avg Tenure (Po)", "Razlika", "Sprememba (%)"]
print(results_df[cols_basic].to_string(index=False))

print("\n=== Six Sigma Analiza (Tolerance: Tenure >= 15 mesecev) ===")
cols_sigma = ["Scenario", "DPMO PREJ", "Sigma PREJ", "DPMO POTEM", "Sigma POTEM", "Izboljšava Sigma"]
print(results_df[cols_sigma].to_string(index=False))

# Interpretacija
print("\n--- Interpretacija učinkov ---")
for res in results:
    s = res['Scenario']
    d = res['Razlika']
    imp = res['Izboljšava Sigma']
    sig_prej = res['Sigma PREJ']
    sig_potem = res['Sigma POTEM']
    
    if d > 0:
        desc = f"Povečanje sigma nivoja iz {sig_prej} na {sig_potem} (+{imp})."
        stat_sig = "POMEMBNA" if imp > 0.1 else "MANJŠA"
        print(f"- {s}: {desc} Izboljšava je {stat_sig}." + (" Upravičuje implementacijo." if imp > 0.3 else ""))
    else:
        print(f"- {s}: Ni izboljšanja sigma nivoja ({imp}). Verjetno neupravičeno.")

# Vizualizacija
scenarios = [r['Scenario'] for r in results]
diffs = [r['Razlika'] for r in results]

plt.figure(figsize=(12, 6))
sns.barplot(x=diffs, y=scenarios, palette="viridis", orient='h')
plt.axvline(0, color='black', linewidth=1)
plt.xlabel("Sprememba v pričakovani dobi (Meseci)")
plt.title("Vpliv simuliranih sprememb na Tenure (Predicted)")
plt.tight_layout()
plt.show()