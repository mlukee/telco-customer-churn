import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("Telco-Customer-Churn-Cleaned.csv")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna(subset=['TotalCharges', 'tenure'])

if 'customerID' in df.columns:
    df = df.drop(columns=['customerID'])

target = 'tenure'

num_cols = df.select_dtypes(include=[np.number]).columns.drop(target)
cat_cols = df.select_dtypes(exclude=[np.number]).columns

# --- Funkcija za preverjanje normalnosti (Shapiro-Wilk) ---
def is_normal(data):
    if len(data) < 3: return False 
    k2, p = stats.normaltest(data)
    return p > 0.05 

results = []

# ==========================================
# 1. VEJA: 2 NUMERIČNI (X vs Tenure)
# ==========================================
print("--- Analiza Numeričnih spremenljivk ---")
for col in num_cols:
    normal_x = is_normal(df[col])
    normal_y = is_normal(df[target])
    
    if normal_x and normal_y:
        test_name = "Pearson Correlation"
        stat, p = stats.pearsonr(df[col], df[target])
    else:
        test_name = "Spearman Correlation"
        stat, p = stats.spearmanr(df[col], df[target])
    
    results.append({
        "Spremenljivka": col,
        "Tip testa": test_name,
        "Statistika": round(stat, 3),
        "P-vrednost": round(p, 3) if p >= 0.001 else "< 0.001",
        "Signifikantno": "DA" if p < 0.05 else "NE"
    })
    
    # Grafični prikaz (Scatterplot)
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=df[col], y=df[target], alpha=0.3, color='steelblue')
    plt.title(f"Scatter: {col} vs {target} ({test_name})")
    plt.xlabel(col)
    plt.ylabel(target)
    plt.tight_layout()
    plt.show()

# ==========================================
# 2. VEJA: 1 NOMINALNA + 1 NUMERIČNA (X vs Tenure)
# ==========================================
print("--- Analiza Nominalnih spremenljivk ---")
for col in cat_cols:
    groups = df[col].unique()
    group_data = [df[df[col] == g][target] for g in groups]
    
    # Preverimo normalnost znotraj VSEH skupin
    all_normal = all(is_normal(g_data) for g_data in group_data)
    
    # A) 2 KATEGORIJI
    if len(groups) == 2:
        if all_normal:
            # Preverimo enakost varianc (Levene)
            stat_var, p_var = stats.levene(*group_data)
            if p_var > 0.05:
                test_name = "Student t-test"
                stat, p = stats.ttest_ind(*group_data, equal_var=True)
            else:
                test_name = "Welch t-test"
                stat, p = stats.ttest_ind(*group_data, equal_var=False)
        else:
            # Ni normalna porazdelitev -> Mann-Whitney U
            test_name = "Mann-Whitney U"
            stat, p = stats.mannwhitneyu(*group_data)

    # B) VEČ KOT 2 KATEGORIJI (> 2)
    else:
        if all_normal:
            # Preverimo enakost varianc (Levene)
            stat_var, p_var = stats.levene(*group_data)
            if p_var > 0.05:
                test_name = "ANOVA"
                stat, p = stats.f_oneway(*group_data)
            else:
                test_name = "Welch ANOVA"
                stat, p = stats.f_oneway(*group_data)
        else:
            # Ni normalna porazdelitev -> Kruskal-Wallis
            test_name = "Kruskal-Wallis H"
            stat, p = stats.kruskal(*group_data)

    results.append({
        "Spremenljivka": col,
        "Tip testa": test_name,
        "Statistika": round(stat, 3),
        "P-vrednost": round(p, 3) if p >= 0.001 else "< 0.001",
        "Signifikantno": "DA" if p < 0.05 else "NE"
    })

    # Grafični prikaz (Boxplot)
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col], y=df[target], palette="pastel", hue=df[col], legend=False)
    plt.title(f"Boxplot: {col} vs {target} ({test_name})")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

results_df = pd.DataFrame(results)
print("\n=== REZULTATI BIVARIATNE ANALIZE (Target: tenure) ===")
print(results_df)