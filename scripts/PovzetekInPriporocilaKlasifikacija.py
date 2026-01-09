import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def dataframe_to_markdown(df, float_format=".3f"):
    """Pretvori pandas DataFrame v Markdown tabelo."""
    header = " | ".join(df.columns)
    separator = " | ".join(["---"] * len(df.columns))
    rows = []
    for _, row in df.iterrows():
        row_values = []
        for col in df.columns:
            val = row[col]
            if isinstance(val, float):
                row_values.append(f"{val:{float_format}}")
            else:
                row_values.append(str(val))
        rows.append(" | ".join(row_values))

    return "\n".join([header, separator] + rows)


def calculate_sigma(total_opportunities, total_defects):
    """Izracuna DPMO in nivo Sigma."""
    if total_opportunities == 0 or total_defects < 0:
        return 0, 0

    if total_defects == 0:
        return 0, 6.0

    dpmo = (total_defects / total_opportunities) * 1_000_000
    sigma_level = norm.ppf(1 - (dpmo / 1_000_000)) + 1.5

    if np.isinf(sigma_level):
        return dpmo, 0

    return dpmo, sigma_level


def run_analysis():
    """
    Nalozi najboljsi klasifikacijski model, analizira vpliv spremenljivk in simulira ucinke
    predlaganih sprememb za zmanjsanje odliva strank.
    """
    # Nalaganje modela in podatkov
    try:
        model_path = "../modeli_klasifikacija/RandomForest_best_model.pkl"
        best_model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Napaka: Model na poti '{model_path}' ni bil najden.")
        return

    # Priprava testnih podatkov (enako kot v skripti za ucenje)
    df = pd.read_csv("../Telco-Customer-Churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.loc[df["tenure"] == 0, "TotalCharges"] = 0
    df["Churn_bin"] = df["Churn"].map({"No": 0, "Yes": 1})

    X = df.drop(columns=["Churn", "Churn_bin", "customerID"])
    y = df["Churn_bin"]

    # Razdelitev na ucno in testno mnozico (z enakim random_state)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 1. Analiza vpliva spremenljivk
    try:
        # Pridobivanje imen znacilk po transformaciji
        feature_names = best_model.named_steps['preprocess'].get_feature_names_out()
        importances = best_model.named_steps['model'].feature_importances_

        importance_df = pd.DataFrame({
            'Spremenljivka': feature_names,
            'Vpliv (Gini)': importances
        }).sort_values(by='Vpliv (Gini)', ascending=False)

        # Analiza smeri vpliva (korelacija z odlivom)
        X_test_transformed = best_model.named_steps['preprocess'].transform(X_test)
        X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=feature_names, index=X_test.index)

        correlations = {}
        for feature in importance_df['Spremenljivka'].head(10):
            correlations[feature] = np.corrcoef(X_test_transformed_df[feature], y_test)[0, 1]

        importance_df['Smer vpliva'] = importance_df['Spremenljivka'].map(correlations)
        importance_df['Smer vpliva'] = importance_df['Smer vpliva'].apply(
            lambda x: 'Pozitiven' if x > 0 else 'Negativen' if x < 0 else 'Ni korelacije'
        )

    except Exception as e:
        print(f"Napaka pri analizi vpliva spremenljivk: {e}")
        importance_df = pd.DataFrame()

    # 2. Izracun osnovnih metrik in Six Sigma (pred spremembami)
    y_pred_before = best_model.predict(X_test)
    y_prob_before = best_model.predict_proba(X_test)[:, 1]

    total_opportunities = len(y_test)
    defects_before = np.sum(y_test != y_pred_before)
    dpmo_before, sigma_before = calculate_sigma(total_opportunities, defects_before)

    metrics_before = {
        "AUC": roc_auc_score(y_test, y_prob_before),
        "Accuracy": accuracy_score(y_test, y_pred_before),
        "F1-score": f1_score(y_test, y_pred_before)
    }

    # 3. Simulacije globalnih sprememb
    simulations = {
        "Spodbujanje daljsih pogodb": {
            "change_func": lambda df: df.assign(
                Contract=lambda x: x['Contract'].replace({'Month-to-month': 'One year'})),
            "description": "Sprememba kratkorocnih pogodb (mesec-za-mesec) v enoletne pogodbe."
        },
        "Ponudba tehnoloske podpore": {
            "change_func": lambda df: df.assign(TechSupport=lambda x: x['TechSupport'].replace({'No': 'Yes'})),
            "description": "Zagotovitev tehnoloske podpore vsem strankam, ki je trenutno nimajo."
        },
        "Povecanje zvestobe (tenure)": {
            "change_func": lambda df: df.assign(tenure=lambda x: x['tenure'] + 12),
            "description": "Simulacija uspesne kampanje za ohranjanje strank, ki podaljsa njihovo dobo narocnistva za 12 mesecev."
        }
    }

    simulation_results = []
    sigma_results = []

    for name, sim in simulations.items():
        X_test_after = X_test.copy()
        X_test_after = sim["change_func"](X_test_after)

        y_pred_after = best_model.predict(X_test_after)
        y_prob_after = best_model.predict_proba(X_test_after)[:, 1]

        defects_after = np.sum(y_test != y_pred_after)
        dpmo_after, sigma_after = calculate_sigma(total_opportunities, defects_after)

        sigma_results.append({
            "Model": name,
            "DPMO PREJ": dpmo_before,
            "Sigma PREJ": sigma_before,
            "DPMO POTEM": dpmo_after,
            "Sigma POTEM": sigma_after,
            "Izboljšava": sigma_after - sigma_before
        })

        metrics_after = {
            "AUC": roc_auc_score(y_test, y_prob_after),
            "Accuracy": accuracy_score(y_test, y_pred_after),
            "F1-score": f1_score(y_test, y_pred_after)
        }

        for metric_name in metrics_before:
            simulation_results.append({
                "Simulacija": name,
                "Opis": sim["description"],
                "Metrika": metric_name,
                "Vrednost pred": metrics_before[metric_name],
                "Vrednost po": metrics_after[metric_name],
                "Razlika": metrics_after[metric_name] - metrics_before[metric_name]
            })

    sim_df = pd.DataFrame(simulation_results)
    sigma_df = pd.DataFrame(sigma_results)

    # 4. Izpis rezultatov v konzolo
    print("\n# 1. Spremenljivke z najvecjim vplivom")
    print(dataframe_to_markdown(importance_df.head(10)))

    print("\n# 2. Simulacije in strateska priporocila")
    print(dataframe_to_markdown(sim_df, float_format=".4f"))

    print("\n# 3. Six Sigma analiza – PREJ in POTEM")
    print(dataframe_to_markdown(sigma_df, float_format=".4f"))

    print("\nAnaliza je koncana.")



if __name__ == '__main__':
    run_analysis()