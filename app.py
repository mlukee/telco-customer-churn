import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- KONFIGURACIJA STRANI ---
st.set_page_config(page_title="Telco Tenure Prediction", layout="wide")

st.title("📊 Telco Tenure Simulator & Optimizacija")
st.markdown("""
Ta aplikacija omogoča napovedovanje dolžine naročniškega razmerja (**Tenure**) na podlagi ključnih parametrov.
Spreminjajte vrednosti in opazujte, kako različni dejavniki vplivajo na zvestobo stranke.
""")

# --- 1. NALAGANJE MODELOV ---
@st.cache_resource
def load_models():
    # Poskrbi, da so imena datotek enaka tistim, ki si jih shranil
    try:
        models = {
            "Gradient Boosting": joblib.load("./modeli_regresija/best_model_GradientBoosting.pkl"),
            "Random Forest": joblib.load("./modeli_regresija/best_model_RandomForest.pkl"),
            "Ridge Regression": joblib.load("./modeli_regresija/best_model_Ridge.pkl")
        }
        return models
    except FileNotFoundError:
        st.error("Modeli niso najdeni! Prepričaj se, da so .pkl datoteke v isti mapi.")
        return {}

models = load_models()

# Seznam značilk, ki jih model pričakuje (vrstni red je pomemben!)
expected_features = [
    "Contract_Two year", 
    "Contract_One year", 
    "MonthlyCharges", 
    "MultipleLines_Yes", 
    "Partner_Yes", 
    "OnlineBackup_Yes", 
    "PaymentMethod_Mailed check", 
    "Churn_Yes", 
    "OnlineSecurity_Yes", 
    "DeviceProtection_Yes",
    "PaymentMethod_Electronic check"
]

# --- 2. STRANSKA VRSTICA (VNOS PODATKOV) ---
st.sidebar.header("1. Izbira modela")
selected_model_name = st.sidebar.selectbox("Izberi algoritem", list(models.keys()))
model = models[selected_model_name] if models else None

st.sidebar.header("2. Podatki o stranki")

# Vnosna polja (UI) - pretvorili jih bomo v format za model
def user_input_features():
    # Pogodba
    contract = st.sidebar.selectbox("Vrsta pogodbe (Contract)", 
                                    ["Month-to-month", "One year", "Two year"])
    
    # Mesečni strošek
    monthly_charges = st.sidebar.slider("Mesečni strošek ($)", 
                                        min_value=18.0, max_value=120.0, value=70.0, step=0.5)
    
    # Dodatne storitve
    col1, col2 = st.sidebar.columns(2)
    with col1:
        partner = st.checkbox("Ima partnerja?", value=True)
        multiple_lines = st.checkbox("Več linij?", value=False)
        online_security = st.checkbox("Online Security?", value=False)
    with col2:
        online_backup = st.checkbox("Online Backup?", value=False)
        device_protection = st.checkbox("Device Protection?", value=False)
    
    # Plačilo
    payment_method = st.sidebar.selectbox("Način plačila", 
                                          ["Electronic check", "Mailed check", "Credit card/Bank transfer"])
    
    # Status (za simulacijo po navadi predpostavimo, da še ni odšel, a model to rabi)
    churn_status = st.sidebar.radio("Status stranke (Churn)", ["Še je stranka (No)", "Je odšla (Yes)"], index=0)

    # --- PRETVORBA V FORMAT MODELA (Mapping) ---
    data = {feature: 0 for feature in expected_features}
    
    # Numerične
    data['MonthlyCharges'] = monthly_charges
    
    # Kategorialne (Contract)
    if contract == "One year":
        data['Contract_One year'] = 1
    elif contract == "Two year":
        data['Contract_Two year'] = 1
    # Month-to-month je 0 in 0
    
    # Boolean mapping
    data['MultipleLines_Yes'] = 1 if multiple_lines else 0
    data['Partner_Yes'] = 1 if partner else 0
    data['OnlineBackup_Yes'] = 1 if online_backup else 0
    data['OnlineSecurity_Yes'] = 1 if online_security else 0
    data['DeviceProtection_Yes'] = 1 if device_protection else 0
    data['Churn_Yes'] = 1 if churn_status == "Je odšla (Yes)" else 0
    
    # Payment Method mapping
    if payment_method == "Mailed check":
        data['PaymentMethod_Mailed check'] = 1
    elif payment_method == "Electronic check":
        data['PaymentMethod_Electronic check'] = 1
    
    features = pd.DataFrame(data, index=[0])
    return features, contract, monthly_charges  # Vrnemo tudi raw vrednosti za grafe

input_df, raw_contract, raw_charges = user_input_features()

# --- 3. GLAVNI DEL: PRIKAZ NAPOVEDI ---

st.subheader("3. Rezultat napovedi")

if model:
    prediction = model.predict(input_df)[0]
    
    # Prikaz velike številke
    col_pred1, col_pred2 = st.columns([1, 2])
    
    with col_pred1:
        st.metric(label="Pričakovana doba (Tenure)", value=f"{prediction:.1f} mesecev")
        
        # Interpretacija
        if prediction < 12:
            st.error("Kratkoročna stranka (visoko tveganje)")
        elif prediction < 48:
            st.warning("Srednjeročna stranka")
        else:
            st.success("Dolgoročna zvesta stranka")

    with col_pred2:
        # Prikaz relativno glede na povprečje (fiktivno povprečje dataset-a je ~32 mesecev)
        avg_tenure = 32.4
        fig, ax = plt.subplots(figsize=(6, 1.5))
        ax.barh(["Povprečje", "Napoved"], [avg_tenure, prediction], color=["lightgray", "#4CAF50"])
        ax.set_xlim(0, 75)
        ax.set_xlabel("Meseci")
        st.pyplot(fig)

    # --- 4. SIMULACIJA OPTIMIZACIJE (Vpliv spremenljivke) ---
    st.subheader("4. Simulacija optimizacije: Vpliv cene na dobo")
    st.markdown("Kaj se zgodi, če spremenimo mesečno ceno, ostale parametre pa pustimo enake?")
    
    # Generiramo razpon cen za simulacijo
    sim_charges = np.linspace(20, 120, 50)
    sim_preds = []
    
    # Kopiramo input in spreminjamo samo ceno
    temp_df = input_df.copy()
    
    for c in sim_charges:
        temp_df['MonthlyCharges'] = c
        sim_preds.append(model.predict(temp_df)[0])
        
    # Graf
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sns.lineplot(x=sim_charges, y=sim_preds, ax=ax2, color="blue", linewidth=2)
    
    # Dodamo točko za trenutni izbor
    ax2.scatter([raw_charges], [prediction], color="red", s=100, label="Trenutni izbor", zorder=5)
    
    ax2.set_xlabel("Mesečni strošek ($)")
    ax2.set_ylabel("Napovedan Tenure (meseci)")
    ax2.set_title("Kako cena vpliva na zvestobo (ceteris paribus)")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    st.pyplot(fig2)
    
    # --- 5. IZVOZ PODATKOV ---
    st.subheader("5. Izvoz rezultatov")
    
    export_df = input_df.copy()
    export_df["Predicted_Tenure"] = prediction
    export_df["Selected_Model"] = selected_model_name
    
    csv = export_df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📥 Prenesi rezultat kot CSV",
        data=csv,
        file_name='tenure_prediction.csv',
        mime='text/csv',
    )

else:
    st.warning("Modeli niso naloženi. Preveri .pkl datoteke.")