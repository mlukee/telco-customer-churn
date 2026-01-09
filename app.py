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
            "Gradient Boosting Regression": joblib.load("./modeli_regresija/best_model_GradientBoosting.pkl"),
            "Random Forest Regression": joblib.load("./modeli_regresija/best_model_RandomForest.pkl"),
            "Ridge Regression": joblib.load("./modeli_regresija/best_model_Ridge.pkl"),
            "Extra Trees Classifier": joblib.load("./modeli_klasifikacija/ExtraTrees_best_model.pkl"),
            "Random Forest Classifier": joblib.load("./modeli_klasifikacija/RandomForest_best_model.pkl"),
            "MLP Classifier": joblib.load("./modeli_klasifikacija/MLP_best_model.pkl")
        }
        return models
    except FileNotFoundError:
        st.error("Modeli niso najdeni! Prepričaj se, da so .pkl datoteke v isti mapi.")
        return {}

models = load_models()

# Seznam značilk, ki jih pričakujejo REGRESIJSKI modeli
expected_features_reg = [
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

# Seznam značilk, ki jih pričakujejo KLASIFIKACIJSKI modeli
expected_features_clf = [
    'tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen',
    'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year',
    'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No',
    'TechSupport_No', 'TechSupport_Yes',
    'OnlineSecurity_No', 'OnlineSecurity_Yes',
    'OnlineBackup_No', 'OnlineBackup_Yes',
    'DeviceProtection_No', 'DeviceProtection_Yes',
    'PaperlessBilling_No', 'PaperlessBilling_Yes',
    'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
]

# --- 2. STRANSKA VRSTICA (VNOS PODATKOV) ---
st.sidebar.header("1. Izbira modela")
selected_model_name = st.sidebar.selectbox("Izberi algoritem", list(models.keys()))
model = models[selected_model_name] if models else None

st.sidebar.header("2. Podatki o stranki")

# Določimo, katere značilke in vnosna polja potrebujemo glede na tip modela
is_classifier = "Classifier" in selected_model_name
expected_features = expected_features_clf if is_classifier else expected_features_reg

# Vnosna polja (UI) - pretvorili jih bomo v format za model
def user_input_features():
    # --- PRETVORBA V FORMAT MODELA (Mapping) ---
    if is_classifier:
        # --- VNOSI ZA KLASIFIKATOR (surove vrednosti) ---
        st.sidebar.markdown("---")
        st.sidebar.subheader("Demografski podatki")
        gender = st.sidebar.selectbox("Spol (gender)", ["Male", "Female"])
        senior_citizen = st.sidebar.checkbox("Je upokojenec (SeniorCitizen)?", value=False)
        partner = st.sidebar.checkbox("Ima partnerja (Partner)?", value=True)
        dependents = st.sidebar.checkbox("Ima vzdrževane člane (Dependents)?", value=False)

        st.sidebar.markdown("---")
        st.sidebar.subheader("Podatki o naročnini")
        tenure = st.sidebar.slider("Trenutna doba (meseci)", 0, 72, 12)
        contract = st.sidebar.selectbox("Vrsta pogodbe (Contract)", ["Month-to-month", "One year", "Two year"])
        monthly_charges = st.sidebar.slider("Mesečni strošek ($)", min_value=18.0, max_value=120.0, value=70.0, step=0.5)
        total_charges = st.sidebar.number_input("Skupni stroški ($)", min_value=0.0, value=float(monthly_charges * tenure), step=50.0)
        paperless_billing = st.sidebar.selectbox("Brezpapirno poslovanje (PaperlessBilling)", ["Yes", "No"])
        payment_method = st.sidebar.selectbox("Način plačila", ["Electronic check", "Mailed check", "Credit card (automatic)", "Bank transfer (automatic)"])

        st.sidebar.markdown("---")
        st.sidebar.subheader("Telekomunikacijske storitve")
        phone_service = st.sidebar.selectbox("Telefonska storitev (PhoneService)", ["Yes", "No"])
        multiple_lines = st.sidebar.selectbox("Več telefonskih linij (MultipleLines)", ["No", "Yes", "No phone service"])
        internet_service = st.sidebar.selectbox("Internetna storitev (InternetService)", ["DSL", "Fiber optic", "No"])

        # Pogojni prikazi glede na internetno storitev
        if internet_service != "No":
            online_security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
            online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            device_protection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            tech_support = st.sidebar.selectbox("Tehnična podpora (TechSupport)", ["Yes", "No", "No internet service"])
            streaming_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            streaming_movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        else:
            # Če ni interneta, so te storitve nedostopne
            no_internet_service = "No internet service"
            online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies = (no_internet_service,) * 6


        # --- ZGRADIMO DataFrame s surovimi vrednostmi ---
        data = {
            'gender': [gender],
            'SeniorCitizen': [1 if senior_citizen else 0],
            'Partner': [ "Yes" if partner else "No"],
            'Dependents': ["Yes" if dependents else "No"],
            'tenure': [tenure],
            'PhoneService': [phone_service],
            'MultipleLines': [multiple_lines],
            'InternetService': [internet_service],
            'OnlineSecurity': [online_security],
            'OnlineBackup': [online_backup],
            'DeviceProtection': [device_protection],
            'TechSupport': [tech_support],
            'StreamingTV': [streaming_tv],
            'StreamingMovies': [streaming_movies],
            'Contract': [contract],
            'PaperlessBilling': [paperless_billing],
            'PaymentMethod': [payment_method],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges]
        }

        features = pd.DataFrame(data)
        return features, contract, monthly_charges

    else:
        # --- VNOSI ZA REGRESOR (obstoječa logika) ---
        data = {feature: 0 for feature in expected_features_reg}
        contract = st.sidebar.selectbox("Vrsta pogodbe (Contract)", ["Month-to-month", "One year", "Two year"])
        monthly_charges = st.sidebar.slider("Mesečni strošek ($)", min_value=18.0, max_value=120.0, value=70.0, step=0.5)
        col1, col2 = st.sidebar.columns(2)
        with col1:
            partner = st.checkbox("Ima partnerja?", value=True)
            multiple_lines = st.checkbox("Več linij?", value=False)
            online_security = st.checkbox("Online Security?", value=False)
        with col2:
            online_backup = st.checkbox("Online Backup?", value=False)
            device_protection = st.checkbox("Device Protection?", value=False)
        payment_method = st.sidebar.selectbox("Način plačila", ["Electronic check", "Mailed check", "Credit card/Bank transfer"])
        churn_status = st.sidebar.radio("Status stranke (Churn)", ["Še je stranka (No)", "Je odšla (Yes)"], index=0)

        # --- MAPIRANJE ZA REGRESOR ---
        data['MonthlyCharges'] = monthly_charges
        if contract == "One year":
            data['Contract_One year'] = 1
        elif contract == "Two year":
            data['Contract_Two year'] = 1
        data['MultipleLines_Yes'] = 1 if multiple_lines else 0
        data['Partner_Yes'] = 1 if partner else 0
        data['OnlineBackup_Yes'] = 1 if online_backup else 0
        data['OnlineSecurity_Yes'] = 1 if online_security else 0
        data['DeviceProtection_Yes'] = 1 if device_protection else 0
        data['Churn_Yes'] = 1 if churn_status == "Je odšla (Yes)" else 0
        if payment_method == "Mailed check":
            data['PaymentMethod_Mailed check'] = 1
        elif payment_method == "Electronic check":
            data['PaymentMethod_Electronic check'] = 1

        features = pd.DataFrame(data, index=[0])[expected_features_reg]
        return features, contract, monthly_charges

input_df, raw_contract, raw_charges = user_input_features()

# --- 3. GLAVNI DEL: PRIKAZ NAPOVEDI ---

st.subheader("3. Rezultat napovedi")

if model:
    # Za klasifikatorje želimo verjetnost za "Yes" (razred 1)
    if is_classifier:
        # predict_proba vrne verjetnosti za vsak razred, nas zanima drugi (index 1)
        prediction_proba = model.predict_proba(input_df)[0][1]
        prediction = prediction_proba # Uporabimo verjetnost kot glavno vrednost

        # Prikaz velike številke
        col_pred1, col_pred2 = st.columns([1, 2])

        with col_pred1:
            st.metric(label="Verjetnost odhoda (Churn)", value=f"{prediction:.2%}")

            # Interpretacija
            if prediction > 0.7:
                st.error("Zelo visoko tveganje za odhod")
            elif prediction > 0.4:
                st.warning("Povišano tveganje za odhod")
            else:
                st.success("Nizko tveganje za odhod")

        with col_pred2:
            # Prikaz v obliki "gauge" grafa
            fig, ax = plt.subplots(figsize=(6, 1.5))
            ax.barh(["Verjetnost"], [prediction], color=["#FF4B4B" if prediction > 0.5 else "#4CAF50"])
            ax.set_xlim(0, 1)
            ax.set_xlabel("Verjetnost")
            st.pyplot(fig)

    else: # Logika za regresijske modele
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
            # Prikaz relativno glede na povprečje
            avg_tenure = 32.4
            fig, ax = plt.subplots(figsize=(6, 1.5))
            ax.barh(["Povprečje", "Napoved"], [avg_tenure, prediction], color=["lightgray", "#4CAF50"])
            ax.set_xlim(0, 75)
            ax.set_xlabel("Meseci")
            st.pyplot(fig)

    # --- 4. SIMULACIJA OPTIMIZACIJE (prilagojeno za oba tipa) ---
    st.subheader("4. Simulacija optimizacije: Vpliv spremenljivke na rezultat")
    st.markdown("Kaj se zgodi, če spremenimo eno vrednost, ostale pa pustimo enake?")

    # Izbira spremenljivke za simulacijo
    sim_feature = st.selectbox("Izberi spremenljivko za simulacijo",
                               ['MonthlyCharges', 'TotalCharges'] if is_classifier else ['MonthlyCharges'])

    # Generiramo razpon vrednosti za simulacijo
    if sim_feature == 'MonthlyCharges':
        # Fiksni razpon za ceno
        sim_values = np.linspace(0, 125, 100)
    else:
        # Dinamični razpon za druge spremenljivke
        min_val = input_df[sim_feature].iloc[0] * 0.5
        max_val = input_df[sim_feature].iloc[0] * 1.5
        sim_values = np.linspace(15, 9000, 300)

    sim_preds = []
    temp_df = input_df.copy()

    for v in sim_values:
        temp_df[sim_feature] = v
        if is_classifier:
            sim_preds.append(model.predict_proba(temp_df)[0][1])
        else:
            sim_preds.append(model.predict(temp_df)[0])

    # Graf
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sns.lineplot(x=sim_values, y=sim_preds, ax=ax2, color="blue", linewidth=2)

    # Dodamo točko za trenutni izbor
    current_val = input_df[sim_feature].iloc[0]
    ax2.scatter([current_val], [prediction], color="red", s=100, label="Trenutni izbor", zorder=5)

    ax2.set_xlabel(f"Vrednost za '{sim_feature}'")
    ax2.set_ylabel("Napovedan rezultat (Verjetnost ali Tenure)")
    ax2.set_title(f"Kako '{sim_feature}' vpliva na rezultat")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Nastavimo meje x-osi, če gre za MonthlyCharges
    # if sim_feature == 'MonthlyCharges':
    #     ax2.set_xlim(0, 125)

    st.pyplot(fig2)

    # --- 5. IZVOZ PODATKOV ---
    st.subheader("5. Izvoz rezultatov")

    export_df = input_df.copy()
    if is_classifier:
        export_df["Predicted_Churn_Probability"] = prediction
    else:
        export_df["Predicted_Tenure"] = prediction
    export_df["Selected_Model"] = selected_model_name

    csv = export_df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="📥 Prenesi rezultat kot CSV",
        data=csv,
        file_name='prediction_result.csv',
        mime='text/csv',
    )

else:
    st.warning("Modeli niso naloženi. Preveri .pkl datoteke.")