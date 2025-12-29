import pandas as pd
import numpy as np

df = pd.read_csv("../Telco-Customer-Churn.csv")

print(df.info())
print(df.head())
print(df.describe(include="all"))

missing = df.isna().sum()
print("Manjkajoče vrednosti:\n", missing)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

df.loc[df["tenure"] == 0, "TotalCharges"] = 0

df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

print("Manjkajoče po pretvorbi:\n", df.isna().sum())

for col in df.columns:
    if df[col].dtype in ["float64", "int64"]:
        df[col].fillna(df[col].median(), inplace=True)
    else:
        df[col].fillna(df[col].mode()[0], inplace=True)

duplicates = df.duplicated().sum()
print("Število podvojenih vrstic:", duplicates)

df = df.drop_duplicates()

numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    print(f"{col} – število ekstremnih vrednosti: {outliers}")

    df[col] = np.clip(df[col], lower, upper)

print(df.info())
print(df.describe())

df.to_csv("../Telco-Customer-Churn-Cleaned.csv", index=False)



