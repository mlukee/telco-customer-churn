import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Branje csv datoteke
df = pd.read_csv("Telco-Customer-Churn-Cleaned.csv")


# ID stranke ne potrebujemo za analizo
customer_ids = df["customerID"]  
df = df.drop(columns=["customerID"]) 

# Ločimo numerične in kategorialne spremenljivke
num_cols = df.select_dtypes(include=[np.number]).columns
cat_cols = df.select_dtypes(exclude=[np.number]).columns

num_stats = df[num_cols].describe().T
num_stats["median"] = df[num_cols].median()
num_stats["Q1"] = df[num_cols].quantile(0.25)
num_stats["Q3"] = df[num_cols].quantile(0.75)

num_stats = num_stats.round(3)

num_stats["Mean ± SD"] = (
    num_stats["mean"].astype(str) + " ± " + num_stats["std"].astype(str)
)
num_stats["Median (Q1–Q3)"] = (
    num_stats["median"].astype(str)
    + " ("
    + num_stats["Q1"].astype(str)
    + "–"
    + num_stats["Q3"].astype(str)
    + ")"
)
num_stats["Min–Max"] = (
    num_stats["min"].astype(str) + "–" + num_stats["max"].astype(str)
)

summary_table = num_stats[["Mean ± SD", "Median (Q1–Q3)", "Min–Max"]]

# Kategorialne spremenljivke (n in %)
cat_stats = (
    df[cat_cols]
    .apply(lambda x: x.value_counts())
    .fillna(0)
    .astype(int)
)
cat_percent = df[cat_cols].apply(lambda x: x.value_counts(normalize=True) * 100).fillna(0)
cat_summary = cat_stats.astype(str) + " (" + cat_percent.round(1).astype(str) + "%)"

# Numerične
for col in num_cols:
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(df[col], kde=True, color="steelblue")
    plt.title(f"Histogram: {col}")
    plt.subplot(1, 2, 2)
    sns.boxplot(x=df[col], color="orange")
    plt.title(f"Boxplot: {col}")
    plt.tight_layout()
    plt.show()

# Kategorialne
for col in cat_cols:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=df[col], palette="pastel")
    plt.title(f"Barplot: {col}")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

print("\nNumerične spremenljivke:")
summary_table = num_stats[["Mean ± SD", "Median (Q1–Q3)", "Min–Max"]]
print(summary_table)
# display(num_stats[["mean", "std", "median", "Q1", "Q3", "min", "max"]]) TODO: Use in notebook

print("\nKategorialne spremenljivke (n in %):")
cat_summary_list = []
for col in cat_cols:
    counts = df[col].value_counts(dropna=False)
    percents = df[col].value_counts(normalize=True, dropna=False) * 100
    temp_df = pd.DataFrame({
        "Spremenljivka": col,
        "Kategorija": counts.index,
        "Število": counts.values,
        "%": percents.round(1).values
    })
    cat_summary_list.append(temp_df)

cat_summary_table = pd.concat(cat_summary_list, ignore_index=True)
print(cat_summary_table)
# print(cat_summary)
# display(cat_summary) Todo: Use in notebook