import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, mannwhitneyu
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

df = pd.read_csv("../Telco-Customer-Churn.csv")
df = df.drop(columns=["customerID"])
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.loc[df["tenure"] == 0, "TotalCharges"] = 0
df["Churn_bin"] = df["Churn"].map({"No": 0, "Yes": 1})

numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
categorical_cols = df.select_dtypes(include="object").columns.drop("Churn")

results = []

for col in numeric_cols:
    group_yes = df[df["Churn"]=="Yes"][col]
    group_no = df[df["Churn"]=="No"][col]
    stat, p = mannwhitneyu(group_yes, group_no, alternative="two-sided")
    signif = p < 0.05
    p_display = round(p,3) if p >= 0.001 else "< 0.001"
    results.append({"Variable": col,"Type":"Numerical","Test":"Mann–Whitney U","P-value":p,"P-display":p_display,"Significant":signif})
    plt.figure()
    sns.boxplot(x="Churn", y=col, data=df)
    plt.title(f"{col} vs Churn")
    plt.show()

for col in categorical_cols:
    contingency = pd.crosstab(df[col], df["Churn"])
    chi2, p, _, _ = chi2_contingency(contingency)
    signif = p < 0.05
    p_display = round(p,3) if p >= 0.001 else "< 0.001"
    results.append({"Variable": col,"Type":"Categorical","Test":"Chi-square","P-value":p,"P-display":p_display,"Significant":signif})
    prop = pd.crosstab(df[col], df["Churn"], normalize="index")
    prop.plot(kind="bar", stacked=True)
    plt.title(f"{col} vs Churn")
    plt.ylabel("Proportion")
    plt.show()

bivariate_results = pd.DataFrame(results)
print("\nBIVARIATE ANALYSIS RESULTS")
print(bivariate_results.sort_values("P-value")[["Variable","Type","Test","P-display","Significant"]])

X = df.drop(columns=["Churn","Churn_bin"])
y = df["Churn_bin"]

cat_cols = X.select_dtypes(include="object").columns
num_cols = X.select_dtypes(exclude="object").columns

preprocessor = ColumnTransformer([
    ("num","passthrough",num_cols),
    ("cat",OneHotEncoder(drop="first"),cat_cols)
])

rf = Pipeline([
    ("prep", preprocessor),
    ("model", RandomForestClassifier(random_state=42))
])
rf.fit(X, y)
feature_names = rf.named_steps["prep"].get_feature_names_out()
rf_importance = rf.named_steps["model"].feature_importances_
rf_df = pd.DataFrame({"Feature":feature_names,"Importance":rf_importance}).sort_values("Importance",ascending=False)
print("\nTOP RANDOM FOREST FEATURES")
print(rf_df.head(15))

lasso = Pipeline([
    ("prep", preprocessor),
    ("model", LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000))
])
lasso.fit(X, y)
lasso_coef = lasso.named_steps["model"].coef_[0]
lasso_df = pd.DataFrame({"Feature":feature_names,"Coefficient":lasso_coef})
lasso_selected = lasso_df[lasso_df["Coefficient"] != 0]
print("\nLASSO SELECTED FEATURES")
print(lasso_selected)

X_transformed = preprocessor.fit_transform(X)
logreg = LogisticRegression(max_iter=5000, solver="lbfgs")
rfe = RFE(logreg, n_features_to_select=10)
rfe.fit(X_transformed, y)
rfe_features = feature_names[rfe.support_]
print("\nRFE SELECTED FEATURES")
print(rfe_features)

final_features = set(rf_df.head(15)["Feature"]).intersection(set(lasso_selected["Feature"]))
print("\nFINAL SELECTED FEATURES")
for f in final_features:
    print(f)
