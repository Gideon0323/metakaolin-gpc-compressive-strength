"""
SVR Model for Predicting Compressive Strength of Bacteria-Induced Metakaolin-Based Geopolymer Concrete(GPC)

Author: (Chibuike Gideon's Project)
Description:
Trains a Support Vector Regression (SVR) model to predict compressive strength (MPa)
from mix design  and curing variables:
- NaOH molarity
- Activator_Ratio
- A_B_Ratio
- Bacteria(%)
- Temperature_Degree_Celsius
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance

DATA_FILE = "gpc_dataset.csv"

FEATURE_COLUMNS = [
    "NaOH_Molarity",
    "Activator_Ratio",
    "A_B_Ratio",
    "Bacteria(%)",
    "Temperature_Degree_Celsius",
   
]
TARGET_COLUMN = "Compressive_Strength_MPa"

def load_data():
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        print(f"Loaded real dataset from {DATA_FILE}: {df.shape[0]} rows")
    else:
        print(f" '{DATA_FILE}' not found")
        df = pd.DataFrame(rows)

    return df

def train_svr(X_train, y_train):
    pipeline= Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf"))
    ])
    param_grid = {
    "svr__C": [1,10,50,100,200],
    "svr__epsilon": [0.01,0.1,0.5,1],
    "svr__gamma": ["scale", "auto", 0.01,0.1,1],
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv,
    scoring="r2",
    n_jobs= -1,
    verbose=1,
    )
    grid.fit(X_train, y_train)
    print("\nBest parameters found:")
    print(grid.best_params_)
    print(f"Best CV R^2 score: {grid.best_score_:.4f}")
    return grid.best_estimator_

df = load_data()
df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
X = df[FEATURE_COLUMNS].values
y = df[TARGET_COLUMN].values
X_train,X_test, y_train, y_test = train_test_split(
    X,y, test_size=0.2, random_state=42
)
model = train_svr(X_train, y_train)

import matplotlib.pyplot as plt
fig, axes = plt.subplots(2,3, figsize=(15,8))
axes =axes.flatten()
for i, col in enumerate(FEATURE_COLUMNS):
    axes[i].scatter(df[col], df[TARGET_COLUMN], alpha=0.6)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("COmpressive Stregth (MPa)")
    axes[i].set_title(f"{col} vs Strength")

plt.tight_layout()
plt.savefig("scatter_diagnostics.png", dpi=150)
plt.show()

outliers=df[df[TARGET_COLUMN]<22.5].sort_values(TARGET_COLUMN)
print(outliers)

df_clean=df[df[TARGET_COLUMN]>=22.5]
print(df_clean[FEATURE_COLUMNS +[TARGET_COLUMN]].corr()[TARGET_COLUMN])

df = load_data()
df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
df_clean=df[df[TARGET_COLUMN] >=22.5]
X = df_clean[FEATURE_COLUMNS].values
y = df_clean[TARGET_COLUMN].values
X_train,X_test, y_train, y_test = train_test_split(
    X,y, test_size=0.2, random_state=42
)
model = train_svr(X_train, y_train)

from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
import numpy as np

y_pred= model.predict(X_test)

mae=mean_absolute_error(y_test,y_pred)
mse= mean_squared_error(y_test,y_pred)
rsme=np.sqrt(mse)
r2=r2_score(y_test,y_pred)
mape=np.mean(np.abs((y_test - y_pred)/ y_test))*100

print(f"R^2:{r2:.4f}")
print(f"MAE:{mae:.4f} MPa")
print(f"MSE:{mse:.4f} MPa")
print(f"RMSE:{np.sqrt(mse):.4f} MPa")
print(f"MAPE:{mape:.2f} %")

import pandas as pd
def get_metrics(y_true, y_pred, label):
    return{
        "Set": label,
        "R^2": r2_score(y_true,y_pred),
        "MAE": mean_absolute_error(y_true,y_pred),
        "MSE": mean_squared_error(y_true,y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true,y_pred)),
        "MAPE(%)": np.mean(np.abs((y_true - y_pred)/ y_true))*100
}
y_train_pred=model.predict(X_train)
y_test_pred=model.predict(X_test)

results=pd.DataFrame([
    get_metrics(y_train, y_train_pred, "Train"),
    get_metrics(y_test, y_test_pred, "Test"),
])
print(results)

import matplotlib.pyplot as plt
import numpy as np

y_test_pred=model.predict(X_test)
plt.figure(figsize=(7,7))
plt.scatter(y_test,y_test_pred, color="royalblue", edgecolor="k", s=70, alpha=0.8, label="Test specimens")
min_val=min(y_test.min(), y_test_pred.min()) - 0.5
max_val=min(y_test.max(), y_test_pred.max()) + 0.5
plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="1:1 line (perfect prediction)")

plt.xlabel("Actual Compressive Strength (MPa)", fontsize=12)
plt.ylabel("Predicted Compressive Strength (MPa)", fontsize=12)
plt.title("SVR Model: Predicted vs Actual Compressive Strength (Test Set)", fontsize=13)
plt.legend()
plt.grid(alpha=0.3)
plt.axis("equal")
plt.tight_layout()

plt.savefig("predicted_vs_actual.png", dpi=200)
print("Saved plot as predicted_vs_actual.png")
plt.show()

import numpy as np
import pandas as pd
import itertools

naoh_molarity_range=np.arange(8, 14.01,0.5)
activator_ratio_range=np.arange(1.5,2.51,0.1)
a_b_ratio_range=np.arange(0.4,0.51,0.1)
bacteria_range=np.arange(0,3.01,0.5)
temperature_range=np.arange(27,31.01,1.0)

FEATURE_COLUMNS = [
    "NaOH_Molarity",
    "Activator_Ratio",
    "A_B_Ratio",
    "Bacteria(%)",
    "Temperature_Degree_Celsius",
   
]
combinations=list(itertools.product(
    naoh_molarity_range,
    activator_ratio_range,
    a_b_ratio_range,
    bacteria_range,
    temperature_range,
))
grid_df=pd.DataFrame(combinations, columns=FEATURE_COLUMNS)
print(f"Total combinations to evaluate:{len(grid_df)}")
X_grid=grid_df[FEATURE_COLUMNS].values
predicted_strength=model.predict(X_grid)
grid_df["Predicted_Strength_MPa"]=predicted_strength

top_10=grid_df.sort_values("Predicted_Strength_MPa", ascending=False).head(10)
print("\n--- Top 10 Model-Predicted Mixes(Exploratory Only)---")
print(top_10.to_string(index=False))

best_mix=top_10.iloc[0]
print("\n--- Single Best Predicted Mix---")
for col in FEATURE_COLUMNS:
    print(f"{col}: {best_mix[col]}")
print(f"Predicted Compressive Strength:{best_mix['Predicted_Strength_MPa']:.2f} MPa")

top_10.to_csv("top_10_predicted_mixes.csv", index=False)
print("\nSaved top_10_predicted_mixes.csv")

result = permutation_importance(
    model, X_test, y_test,
    n_repeats=30, random_state=42, scoring="r2"
)
importances = pd.Series(result.importances_mean, index=FEATURE_COLUMNS)
importances = importances.sort_values(ascending=True)

print("--- Permutation Feature Importance (mean R2 drop) ---")
print(importances.sort_values(ascending=False))

plt.figure(figsize=(7,5))
importances.plot(kind="barh")
plt.xlabel("Mean decrease in R2 when feature is shuffled")
plt.title("Feature Importance (Permutation)")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=200)
plt.show()
print("Saved feature_importance.png")

scaler = model.named_steps["scaler"]

scaler_df = pd.DataFrame({
    "Feature": FEATURE_COLUMNS, 
    "Mean (mu)": scaler.mean_,
    "Std_Dev (sigma)": scaler.scale_,
})

scaler_df.to_csv("scaler_parameters.csv",index=False)
print(scaler_df)
print("Saved scaler_parameters.csv")
svr=model.named_steps["svr"]
support_vectors=svr.support_vectors_
dual_coefs=svr.dual_coef_[0]

n_sv = len(support_vectors)
sv_df=pd.DataFrame(support_vectors, columns=[f"{f}_scaled" for f in FEATURE_COLUMNS])
sv_df.insert(0, "Support_Vector_Index", range(1, n_sv + 1))
sv_df["Dual_Coefficient (alpha_i - alpha_i*)"] = dual_coefs

sv_df.to_csv("support_vectors.csv", index=False)
print(f"Saved {n_sv} support vectors + dual coefficients to support_vectors.csv")
print(sv_df.head())

intercept = svr.intercept_[0]
print(f"Intercept (b): {intercept:.6f}")