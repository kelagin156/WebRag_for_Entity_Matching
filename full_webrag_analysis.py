
import pandas as pd
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Daten laden
df = pd.read_csv("entity_matching_results_400.csv")
df["y_true"] = df["y_true"].astype(int)

# 1. PERFORMANCE-METRIKEN
print("=== F1-Scores ===")
f1_scores = {
    "Baseline": f1_score(df["y_true"], df["ChatGPT40-mini_baseline_y_pred"]),
    "WebRAG_n1": f1_score(df["y_true"], df["WebRag_y_pred_n1"]),
    "WebRAG_n3": f1_score(df["y_true"], df["WebRag_y_pred_n3"]),
    "WebRAG_n5": f1_score(df["y_true"], df["WebRag_y_pred_n5"]),
}
print(f1_scores)

print("\n=== Klassifikationsbericht (WebRAG n=5) ===")
print(classification_report(df["y_true"], df["WebRag_y_pred_n5"]))

print("\n=== Konfusionsmatrix (WebRAG n=5) ===")
cm = confusion_matrix(df["y_true"], df["WebRag_y_pred_n5"])
print(cm)

# 2. FEHLERANALYSE
def classify_error(row):
    if row["y_true"] == 1 and row["WebRag_y_pred_n5"] == 0:
        return "False Negative"
    elif row["y_true"] == 0 and row["WebRag_y_pred_n5"] == 1:
        return "False Positive"
    else:
        return "Correct"

df["Error_Type"] = df.apply(classify_error, axis=1)

def error_category(row):
    left = row["Entity1"].lower()
    right = row["Entity2"].lower()
    if "watch" in left or "watch" in right:
        return "Fashion/Accessories"
    elif "printer" in left or "toner" in left or "laser" in left:
        return "Office Supplies"
    elif "camera" in left or "lens" in left or "dslr" in left:
        return "Photography"
    elif "bag" in left or "backpack" in left or "sling" in left:
        return "Bags"
    elif "bike" in left or "bicycle" in left or "tube" in left:
        return "Cycling/Outdoor"
    elif len(left) < 100 or len(right) < 100:
        return "Sparse Info"
    else:
        return "Other/General"

df["Error_Category"] = df[df["Error_Type"] != "Correct"].apply(error_category, axis=1)
error_summary = df[df["Error_Type"] != "Correct"].groupby(["Error_Type", "Error_Category"]).size().unstack(fill_value=0)
print("\n=== Fehlerklassifikation ===")
print(error_summary)

# 3. KOSTENANALYSE
print("\n=== Kostenanalyse (Tokenverbrauch pro n) ===")
cost_df = pd.DataFrame({
    "n": [1, 3, 5],
    "F1": [f1_scores["WebRAG_n1"], f1_scores["WebRAG_n3"], f1_scores["WebRAG_n5"]],
    "Travily_Input": [
        df["Travily_Input_Tokens_n1"].mean(),
        df["Travily_Input_Tokens_n3"].mean(),
        df["Travily_Input_Tokens_n5"].mean(),
    ],
    "Travily_Output": [
        df["Travily_Output_Tokens_n1"].mean(),
        df["Travily_Output_Tokens_n3"].mean(),
        df["Travily_Output_Tokens_n5"].mean(),
    ],
    "GPT_Input": [
        df["WebRAG_1_GPT_Input_Tokens"].mean(),
        df["WebRAG_3_GPT_Input_Tokens"].mean(),
        df["WebRAG_5_GPT_Input_Tokens"].mean(),
    ],
    "GPT_Output": [
        df["WebRAG_1_GPT_Output_Tokens"].mean(),
        df["WebRAG_3_GPT_Output_Tokens"].mean(),
        df["WebRAG_5_GPT_Output_Tokens"].mean(),
    ]
})
print(cost_df)

# 4. WEBRAG HILFT ODER SCHADEN
df["WebRAG_helped"] = (df["WebRag_y_pred_n5"] == df["y_true"]) & (df["ChatGPT40-mini_baseline_y_pred"] != df["y_true"])
df["WebRAG_hurt"] = (df["WebRag_y_pred_n5"] != df["y_true"]) & (df["ChatGPT40-mini_baseline_y_pred"] == df["y_true"])

print("\n=== Wirkung von WebRAG (n=5) ===")
print("WebRAG verbessert Entscheidungen:", df["WebRAG_helped"].sum())
print("WebRAG verschlechtert Entscheidungen:", df["WebRAG_hurt"].sum())
print("Netto-Verbesserung:", df["WebRAG_helped"].sum() - df["WebRAG_hurt"].sum())

# 5. OPTIONAL: Visualisierung (Konfusionsmatrix)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Vorhersage")
plt.ylabel("Wahrheit")
plt.title("Konfusionsmatrix WebRAG (n=5)")
plt.show()
