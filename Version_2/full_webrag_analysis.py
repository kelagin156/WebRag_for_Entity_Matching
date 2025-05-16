
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
