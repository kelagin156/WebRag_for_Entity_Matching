import openai
from tavily import TavilyClient
import json
from sklearn.metrics import f1_score
import pandas as pd
from datetime import datetime

df_all = pd.read_csv("entity_matching_results_400.csv")
baseline_f1 = f1_score(df_all["y_true"], df_all["ChatGPT40-mini_baseline_y_pred"])
webrag_f1 = f1_score(df_all["y_true"], df_all["WebRag_y_pred"])

print("Baseline F1 score: ", baseline_f1)
print("WebRag F1 Score: ", webrag_f1)

df_f1 = pd.DataFrame([{"Date-Time": datetime.now(), "Baseline_F1": baseline_f1, "WebRag_F1": webrag_f1}])
df_f1.to_csv("f1_results.csv", index=False, encoding="utf-8", mode="a", header=not pd.io.common.file_exists("f1_results.csv"))

print("GPT was correct:", (df_all["y_true"] == df_all["ChatGPT40-mini_baseline_y_pred"]).sum())
print("Webrag was correct:", (df_all["y_true"] == df_all["WebRag_y_pred"]).sum())
print("Webrag help with right labeling:", ((df_all["y_true"] == df_all["WebRag_y_pred"]) & (df_all["y_true"] != df_all["ChatGPT40-mini_baseline_y_pred"])).sum())
print("Webrag destroyed right labeling:", ((df_all["y_true"] == df_all["ChatGPT40-mini_baseline_y_pred"]) & (df_all["y_true"] != df_all["WebRag_y_pred"])).sum())