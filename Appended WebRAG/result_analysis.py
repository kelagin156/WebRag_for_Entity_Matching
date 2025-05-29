from sklearn.metrics import f1_score, classification_report, confusion_matrix
import pandas as pd

# Load CSV file
df = pd.read_csv("entity_matching_results_400_version2.csv")
df["y_true"] = df["y_true"].astype(int)

# Calculate F1 Scores
print("=== F1 Scores ===")
f1_scores = {
    "Baseline": f1_score(df["y_true"], df["ChatGPT40-mini_baseline_y_pred"]),
    "WebRAG_n1": f1_score(df["y_true"], df["WebRag_y_pred_n1"]),
    "WebRAG_n3": f1_score(df["y_true"], df["WebRag_y_pred_n3"]),
    "WebRAG_n5": f1_score(df["y_true"], df["WebRag_y_pred_n5"]),
}
print(f1_scores)

# Classification reports and confusion matrices
print("\n=== Classification Report (Baseline) ===")
print(classification_report(df["y_true"], df["ChatGPT40-mini_baseline_y_pred"], digits=4))

print("\n=== Classification Report (WebRAG n=1) ===")
print(classification_report(df["y_true"], df["WebRag_y_pred_n1"], digits=4))

print("\n=== Confusion Matrix (WebRAG n=1) ===")
cm = confusion_matrix(df["y_true"], df["WebRag_y_pred_n1"])
print(cm)

print("\n=== Classification Report (WebRAG n=3) ===")
print(classification_report(df["y_true"], df["WebRag_y_pred_n3"], digits=4))

print("\n=== Confusion Matrix (WebRAG n=3) ===")
cm = confusion_matrix(df["y_true"], df["WebRag_y_pred_n3"])
print(cm)

print("\n=== Classification Report (WebRAG n=5) ===")
print(classification_report(df["y_true"], df["WebRag_y_pred_n5"], digits=4))

print("\n=== Confusion Matrix (WebRAG n=5) ===")
cm = confusion_matrix(df["y_true"], df["WebRag_y_pred_n5"])
print(cm)

# GPT-4o mini pricing (USD per 1,000 tokens)
GPT_INPUT_COST = 0.00015
GPT_OUTPUT_COST = 0.00060

# Travily credit pricing (assumed from Bootstrap Plan)
TRAVILY_CREDIT_COST = 0.0067

# Assumption: 2 searches + up to 10 extracts per sample = 4 credits
TRAVILY_CREDITS_PER_SAMPLE = 2
TRAVILY_COST_PER_SAMPLE = TRAVILY_CREDIT_COST * TRAVILY_CREDITS_PER_SAMPLE

# Cost calculation per sample
def calculate_total_cost(input_tokens, output_tokens):
    gpt_cost = (input_tokens / 1000) * GPT_INPUT_COST + (output_tokens / 1000) * GPT_OUTPUT_COST
    travily_cost = TRAVILY_COST_PER_SAMPLE
    return gpt_cost, travily_cost

# Baseline
df[f"Total_Cost_GPT_USD_baseline"],df[f"Total_Cost_Travily_USD_baseline"]  = calculate_total_cost(
        df[f"Baseline_Input_Tokens"],
        df[f"Baseline_Output_Tokens"]  # Only GPT output counts toward output cost
    )

# Compute for n = 1, 3, 5
for n in [1, 3, 5]:
    df[f"Total_Input_Tokens_n{n}"] = df[f"Travily_Input_Tokens_n{n}"] + df[f"WebRAG_{n}_GPT_Input_Tokens"]
    df[f"Total_Output_Tokens_n{n}"] = df[f"Travily_Output_Tokens_n{n}"] + df[f"WebRAG_{n}_GPT_Output_Tokens"]
    
    df[f"Total_Cost_GPT_USD_n{n}"],df[f"Total_Cost_Travily_USD_n{n}"]  = calculate_total_cost(
        df[f"Total_Input_Tokens_n{n}"],
        df[f"WebRAG_{n}_GPT_Output_Tokens"],  # Only GPT output counts toward output cost
    )

# Summary
cost_summary = pd.DataFrame({
    "n": [0, 1, 3, 5],
    "F1": [
        f1_score(df["y_true"], df["ChatGPT40-mini_baseline_y_pred"]),
        f1_score(df["y_true"], df["WebRag_y_pred_n1"]),
        f1_score(df["y_true"], df["WebRag_y_pred_n3"]),
        f1_score(df["y_true"], df["WebRag_y_pred_n5"]),
    ],
    "Avg. Input Tokens": [
        df["Baseline_Input_Tokens"].mean(),
        df["Total_Input_Tokens_n1"].mean(),
        df["Total_Input_Tokens_n3"].mean(),
        df["Total_Input_Tokens_n5"].mean(),
    ],
    "Avg. Output Tokens": [
        df["Baseline_Output_Tokens"].mean(),
        df["Total_Output_Tokens_n1"].mean(),
        df["Total_Output_Tokens_n3"].mean(),
        df["Total_Output_Tokens_n5"].mean(),
    ],
    "Avg. GPT Cost": [
        df["Total_Cost_GPT_USD_baseline"].mean(),
        df["Total_Cost_GPT_USD_n1"].mean(),
        df["Total_Cost_GPT_USD_n3"].mean(),
        df["Total_Cost_GPT_USD_n5"].mean(),
    ],
    "Avg. Travily Cost": [
        0,
        df["Total_Cost_Travily_USD_n1"].mean(),
        df["Total_Cost_Travily_USD_n3"].mean(),
        df["Total_Cost_Travily_USD_n5"].mean(),
    ],
    "Avg. Total Cost": [
        0,
        df["Total_Cost_Travily_USD_n1"].mean() + df["Total_Cost_GPT_USD_n1"].mean(),
        df["Total_Cost_Travily_USD_n3"].mean() + df["Total_Cost_GPT_USD_n3"].mean(),
        df["Total_Cost_Travily_USD_n5"].mean() + df["Total_Cost_GPT_USD_n5"].mean(),
    ]
})

print("\n Cost Analysis (GPT-4o mini + estimated Travily costs):")
print(cost_summary)

# F1 scores
f1_n = {
    1: f1_score(df["y_true"], df["WebRag_y_pred_n1"]),
    3: f1_score(df["y_true"], df["WebRag_y_pred_n3"]),
    5: f1_score(df["y_true"], df["WebRag_y_pred_n5"]),
}

# Calculate GPT costs
cost_data = {}
for n in [1, 3, 5]:
    input_tokens = df[f"Travily_Input_Tokens_n{n}"] + df[f"WebRAG_{n}_GPT_Input_Tokens"]
    output_tokens = df[f"WebRAG_{n}_GPT_Output_Tokens"]

    gpt_cost = (input_tokens / 1000) * GPT_INPUT_COST + (output_tokens / 1000) * GPT_OUTPUT_COST
    total_cost = gpt_cost.sum() + 400 * TRAVILY_COST_PER_SAMPLE

    cost_data[n] = {
        "total_cost_with_travily": total_cost,
        "f1": f1_n[n]
    }

print(pd.DataFrame(cost_data))
# Calculate cost per F1 point
def cost_delta(n):
    extra_f1 = cost_data[n]["f1"] - 0.15730337078651685
    cost = cost_data[n]["total_cost_with_travily"] - 0.021935

    return {
        "Cost per F1 Point": cost/ (extra_f1*100),

    }

# Output as DataFrame
results = pd.DataFrame([
    cost_delta(1),
    cost_delta(3),
    cost_delta(5)
])

print("\n Cost analysis including realistic Travily credits:")
print(results)

# Did WebRAG help or hurt compared to baseline?
df["WebRAG_helped1"] = (df["WebRag_y_pred_n1"] == df["y_true"]) & (df["ChatGPT40-mini_baseline_y_pred"] != df["y_true"])
df["WebRAG_hurt1"] = (df["WebRag_y_pred_n1"] != df["y_true"]) & (df["ChatGPT40-mini_baseline_y_pred"] == df["y_true"])
df["WebRAG_helped3"] = (df["WebRag_y_pred_n3"] == df["y_true"]) & (df["ChatGPT40-mini_baseline_y_pred"] != df["y_true"])
df["WebRAG_hurt3"] = (df["WebRag_y_pred_n3"] != df["y_true"]) & (df["ChatGPT40-mini_baseline_y_pred"] == df["y_true"])
df["WebRAG_helped5"] = (df["WebRag_y_pred_n5"] == df["y_true"]) & (df["ChatGPT40-mini_baseline_y_pred"] != df["y_true"])
df["WebRAG_hurt5"] = (df["WebRag_y_pred_n5"] != df["y_true"]) & (df["ChatGPT40-mini_baseline_y_pred"] == df["y_true"])

print("\n=== WebRAG Impact (n=1) ===")
print("WebRAG improved decision:", df["WebRAG_helped1"].sum())
print("WebRAG worsened decision:", df["WebRAG_hurt1"].sum())
print("Net improvement:", df["WebRAG_helped1"].sum() - df["WebRAG_hurt1"].sum())

print("\n=== WebRAG Impact (n=3) ===")
print("WebRAG improved decision:", df["WebRAG_helped3"].sum())
print("WebRAG worsened decision:", df["WebRAG_hurt3"].sum())
print("Net improvement:", df["WebRAG_helped3"].sum() - df["WebRAG_hurt3"].sum())

print("\n=== WebRAG Impact (n=5) ===")
print("WebRAG improved decision:", df["WebRAG_helped5"].sum())
print("WebRAG worsened decision:", df["WebRAG_hurt5"].sum())
print("Net improvement:", df["WebRAG_helped5"].sum() - df["WebRAG_hurt5"].sum())

