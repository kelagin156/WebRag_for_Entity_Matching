import openai
import json
import pandas as pd
import re
import ast
import random


RESULTS_FILE = "entity_matching_results_400.csv"
OUTPUT_FILE = "error_analysis_structured_full.json"

# === EXTRACT STRUCTURED ERROR DATA ===
def extract_error_data(n, label_type, indices):
    examples = []
    results_df = pd.read_csv(RESULTS_FILE)
    for idx in indices:
        row = results_df.loc[idx]

        entry = {
            "label_type": label_type,
            "ground_truth": int(row["y_true"]),
            "baseline_prediction": int(row["ChatGPT40-mini_baseline_y_pred"]),
            "webrag_prediction": int(row[f"WebRag_y_pred_n{n}"]),
            "entity_1": row["Entity1"],
            "entity_2": row["Entity2"],
            "webrag_prompt": row[f"WebRAG_{n}_Prompt"],
            "webrag_response": row[f"WebRAG_{n}_Response"],
        }
        examples.append(entry)
    return examples

def generate_classification_prompt(example, error_classes):
    prompt = (
        "Given the following error classes for a product matching classification system, "
        "please classify the following product pair into all error classes by their number "
        "if they are relevant for this pair and its explanation. Please give a short explanation "
        "of every decision as a list first. Finally, also provide a confidence score for each classification "
        "adhering to the JSON format of the following example:\n\n"
        '{"2":"90","4":"30","5":"75"}\n\n'
        f"Error classes:\n {error_classes} \n" 
    )

    label_str = "Match" if example["ground_truth"] == 1 else "Non-Match"
    pred = "Match" if example["webrag_prediction"] == 1 else "Non-Match"

    prompt += "\nNow classify this pair:\n"
    prompt += f"Original Label: {label_str}\n"
    prompt += f"Predicted Label: {pred}\n"
    prompt += f"Entity 1: {example['entity_1']}\n"
    prompt += f"Entity 2: {example['entity_2']}\n"
    prompt += f"Entity Matching Prompt: {example['webrag_prompt']}\n"
    prompt += f"Entity Matching Response: {example['webrag_response']}\n"

    openai.api_key = "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"
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",  
        messages=[{"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"]

if __name__ == "__main__":
    
    # Load your results CSV
    df = pd.read_csv("entity_matching_results_400.csv")

    # Get row indices for False Positives: predicted match but actually non-match
    false_positives1 = df[(df["WebRag_y_pred_n1"] == 1) & (df["y_true"] == 0)].index.tolist()
    # Get row indices for False Negatives: predicted non-match but actually match
    false_negatives1 = df[(df["WebRag_y_pred_n1"] == 0) & (df["y_true"] == 1)].index.tolist()

    # Get row indices for False Positives: predicted match but actually non-match
    false_positives3 = df[(df["WebRag_y_pred_n3"] == 1) & (df["y_true"] == 0)].index.tolist()
    # Get row indices for False Negatives: predicted non-match but actually match
    false_negatives3 = df[(df["WebRag_y_pred_n3"] == 0) & (df["y_true"] == 1)].index.tolist()

    # Get row indices for False Positives: predicted match but actually non-match
    false_positives5 = df[(df["WebRag_y_pred_n5"] == 1) & (df["y_true"] == 0)].index.tolist()
    # Get row indices for False Negatives: predicted non-match but actually match
    false_negatives5 = df[(df["WebRag_y_pred_n5"] == 0) & (df["y_true"] == 1)].index.tolist()

    # Print them or use them programmatically
    print("False Positives for n=1:", false_positives1)
    print("False Negatives for n=1:", false_negatives1)

    # Print them or use them programmatically
    print("False Positives for n=3:", false_positives3)
    print("False Negatives for n=3:", false_negatives3)

    # Print them or use them programmatically
    print("False Positives for n=5:", false_positives5)
    print("False Negatives for n=5:", false_negatives5)

    # === BUILD STRUCTURED SET ===
    fp_data1 = extract_error_data(1, "False Positive", false_positives1)
    fn_data1 = extract_error_data(1, "False Negative", false_negatives1)
    fp_data1_subsampled = random.sample(fp_data1, 15)
    fn_data1_subsampled = random.sample(fn_data1, 15)

    fp_data3 = extract_error_data(3, "False Positive", false_positives3)
    fn_data3 = extract_error_data(3, "False Negative", false_negatives3)
    fp_data3_subsampled = random.sample(fp_data3, 15)
    fn_data3_subsampled = random.sample(fn_data3, 15)

    fp_data5 = extract_error_data(5, "False Positive", false_positives5)
    fn_data5 = extract_error_data(5, "False Negative", false_negatives5)
    fp_data5_subsampled = random.sample(fp_data5, 15)
    fn_data5_subsampled = random.sample(fn_data5, 15)
    all_errors = fp_data1_subsampled  + fp_data3_subsampled  +fp_data5_subsampled  + fn_data1_subsampled  + fn_data3_subsampled  + fn_data5_subsampled 

    # === SAVE TO FILE ===
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_errors, f, indent=2)

    print(f"Saved {len(all_errors)} structured error cases to {OUTPUT_FILE}")

    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        errors = json.load(f)

    # Separate FPs and FNs
    false_negatives = [e for e in errors if e["label_type"] == "False Negative"]
    false_positives = [e for e in errors if e["label_type"] == "False Positive"]

    def format_block(title, entries):
        formatted = [f"### {title}:\n"]
        for i, ex in enumerate(entries, 1):
            block = f"{title[:2]}{i}\n"
            block += f"Entity 1: {ex['entity_1']}\n"
            block += f"Entity 2: {ex['entity_2']}\n"
            block += f"Explanation: {ex['webrag_response']}\n"
            formatted.append(block)
        return "\n".join(formatted)

    # Build full prompt
    prompt = (
        "The following list contains false positive and false negative product pairs "
        "from the output of a product matching classification system.\n\n"
        "Given the product pairs and the associated explanations, come up with a set of "
        "error classes, separately for both false positives and false negatives, that "
        "explain why the classification system fails on these examples.\n\n"
    )

    prompt += format_block("False Negatives", false_negatives)
    prompt += "\n\n"
    prompt += format_block("False Positives", false_positives)

    # Output to file for pasting into ChatGPT or Claude etc.
    with open("gpt_error_class_prompt.txt", "w", encoding="utf-8") as f:
        f.write(prompt)

    openai.api_key = "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"
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",  
        messages=[{"role": "user", "content": prompt}]
    )

    # Output the GPT response
    print("\n--- GPT-Generated Error Classes ---\n")
    with open("gpt_error_class_output.txt", "w", encoding="utf-8") as f:
        f.write(response["choices"][0]["message"]["content"])
    print(response["choices"][0]["message"]["content"])
    error_classes = response["choices"][0]["message"]["content"]
    error_classes_list = re.findall(r"\d+\.\s+\*\*(.*?)\*\*", error_classes)
    print(error_classes_list)
    amount_of_errors = int(len(error_classes_list)/2)
    print(amount_of_errors)
    error_classes_list_fp = error_classes_list[-amount_of_errors:]
    error_classes_list_fn = error_classes_list[:amount_of_errors]

    false_positive_dic = {error: 0 for error in error_classes_list_fp}
    false_negative_dic = {error: 0 for error in error_classes_list_fn}

    with open(OUTPUT_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)
        for i, record in enumerate(data):
            if record["label_type"] == "False Positive":
                response = generate_classification_prompt(record, error_classes_list_fp)
            if record["label_type"] == "False Negative":
                response = generate_classification_prompt(record, error_classes_list_fn)
            match = re.search(r"\{.*?\}", response, re.DOTALL)
            if match:
                json_str = match.group(0)
                # Remove JavaScript-style comments (//...)
                cleaned_str = re.sub(r"//.*?$", "", json_str, flags=re.MULTILINE)

                try:
                    score_dict = json.loads(cleaned_str)
                except json.JSONDecodeError:
                    try:
                        # As fallback, use ast.literal_eval (safe eval of Python dict-like string)
                        score_dict = ast.literal_eval(cleaned_str)
                    except Exception as e:
                        print(f"Failed to parse JSON: {e}")
                        score_dict = {}

                # Step 3: Loop through and update based on score threshold
                for key, value in score_dict.items():
                    index = int(re.match(r"(\d+)", key).group(0))-1
                    score = int(re.match(r"(\d+)", value).group(0))
                    if int(value) > 0:
                        if record["label_type"] == "False Positive":
                            if index < len(error_classes_list_fp):  # Check index bounds
                                class_name_fp = error_classes_list_fp[index]
                                false_positive_dic[class_name_fp] += 1

                        if record["label_type"] == "False Negative":
                            if index < len(error_classes_list_fn):  # Check index bounds
                                class_name_fn = error_classes_list_fn[index]
                                false_negative_dic[class_name_fn] += 1

                #print("False Positives:", false_positive_dic)
                #print("False Negatives:", false_negative_dic)
            else:
                print("No JSON found in string.")
        
            
            with open("error_to_errorClasses.txt", "a", encoding="utf-8") as file:
                file.write("For record: ")
                file.write(str(record))
                file.write("\n GPT ")
                file.write(response)
                file.write("\n")

        with open("errors_dicts.txt", "w", encoding="utf-8") as file:
            file.write("False Positives: " + str(false_positive_dic) + "\n")
            file.write("False Negatives: " + str(false_negative_dic) + "\n")
