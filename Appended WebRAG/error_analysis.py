import time
import openai
import json
import pandas as pd
import re
import ast
from openai.error import APIError, Timeout, RateLimitError, ServiceUnavailableError

# === CONFIGURATION ===
OUTPUT_FILE = "error_analysis_structured_full.json"

# === EXTRACT STRUCTURED ERROR DATA ===
def extract_error_data(n, label_type, indices):
    examples = []
    results_df = pd.read_csv( "entity_matching_results_400_version2.csv")
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
            "Which n": n
        }
        examples.append(entry)
    return examples

# === GENERATE CLASSIFICATION PROMPT ===
def generate_classification_prompt(example, error_classes, retries=3, delay=5):
    prompt = (
        "Given the following error classes for a product matching classification system, "
        "please classify the following product pair into all error classes by their number "
        "if they are relevant for this pair and its explanation. Please give a short explanation "
        "of every decision as a list first. Finally, also provide a confidence score for each classification "
        "adhering to the JSON format of the following example:\n\n"
        '{"2":"90","4":"30","5":"75"}\n\n'
        "Please make sure that you return a json and the key in the json matches the ID of the Error!"
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

    # Retry logic for OpenAI API calls in case it fails
    for attempt in range(1, retries + 1):
        try:
            openai.api_key = "YOUR API KEY HERE"  # TODO: Replace with your OpenAI API key 
            response = openai.ChatCompletion.create(
                model="gpt-4o",  
                messages=[{"role": "user", "content": prompt}]
            )
            return response["choices"][0]["message"]["content"]
        except (openai.APIError, Timeout, RateLimitError, ServiceUnavailableError) as e:
            print(f"[Attempt {attempt}] OpenAI API error: {e}")
            if attempt < retries:
                time.sleep(delay)
            else:
                print(" All retries failed. Exiting.")
                raise e  
            
# === MAIN FUNCTION TO SAVE ERRORS ===
def save_errors():
    # Load your results CSV
    df = pd.read_csv("entity_matching_results_400_version2.csv")

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

    fp_data3 = extract_error_data(3, "False Positive", false_positives3)
    fn_data3 = extract_error_data(3, "False Negative", false_negatives3)

    fp_data5 = extract_error_data(5, "False Positive", false_positives5)
    fn_data5 = extract_error_data(5, "False Negative", false_negatives5)

    all_errors = fp_data1  + fp_data3  +fp_data5  + fn_data1 + fn_data3 + fn_data5

    # === SAVE TO FILE ===
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_errors, f, indent=2)

    print(f"Saved {len(all_errors)} structured error cases to {OUTPUT_FILE}")

# === MAIN EXECUTION ===
if __name__ == "__main__":
    save_errors()

    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        errors = json.load(f)

    # Separate FPs and FNs
    false_negatives = [e for e in errors if e["label_type"] == "False Negative"]
    false_positives = [e for e in errors if e["label_type"] == "False Positive"]

    # Format the prompt for GPT
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

    with open("gpt_error_class_prompt.txt", "w", encoding="utf-8") as f:
        f.write(prompt)

    openai.api_key = "YOUR API KEY HERE"  # TODO: Replace with your OpenAI API key
    response = openai.ChatCompletion.create(
        model="gpt-4.1-mini",  
        messages=[{"role": "user", "content": prompt}]
    )

    # Output the GPT response
    print("\n--- GPT-Generated Error Classes ---\n")
    with open("gpt_error_class_output.txt", "w", encoding="utf-8") as f:
        f.write(response["choices"][0]["message"]["content"])
    print(response["choices"][0]["message"]["content"])

    # Extract error classes from the response
    error_classes = response["choices"][0]["message"]["content"]
    # Use regex to find all error classes in the format "1. **Error Class Name**"
    error_classes_list = re.findall(r"\d+\.\s+\*\*(.*?)\*\*", error_classes)

    # Print the number of error classes and the list to make sure
    amount_of_errors = int(len(error_classes_list)/2)
    print(f"Amount of errors: {amount_of_errors}")
    print(f"Error classes list: {error_classes_list}")

    # Split the error classes into False Positive and False Negative lists
    error_classes_list_fp = error_classes_list[-amount_of_errors:]
    error_classes_list_fn = error_classes_list[:amount_of_errors]

    # Initialize dictionaries to count false positives and false negatives
    false_positive_dic = {error: 0 for error in error_classes_list_fp}
    false_negative_dic = {error: 0 for error in error_classes_list_fn}


    with open(OUTPUT_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)
        for i, record in enumerate(data):
            if i >= 0: # In Case issue with the API, you can set this to 0 to start from the beginning
                try:
                    # Step 1: Generate the classification prompt for each record
                    if record["label_type"] == "False Positive":
                        response = generate_classification_prompt(record, error_classes_list_fp)
                    if record["label_type"] == "False Negative":
                        response = generate_classification_prompt(record, error_classes_list_fn)
                    
                    # Check for JSON-like structure
                    match = re.search(r"\{.*?\}", response, re.DOTALL) 
                    # Step 2: Extract the JSON-like structure from the response
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

                except Exception as e:
                    print(f"Script interrupted at i = ", i)
            
            with open("error_to_errorClasses.txt", "a", encoding="utf-8") as file:
                file.write("For record: ")
                file.write(str(record))
                file.write("\n GPT ")
                file.write(response)
                file.write("\n")

        with open("errors_dicts.txt", "w", encoding="utf-8") as file:
            file.write("False Positives: " + str(false_positive_dic) + "\n")
            file.write("False Negatives: " + str(false_negative_dic) + "\n")
