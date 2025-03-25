import openai
import json
import pandas as pd
import time
from openai.error import APIError, Timeout, RateLimitError, ServiceUnavailableError
from sklearn.metrics import f1_score

# Due to the travily and chat gpt needing 

def process_record(record):
    """Extracts fields, processes them, and returns modified left and right texts."""

    # Extract 'left' values (excluding ID)
    left_parts = [
        str(record.get("brand_left")) if record.get("brand_left") is not None else "",
        str(record.get("title_left")) if record.get("title_left") is not None else "",
        str(record.get("description_left")) if record.get("description_left") is not None else "",
        str(record.get("price_left")) if record.get("price_left") is not None else "",
        str(record.get("priceCurrency_left")) if record.get("priceCurrency_left") is not None else ""
    ]
    left_text = " ".join(filter(None, left_parts))  # filter out empty strings

    # Extract 'right' values (excluding ID)
    right_parts = [
        str(record.get("brand_right")) if record.get("brand_right") is not None else "",
        str(record.get("title_right")) if record.get("title_right") is not None else "",
        str(record.get("description_right")) if record.get("description_right") is not None else "",
        str(record.get("price_right")) if record.get("price_right") is not None else "",
        str(record.get("priceCurrency_right")) if record.get("priceCurrency_right") is not None else ""
    ]
    right_text = " ".join(filter(None, right_parts))  # filter out empty strings


    label = record.get("label", "")

    # Example of further processing (modify as required)
    left_text = left_text.replace("/", " ")  # Replace slashes with spaces
    right_text = right_text.replace("/", " ")  # Replace slashes with spaces

    return left_text, right_text, label

def llm_entity_match(openai_api_key, entity_1, entity_2):
    openai.api_key = openai_api_key
    prompt = f"Do these two entities refer to the same real-world object? Entity 1: {entity_1}, Entity 2: {entity_2}. Respond with only 'Yes' or 'No' and give me the percentage of how certain you are about your response only."
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"].strip()

def safe_llm_entity_match(api_key, entity_1, entity_2, retries=3, delay=5):
    for attempt in range(1, retries + 1):
        try:
            return llm_entity_match(api_key, entity_1, entity_2)
        except (APIError, Timeout, RateLimitError, ServiceUnavailableError) as e:
            print(f"[Attempt {attempt}] OpenAI API error: {e}")
            if attempt < retries:
                time.sleep(delay)
            else:
                print("❌ All retries failed. Exiting.")
                raise e  # or handle how you'd like
def test_llm_what_the_llm_knows():
    API_key = "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"
    file_path = "80pair\wdcproducts80cc20rnd100un_gs.json"
    
    rows = []
    non_match = 0
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            for i,line in enumerate(file):
                if i >= 0: # in case the code fails the number can be adjusted to the last i
                    record = json.loads(line)  # Load each JSON object separately
                    entity_1, entity_2, label = process_record(record)  # Process data
                    
                    prompt = f"Do these two entities refer to the same real-world object? Entity 1: {entity_1}, Entity 2: {entity_2}. Respond with 'Yes' or 'No' and give me the percentage of how certain you are about your response only."
                    
                    answer = llm_entity_match(openai_api_key=API_key,entity_1=entity_1, entity_2=entity_2)
                    
                    answer_int = int("Yes" in answer)
                    match = int(answer_int == label)

                    rows.append({
                        "Prompt": prompt,
                        "Answer": answer,
                        "Label": label,
                        "Match": match
                    })

                    if match == 0:
                        non_match = non_match +1

                    print("Index:", i, " Answer: ", answer, " Label:", label, "Match:", match, " Non-matches: ", non_match)
    except Exception as e:
        print(f"\n❗ Script interrupted due to error:\n{e}\n")

    finally:
        # Save whatever has been collected so far
        df = pd.DataFrame(rows)
        df.to_csv("dataset_selection_results.csv", index=False, encoding="utf-8")
        print(f"\n✅ Saved {len(df)} results to dataset_selection_results.csv (partial or full)")
        print(f"the las item was {i}")

def select_dataset():
    # Read the CSV
    df = pd.read_csv("dataset_selection_results.csv")
    # Select 100 rows with Match == 0
    match_0_indices = df[df["Match"] == 0].head(100).index.tolist()
    # Select 100 rows with Match == 1
    match_1_indices = df[df["Match"] == 1].head(100).index.tolist()
    # Combine selected indices
    selected_indices = set(match_0_indices + match_1_indices)
    print(selected_indices)

    input_json_file = "80pair\wdcproducts80cc20rnd100un_gs.json"
    output_json_file = "final200datasets.json"

    # Collect matching JSON objects
    selected_jsons = []

    # Read JSON file line by line
    with open(input_json_file, "r", encoding="utf-8") as file:
        for index, line in enumerate(file):
            if index in selected_indices:  # Check if index is in our selection
                json_obj = json.loads(line)  # Parse JSON
                selected_jsons.append(json_obj)

    # Write the selected JSON objects to a new file
    with open(output_json_file, "w", encoding="utf-8") as outfile:
        json.dump(selected_jsons, outfile, indent=4)  # Pretty-print JSON


if __name__ == "__main__":
    #test_llm_what_the_llm_knows()
    #select_dataset()
    df = pd.read_csv("dataset_selection_results.csv")
    df["Answer_binary"] = df["Answer"].apply(lambda x: 1 if "Yes" in str(x) else 0)
    f1 = f1_score(df["Label"], df["Answer_binary"])
    print(f1)

    count_non_matcn = (df["Match"] == 0).sum()
    print(count_non_matcn)
        