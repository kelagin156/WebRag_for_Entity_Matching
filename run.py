import openai
from tavily import TavilyClient
import json
from sklearn.metrics import f1_score
import pandas as pd
from datetime import datetime


class WebRAGEntityMatcher:
    def __init__(self, openai_api_key, travily_api_key):
        self.openai_api_key = openai_api_key
        self.travily_api_key = travily_api_key
    
    def llm_entity_match(self, entity_1, entity_2):
        openai.api_key = self.openai_api_key
        prompt = f"Do these two entities refer to the same real-world object? Entity 1: [{entity_1}], Entity 2: [{entity_2}]. Respond only with 'Yes' or 'No' and the percentage of how certain you are about your answer. No more information"
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        with open("gpt.txt", "a", encoding="utf-8") as file:
            file.write(str(prompt) + "\n" + str(response["choices"][0]["message"]["content"].strip()) + "\n\n")
        return response["choices"][0]["message"]["content"].strip(), 

    def search_with_travily(self, title):
        tavily_client = TavilyClient(api_key=self.travily_api_key)
        prompt = f"Give me more information on this product: {title}"
        response = tavily_client.search(prompt) # due to travilies max of 400 tokens
        with open("travily.txt", "a", encoding="utf-8") as file:
            file.write(str(prompt) + "\n" + str(response) + "\n\n")
        return response

    def enhanced_entity_match(self, entity_1, entity_2, title1, title2):
        """WebRAG-enhanced entity matching."""
        info_1 = self.search_with_travily(title1)
        info_2 = self.search_with_travily(title2)
        
        context_1 = info_1["results"][0]["content"] if info_1 else "No additional info."
        context_1 = context_1 + " " + info_1["results"][0]["url"] if info_1 else ""
        context_2 = info_2["results"][0]["content"] if info_2 else "No additional info."
        context_2 = context_2 + " " + info_2["results"][0]["url"] if info_2 else ""
        
        webrag_entity_1 = entity_1 + "Additional info: " + context_1
        webrag_entity_2 = entity_2 + "Additional info: " + context_2

        return self.llm_entity_match(webrag_entity_1, webrag_entity_2)

# Define a function to process each record
def process_record(record):
    """Extracts fields, processes them, and returns modified left and right texts."""

    # Extract 'left' values (excluding ID)
    left_parts = [
        "Brand: "+ str(record.get("brand_left")) if record.get("brand_left") is not None else "" ,
        "Title: "+str(record.get("title_left")) if record.get("title_left") is not None else "" ,
        "Price: "+str(record.get("price_left")) if record.get("price_left") is not None else "",
        str(record.get("priceCurrency_left")) if record.get("priceCurrency_left") is not None else "",
        "Description: " + str(record.get("description_left")) if record.get("description_left") is not None else ""
    ]
    left_text = " ".join(filter(None, left_parts))  # filter out empty strings

    # Extract 'right' values (excluding ID)
    right_parts = [
        "Brand: "+ str(record.get("brand_right")) if record.get("brand_right") is not None else "",
        "Title: "+str(record.get("title_right")) if record.get("title_right") is not None else "",
        "Price: "+str(record.get("price_right")) if record.get("price_right") is not None else "",
        str(record.get("priceCurrency_right")) if record.get("priceCurrency_right") is not None else "",
        "Description: " +str(record.get("description_right")) if record.get("description_right") is not None else "" 
    ]
    right_text = " ".join(filter(None, right_parts))  # filter out empty strings

    label = record.get("label", "")

    # Example of further processing (modify as required)
    left_text = left_text.replace("/", " ")  # Replace slashes with spaces
    right_text = right_text.replace("/", " ")  # Replace slashes with spaces

    left_title = str(record.get("title_left")).replace("/", " ")
    right_title = str(record.get("title_right")).replace("/", " ")

    return left_text, right_text, label, left_title, right_title

# Example usage
if __name__ == "__main__":
    API_key = "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"
    TRAVILY_key = "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"
    file_path = "final400datasets.json"
    matcher = WebRAGEntityMatcher(openai_api_key=API_key, travily_api_key=TRAVILY_key)    

    rows = []

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
        try:
            for i, record in enumerate(data):
                if i >= 0: # in case the code fails the number can be adjusted to the last i
                    entity_1, entity_2, label, title1, title2 = process_record(record)  # Process data

                    # Zero-shot entity matching
                    baseline_result = matcher.llm_entity_match(entity_1, entity_2)
                    chatGPT40_mini_baseline_y_pred = int("Yes" in str(baseline_result))
                    print(f"Index: {i}   Baseline Result: {baseline_result, chatGPT40_mini_baseline_y_pred}", "Actual Label: ", label)
                    
                    
                    # WebRAG-enhanced entity matching
                    enhanced_result = matcher.enhanced_entity_match(entity_1, entity_2, title1, title2)
                    webRag_y_pred = int("Yes" in str(enhanced_result))
                    print(f"Index: {i}   WebRAG-Enhanced Result: {enhanced_result, webRag_y_pred}", "Actual Label: ", label)
                    
                    rows.append({
                                "Entity1": str(entity_1),
                                "Entity2": str(entity_2),
                                "GPT_only_response": str(baseline_result),
                                "GPT_Travily_response": str(enhanced_result),
                                "y_true": label,
                                "ChatGPT40-mini_baseline_y_pred": chatGPT40_mini_baseline_y_pred,
                                "WebRag_y_pred": webRag_y_pred,
                            })
        except Exception as e:
            print(f"\n❗ Script interrupted due to error:\n{e}\n")

        finally:
            # Save whatever has been collected so far
            df = pd.DataFrame(rows)
            df.to_csv("entity_matching_results_400.csv", index=False, encoding="utf-8",  mode="a", header=not pd.io.common.file_exists("entity_matching_results_400.csv"))
            print(f"\n✅ Saved {len(df)} results to entity_matching_results_400.csv (partial or full)")
            print(f"the las item was {i}")
            
            df_all = pd.read_csv("entity_matching_results_400.csv")
            baseline_f1 = f1_score(df_all["y_true"], df_all["ChatGPT40-mini_baseline_y_pred"])
            webrag_f1 = f1_score(df_all["y_true"], df_all["WebRag_y_pred"])

            print("Baseline F1 score: ", baseline_f1)
            print("WebRag F1 Score: ", webrag_f1)

            df_f1 = pd.DataFrame([{"Date-Time": datetime.now(), "Baseline_F1": baseline_f1, "WebRag_F1": webrag_f1}])
            df_f1.to_csv("f1_results.csv", index=False, encoding="utf-8", mode="a", header=not pd.io.common.file_exists("f1_results.csv"))