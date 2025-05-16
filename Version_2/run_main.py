import openai
from tavily import TavilyClient
import json
from sklearn.metrics import f1_score
import pandas as pd
from datetime import datetime
import tiktoken

def count_tokens(text, model="gpt-4o"):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


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

        content = response["choices"][0]["message"]["content"].strip()
        usage = response.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        with open("gpt.txt", "a", encoding="utf-8") as file:
            file.write(str(prompt) + "\n" + str(content) + "\n\n")

        return content, response, prompt, input_tokens, output_tokens
    
    def llm_entity_match_webrag(self, entity_1, entity_2, additionalInfo):
        openai.api_key = self.openai_api_key
        prompt = f"Do these two entities refer to the same real-world object? Entity 1: [{entity_1}], Entity 2: [{entity_2}]. Here is some Additional Information: [{additionalInfo}]. Respond only with 'Yes' or 'No' and the percentage of how certain you are about your answer. No more information"
        
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        content = response["choices"][0]["message"]["content"].strip()
        usage = response.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        with open("gpt.txt", "a", encoding="utf-8") as file:
            file.write(str(prompt) + "\n" + str(content) + "\n\n")

        return content, response, prompt, input_tokens, output_tokens

    def search_with_travily(self, title, mex_results):
        tavily_client = TavilyClient(api_key=self.travily_api_key)

        # Input: your query string
        query_input_tokens = count_tokens(title)

        response = tavily_client.search(query=title, max_results=mex_results)

        # Output: their returned content (flattened)
        context_text = " ".join(
            f"{res.get('title', '')} {res.get('content', '')} {res.get('url', '')}".strip()
            for res in response.get("results", [])
        )
        response_output_tokens = count_tokens(context_text)

        with open("travily.txt", "a", encoding="utf-8") as file:
                file.write(str(title) + "\n" + str(response) +"\n\n")

        return {
            "results": response.get("results", []),
            "input_tokens": query_input_tokens,
            "output_tokens": response_output_tokens
        }


    def webRag_entity_match(self, entity_1, entity_2, title1, title2):
        webrag_results = {}

        # Only one Travily query per side
        info_1_full = self.search_with_travily(title1, 5)
        info_2_full = self.search_with_travily(title2, 5)

        full_results_1 = info_1_full["results"]
        full_results_2 = info_2_full["results"]

        for n in [1, 3, 5]:
            results_1_n = full_results_1[:n]
            results_2_n = full_results_2[:n]

            def build_context(results):
                return " ".join(
                    f"{res.get('title', '')} {res.get('content', '')} {res.get('url', '')}".strip()
                    for res in results
                )

            context_1 = build_context(results_1_n)
            context_2 = build_context(results_2_n)

            additionalInfo = context_1 + " " + context_2

            content, response, prompt, input_tokens, output_tokens = self.llm_entity_match_webrag(entity_1, entity_2, additionalInfo)

            # Simulate token usage for sliced context
            travily_input_tokens = count_tokens(title1) + count_tokens(title2)
            travily_output_tokens = count_tokens(context_1) + count_tokens(context_2)

            webrag_results[n] = {
                "response": content,
                "prompt": prompt,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "prediction": int("yes" in content.lower()),
                "travily_input_tokens": travily_input_tokens,
                "travily_output_tokens": travily_output_tokens
            }

            with open("travily.txt", "a", encoding="utf-8") as file:
                file.write(f"n={n}\n{prompt}\n{content}\n\n")

        return webrag_results



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
                    entity_1, entity_2, label, title1, title2 = process_record(record)

                    baseline_result, base_response, base_prompt, base_in_tokens, base_out_tokens = matcher.llm_entity_match(entity_1, entity_2)
                    baseline_pred = int("Yes" in baseline_result)
                    print(f"[{i}] Baseline:", baseline_result, "| Label:", label)

                    webrag_results = matcher.webRag_entity_match(entity_1, entity_2, title1, title2)
                    print(f"[{i}] WebRag1:", webrag_results[1]["prediction"], "| Label:", label)
                    print(f"[{i}] WebRag3:", webrag_results[3]["prediction"], "| Label:", label)
                    print(f"[{i}] WebRag5:", webrag_results[5]["prediction"], "| Label:", label)

                    rows.append({
                        "Entity1": entity_1,
                        "Entity2": entity_2,
                        "y_true": label,
                        "ChatGPT40-mini_baseline_y_pred": baseline_pred,

                        "WebRag_y_pred_n1": webrag_results[1]["prediction"],
                        "WebRag_y_pred_n3": webrag_results[3]["prediction"],
                        "WebRag_y_pred_n5": webrag_results[5]["prediction"],

                        "Baseline_Prompt": base_prompt,
                        "Baseline_Response": baseline_result,
                        "Baseline_Input_Tokens": base_in_tokens,
                        "Baseline_Output_Tokens": base_out_tokens,

                        "WebRAG_1_Prompt": webrag_results[1]["prompt"],
                        "WebRAG_1_Response": webrag_results[1]["response"],
                        "WebRAG_1_GPT_Input_Tokens": webrag_results[1]["input_tokens"],
                        "WebRAG_1_GPT_Output_Tokens": webrag_results[1]["output_tokens"],
                        "Travily_Input_Tokens_n1": webrag_results[1]["travily_input_tokens"],
                        "Travily_Output_Tokens_n1": webrag_results[1]["travily_output_tokens"],

                        "WebRAG_3_Prompt": webrag_results[3]["prompt"],
                        "WebRAG_3_Response": webrag_results[3]["response"],
                        "WebRAG_3_GPT_Input_Tokens": webrag_results[3]["input_tokens"],
                        "WebRAG_3_GPT_Output_Tokens": webrag_results[3]["output_tokens"],
                        "Travily_Input_Tokens_n3": webrag_results[3]["travily_input_tokens"],
                        "Travily_Output_Tokens_n3": webrag_results[3]["travily_output_tokens"],

                        "WebRAG_5_Prompt": webrag_results[5]["prompt"],
                        "WebRAG_5_Response": webrag_results[5]["response"],
                        "WebRAG_5_GPT_Input_Tokens": webrag_results[5]["input_tokens"],
                        "WebRAG_5_GPT_Output_Tokens": webrag_results[5]["output_tokens"],
                        "Travily_Input_Tokens_n5": webrag_results[5]["travily_input_tokens"],
                        "Travily_Output_Tokens_n5": webrag_results[5]["travily_output_tokens"],
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
            webrag_f1_n1 = f1_score(df_all["y_true"], df_all["WebRag_y_pred_n1"])
            webrag_f1_n3 = f1_score(df_all["y_true"], df_all["WebRag_y_pred_n3"])
            webrag_f1_n5 = f1_score(df_all["y_true"], df_all["WebRag_y_pred_n5"])

            print("Baseline F1:", baseline_f1)
            print("WebRAG F1 (n=1):", webrag_f1_n1)
            print("WebRAG F1 (n=3):", webrag_f1_n3)
            print("WebRAG F1 (n=5):", webrag_f1_n5)
