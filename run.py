import openai
from tavily import TavilyClient
import json

class WebRAGEntityMatcher:
    def __init__(self, openai_api_key, travily_api_key):
        self.openai_api_key = openai_api_key
        self.travily_api_key = travily_api_key
    
    def llm_entity_match(self, entity_1, entity_2):
        openai.api_key = self.openai_api_key
        prompt = f"Do these two entities refer to the same real-world object? Entity 1: {entity_1}, Entity 2: {entity_2}. Respond with 'Yes' or 'No' only."
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return 1 if response["choices"][0]["message"]["content"].strip() == "Yes" else 0
        #return response["choices"][0]["message"]["content"].strip()
    

    def search_with_travily(self, entity):
        tavily_client = TavilyClient(api_key=self.travily_api_key)
        response = tavily_client.search(entity)
        print("Travily",response)
        return response

    def enhanced_entity_match(self, entity_1, entity_2):
        """WebRAG-enhanced entity matching."""
        info_1 = self.search_with_travily(entity_1)
        info_2 = self.search_with_travily(entity_2)
        
        context_1 = info_1["results"][0]["snippet"] if info_1 else "No additional info."
        context_2 = info_2["results"][0]["snippet"] if info_2 else "No additional info."
        
        print(context_1, context_2)
        response = self.llm_entity_match(info_1, info_2)

# Define a function to process each record
def process_record(record):
    """Extracts fields, processes them, and returns modified left and right texts."""

    # Extract 'left' values (excluding ID)
    left_parts = [
        str(record.get("brand_left", "")),
        str(record.get("title_left", "")),
        str(record.get("description_left", "")),
        str(record.get("price_left", "")),
        str(record.get("priceCurrency_left", ""))
    ]
    left_text = " ".join(left_parts)

    # Extract 'right' values (excluding ID)
    right_parts = [
        str(record.get("brand_right", "")),
        str(record.get("title_right", "")),
        str(record.get("description_right", "")),
        str(record.get("price_right", "")),
        str(record.get("priceCurrency_right", ""))
    ]
    right_text = " ".join(right_parts)

    label = record.get("label", "")

    # Example of further processing (modify as required)
    left_text = left_text.replace("/", " ")  # Replace slashes with spaces
    right_text = right_text.replace("/", " ")  # Replace slashes with spaces

    return left_text, right_text, label

# Example usage
if __name__ == "__main__":
    API_key = "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"
    TRAVILY_key = "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"
    file_path = "80pair\wdcproducts80cc20rnd100un_gs.json"

    matcher = WebRAGEntityMatcher(openai_api_key=API_key, travily_api_key=TRAVILY_key)    
    #TODO add a way to save the results
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)  # Load each JSON object separately
            entity_1, entity_2, label = process_record(record)  # Process data

            # Zero-shot entity matching
            baseline_result = matcher.llm_entity_match(entity_1, entity_2)
            print(f"Baseline Result: {baseline_result}", "Actual Label: ", label)
            
            
            # WebRAG-enhanced entity matching
            #enhanced_result = matcher.enhanced_entity_match(entity_1, entity_2)
            #print(f"WebRAG-Enhanced Result: {enhanced_result}")
