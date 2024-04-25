import requests

# Define the API endpoint for text generation
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
API_TOKEN = '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'
headers = {"Authorization": f"Bearer {API_TOKEN}"}

MAX_LENGTH = 1000
def generate_explanations_mistral(original_explanation, versions):
    explanations = {}
    for version in versions:
        prompt = f"<s>[INST]Given the following medical explanation: <<{original_explanation}>>, generate a text of EQUAL or SHORTER length that is suitable for a person with {version} knowledge.[/INST]"
        payload = {
            "inputs": prompt,
            "options": {
                "max_new_tokens": MAX_LENGTH,
                "wait_for_model": True
            }
        }

        response = requests.post(API_URL, headers=headers, json=payload)
        generated_text = response.json()[0]['generated_text']
        generated_text = generated_text[len(prompt):].strip()
        explanations[version] = generated_text
        #print(f"Generated text with {version} expertise level: {generated_text}")
    return explanations