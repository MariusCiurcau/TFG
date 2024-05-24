from openai import OpenAI


def generate_explanations_gpt(original_explanation, versions):
    with open("../key.txt", "r") as f:
        API_KEY = f.read()
    client = OpenAI(api_key=API_KEY)
    explanations = {}
    for version in versions:
        prompt = f"Given the following medical explanation: <<{original_explanation}>>, generate a text of EQUAL or SHORTER length that is suitable for a person with {version} knowledge."
        completion = client.chat.completions.create(
            model="gpt-4-turbo-preview",  # GPT-4-0125-preview engine
            messages=[{"role": "user", "content": prompt}]
        )
        generated_text = completion.choices[0].message.content
        explanations[version] = generated_text
        print(f"Generated text with {version} expertise level: {generated_text}")
    return explanations
