from openai import OpenAI
import openai

# Set your OpenAI API key
API_KEY = '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'

def generate_explanations_gpt(original_explanation, versions):
    client = OpenAI(api_key=API_KEY)
    explanations = {}
    for version in versions:
        prompt = f"[INST]Given the following medical explanation: <<{original_explanation}>>, generate a text of EQUAL or SHORTER length that is fit for a person with {version} knowledge. Please respect this restriction.[/INST]"
        completion = client.chat.completions.create(
            model="gpt-4-turbo-preview",  # GPT-4-0125-preview engine
            messages=[{"role": "user", "content": prompt}]
        )
        generated_text = completion.choices[0].message.content
        explanations[version] = generated_text
        print(f"Generated text with {version} expertise level: {generated_text}")
    return explanations

# Example usage
#original_text = 'A subcapital femoral neck fracture is a specific type of fracture that occurs in the femur bone, specifically in the region of the femoral neck, just below the femoral head. This fracture occurs within the hip joint and can be serious due to the disruption of blood supply to the femoral head, which can lead to complications such as avascular necrosis. Subcapital femoral neck fractures are common in older individuals, especially those with osteoporosis or decreased bone density, which makes bones more prone to fracturing with minimal trauma. These fractures typically occur after a fall or direct impact to the hip.'
#expertise_levels = ['basic', 'intermediate', 'advanced']

#generated_explanations = generate_explanations(original_text, expertise_levels)
