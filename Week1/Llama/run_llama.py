import openai
from openai import OpenAI

# Use Ollama's OpenAI-compatible endpoint
client = OpenAI(
    base_url = "http://localhost:11434/v1",
    api_key = "ollama",  # dummy
)

# Ask a question to your local model
response = client.chat.completions.create(
    model="llama2",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The LA Dodgers won in 2020."},
        {"role": "user", "content": "Where was it played?"}
    ]
)

# Print the model's answer
print("LLM Response:\n")
print(response.choices[0].message.content)

