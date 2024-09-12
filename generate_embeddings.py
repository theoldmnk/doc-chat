import json
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load the scraped content
with open("scraped_content.json", "r") as f:
    scraped_data = json.load(f)

def generate_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

# Process each item in the scraped data
embedded_data = []
for item in scraped_data:
    url = item["url"]
    content = item["content"]
    
    # Generate embedding for the content
    embedding = generate_embedding(content)
    
    # Add the embedding to the item
    embedded_item = {
        "url": url,
        "content": content,
        "embedding": embedding
    }
    embedded_data.append(embedded_item)

# Save the embedded data to a new JSON file
with open("embedded_data.json", "w") as f:
    json.dump(embedded_data, f)

print(f"Embeddings generated for {len(embedded_data)} items and saved to embedded_data.json")