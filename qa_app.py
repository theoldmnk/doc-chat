import streamlit as st
from openai import OpenAI
import json
import numpy as np

# Set page config at the very beginning
st.set_page_config(page_title="Facets Documentation Q&A", page_icon="🤖", layout="wide")

# Initialize OpenAI client using st.secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Load the embedded data
@st.cache_data
def load_embedded_data():
    with open("embedded_data.json", "r") as f:
        return json.load(f)

embedded_data = load_embedded_data()

# Function to generate embedding for a query
def generate_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

# Function to find most similar content
def find_most_similar(query_embedding, top_k=3):
    similarities = []
    for item in embedded_data:
        similarity = np.dot(query_embedding, item['embedding'])
        similarities.append((similarity, item['content'], item['url']))
    return sorted(similarities, reverse=True)[:top_k]

# Function to generate answer
def generate_answer(query, context):
    prompt = f"""Answer the question based on the context provided. If the answer cannot be found in the context, say "I don't have enough information to answer that."

Context: {context}

Question: {query}

Answer:"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the given context."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# Streamlit UI
st.title("Facets Documentation Q&A")

query = st.text_input("Ask a question about Facets:")

if query:
    with st.spinner("Searching for relevant information..."):
        query_embedding = generate_embedding(query)
        most_similar = find_most_similar(query_embedding)
        
        context = "\n\n".join([item[1] for item in most_similar])
        
    with st.spinner("Generating answer..."):
        answer = generate_answer(query, context)
        
    st.write("Answer:", answer)
    
    st.subheader("Sources:")
    for _, content, url in most_similar:
        st.write(f"- [{url}]({url})")

    if st.checkbox("Show used context"):
        st.text_area("Context used for answering:", context, height=300)
