import streamlit as st
import requests
from bs4 import BeautifulSoup
import json
from urllib.parse import urlparse, urljoin
import logging
import io
import re
from newspaper import Article, Config
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a string IO object to capture log messages
log_stream = io.StringIO()
handler = logging.StreamHandler(log_stream)
logger.addHandler(handler)

# Add this new function to scrape content from URLs
def scrape_content(urls, max_urls=10):
    logger.info(f"Starting to scrape content from {len(urls)} URLs (max {max_urls})")
    content = []
    
    # Configure newspaper
    config = Config()
    config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    config.request_timeout = 10
    
    for i, url in enumerate(urls[:max_urls], 1):
        logger.info(f"Scraping URL {i}/{max_urls}: {url}")
        try:
            article = Article(url, config=config)
            article.download()
            article.parse()
            
            content.append({
                'url': url,
                'title': article.title,
                'content': article.text,
                'authors': article.authors,
                'publish_date': str(article.publish_date) if article.publish_date else None,
                'top_image': article.top_image,
                'keywords': article.keywords if article.keywords else []
            })
            
            logger.info(f"Successfully scraped content from {url} ({len(article.text)} characters)")
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
    
    logger.info(f"Finished scraping content from {len(content)} URLs")
    return content

def extract_sitemap(url):
    if not url.startswith('http'):
        url = 'https://' + url

    logger.info(f"Processing URL: {url}")
    
    # List of common sitemap locations
    sitemap_locations = [
        '/sitemap.xml',
        '/sitemap_index.xml',
        '/sitemap/',
        '/sitemap.php',
        '/sitemap.txt',
    ]

    for location in sitemap_locations:
        sitemap_url = urljoin(url, location)
        logger.info(f"Trying sitemap at: {sitemap_url}")
        try:
            response = requests.get(sitemap_url, timeout=10)
            response.raise_for_status()
            
            if 'xml' in response.headers.get('Content-Type', '').lower():
                soup = BeautifulSoup(response.content, 'xml')
                urls = [loc.text for loc in soup.find_all('loc')]
                if urls:
                    logger.info(f"Found sitemap at {sitemap_url}")
                    return urls
        except requests.RequestException as e:
            logger.warning(f"Failed to fetch {sitemap_url}: {str(e)}")

    # If no sitemap found, fallback to scraping the homepage
    logger.info("No sitemap found. Falling back to scraping the homepage.")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        urls = [urljoin(url, a['href']) for a in soup.find_all('a', href=True)]
        return urls
    except requests.RequestException as e:
        logger.error(f"Error fetching homepage: {str(e)}")
        st.error(f"Error fetching homepage: {str(e)}")
        return []

def extract_subdomains(urls):
    logger.info("Extracting subdomains from URLs")
    subdomains = set()
    for url in urls:
        parsed_url = urlparse(url)
        subdomain = parsed_url.netloc
        subdomains.add(subdomain)
    logger.info(f"Extracted {len(subdomains)} unique subdomains")
    return list(subdomains)

# Add this near the top of your script, after the imports
if 'extracted_urls' not in st.session_state:
    st.session_state.extracted_urls = None

st.set_page_config(page_title="Sitemap Extractor and Scraper", page_icon="🌐", layout="wide")

st.markdown("""
    ## Sitemap Extractor and Content Scraper
    1. Enter the domain name (e.g., readme.facets.cloud) or full URL of the website
    2. The app will attempt to find the sitemap or extract links from the homepage
    3. Click 'Extract Sitemap' to begin the process
    4. Optionally, click 'Scrape Content' to extract content from the found URLs
""")

url = st.text_input("Enter the domain or website URL:", help="Enter the domain name or full URL")

if st.button("Extract Sitemap"):
    if url:
        with st.spinner("Extracting sitemap..."):
            progress_bar = st.progress(0)
            urls = extract_sitemap(url)
            progress_bar.progress(50)
            
        if urls:
            st.session_state.extracted_urls = urls  # Store the extracted URLs in session state
            st.success(f"Extracted {len(urls)} URLs from the sitemap.")
            
            with st.spinner("Extracting subdomains..."):
                subdomains = extract_subdomains(urls)
                progress_bar.progress(75)
            
            data = {
                "urls": urls,
                "subdomains": subdomains
            }
            
            st.download_button(
                label="Download Extracted Data",
                data=json.dumps(data, indent=2),
                file_name="sitemap_data.json",
                mime="application/json"
            )
            
            st.subheader("Extracted Subdomains:")
            for subdomain in subdomains:
                st.write(subdomain)
            
            progress_bar.progress(100)
            logger.info("Extraction process completed successfully")
    else:
        st.warning("Please enter a URL.")
        logger.warning("Extraction attempted without a URL")

# Move the "Scrape Content" button outside the "Extract Sitemap" button condition
if st.session_state.extracted_urls:
    max_urls = st.number_input("Max URLs to scrape", min_value=1, max_value=100, value=10)
    if st.button("Scrape Content"):
        with st.spinner("Scraping content..."):
            scraped_content = scrape_content(st.session_state.extracted_urls, max_urls)
        
        st.success(f"Scraped content from {len(scraped_content)} URLs.")
        
        # Display a sample of scraped content
        if scraped_content:
            st.subheader("Sample Scraped Content:")
            for i, item in enumerate(scraped_content[:3]):
                st.write(f"URL: {item['url']}")
                st.write(f"Title: {item['title']}")
                st.text_area("Content Preview:", item['content'][:1000] + "...", height=100, key=f"preview_{i}")
                st.write(f"Authors: {', '.join(item['authors'])}")
                st.write(f"Publish Date: {item['publish_date']}")
                st.write(f"Top Image: {item['top_image']}")
                st.write(f"Keywords: {', '.join(item['keywords'])}")
                st.write("---")
        
        # Provide download button for scraped content
        st.download_button(
            label="Download Scraped Content (JSON)",
            data=json.dumps(scraped_content, indent=2),
            file_name="scraped_content.json",
            mime="application/json"
        )

        # Add information about the JSON format
        st.info("""
        The downloaded JSON file contains an array of objects with the following fields:
        - url: The URL of the scraped page
        - title: The title of the article
        - content: The main text content of the article
        - authors: List of authors (if available)
        - publish_date: The publication date (if available)
        - top_image: URL of the top image in the article (if available)
        - keywords: List of keywords extracted from the article (if available)
        
        This format is suitable for further processing, analysis, or feeding into LLMs.
        """)

st.sidebar.info(
    "Note: The extracted data is provided in its raw form. "
    "Depending on your specific needs, you may want to implement "
    "additional cleaning steps, such as removing duplicates, "
    "filtering out unwanted URLs, or normalizing the format."
)

# Display logging information
if st.checkbox("Show detailed logs"):
    st.text_area("Logs", log_stream.getvalue())

if st.checkbox("Show raw response"):
    if 'response' in locals():
        st.text_area("Raw Response", response.text, height=300)
    else:
        st.info("No response available. Extract the sitemap first.")

def print_full_html(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        logger.info(f"Full HTML for {url}:")
        logger.info(soup.prettify())
    except requests.RequestException as e:
        logger.error(f"Error fetching {url}: {str(e)}")

# Add a section to print full HTML of a specific URL
st.subheader("Debug: Print Full HTML")
debug_url = st.text_input("Enter URL to print full HTML:")
if st.button("Print Full HTML") and debug_url:
    print_full_html(debug_url)
    st.success(f"Full HTML for {debug_url} has been printed to the logs.")

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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