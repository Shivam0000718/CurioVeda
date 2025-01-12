import streamlit as st
import requests
from requests.exceptions import RequestException
import logging
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from sklearn.preprocessing import normalize
import numpy as np
from langchain.docstore.document import Document
from dotenv import load_dotenv
import os
from boilerpy3 import extractors
import pytesseract
from PIL import Image
from io import BytesIO
import re
import time
import plotly.express as px
import pandas as pd
import base64


load_dotenv()

# Load API keys
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


# Initialize session state for URLs and scraped data and Chat History
if "urls" not in st.session_state:
    st.session_state.urls = []

if "scraped_data" not in st.session_state:
    st.session_state.scraped_data = {}

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "visualization_data" not in st.session_state:
    st.session_state.visualization_data = {"ready_for_visualization": False, "data": None}

if "greeting_shown" not in st.session_state:
    st.session_state.greeting_shown = False

# Greet the user if opening the app for the first time
if not st.session_state.greeting_shown:
    st.session_state.chat_history.insert(0, {
        "question": "System Greeting",
        "answer": "Hello! I'm CurioVeda, your AI assistant. How can I assist you today?"
    })
    st.session_state.greeting_shown = True


# Function to add URL
def add_url(url):
    if url and url not in st.session_state.urls:
        st.session_state.urls.append(url)
    elif not url:
        st.warning("URL cannot be empty.")
    elif url in st.session_state.urls:
        st.warning("URL already added.")

# Function to delete URL
def delete_url(idx):
    url_to_delete = st.session_state.urls[idx]
    st.session_state.urls.pop(idx)
    # Also remove scraped data associated with this URL
    if url_to_delete in st.session_state.scraped_data:
        del st.session_state.scraped_data[url_to_delete]

# Set up logging
logging.basicConfig(filename="app_logs.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Enhanced scrape_url with detailed error handling and logging
def scrape_url(url):
    try:
        start_time = time.time()  # Start the timer for scraping duration
        # Attempt to scrape content
        content = scrape_article(url)
        
        if not content or "error" in content:
            error_msg = f"Failed to scrape {url}. Reason: {content.get('error', 'Unknown error')}"
            logging.error(error_msg)
            return error_msg
        
        # Extract relevant content
        text_content = content['text']
        metadata = content['metadata']

        # Include metadata at the beginning of the content
        if metadata:
            metadata_content = f"Title: {metadata.get('title', 'No Title')}\n"
            metadata_content += f"Author: {metadata.get('author', 'No Author')}\n"
            metadata_content += f"Published Date: {metadata.get('date', 'No Date')}\n\n"
            text_content = metadata_content + text_content

        # Optionally, combine text content with image text or inline links if needed
        if content['image_texts']:
            text_content += "\n\n" + "\n".join(content['image_texts'])
        if content['inline_links']:
            text_content += "\n\nInline Links:\n" + "\n".join([f"{link['text']} ({link['url']})" for link in content['inline_links']])
        
        # Add headings and subheadings to enhance content structure
        text_content = enhance_with_headings(content['headings'], text_content)
        # Process the content
        text_content = text_content.strip()

        end_time = time.time()  # End the timer
        elapsed_time = round(end_time - start_time, 2)
        st.info(f"Scraping completed for {url} in {elapsed_time} seconds.")  # Display the timing

        if not text_content:
            logging.warning(f"No meaningful content found for {url}.")
            return f"No meaningful content found for {url}."
        
        return text_content
    
    except RequestException as e:
        logging.error(f"Network error while scraping {url}: {str(e)}")
        return f"Network error while scraping {url}. Please check the URL or your connection."
    except Exception as e:
        logging.error(f"Unexpected error while scraping {url}: {str(e)}")
        return f"Unexpected error while scraping {url}: {str(e)}"

# Function to scrape article with enhanced method
def scrape_article(url):
    try:
        # Step 1: Fetch the HTML content
        response = requests.get(url)
        response.raise_for_status()
        html_content = response.text

        # Step 2: Extract main content with BoilerPy3 (Removes boilerplate content)
        extractor = extractors.ArticleExtractor()
        main_text = extractor.get_content(html_content)

        # Step 3: Parse full HTML with BeautifulSoup for further extraction
        soup = BeautifulSoup(html_content, "html.parser")

        # Extract metadata (title, author, published date)
        metadata = {
            "title": soup.title.string if soup.title else "No title found",
            "meta_description": soup.find("meta", {"name": "description"})["content"]
            if soup.find("meta", {"name": "description"})
            else "No description found",
            "publication_date": soup.find("meta", {"name": "date"})["content"]
            if soup.find("meta", {"name": "date"})
            else "No publication date found",
        }

        # Step 4: Extract images and use OCR for text extraction
        image_texts = []
        for img_tag in soup.find_all("img"):
            img_url = img_tag.get("src")
            if img_url:
                try:
                    # Fetch and process the image
                    img_response = requests.get(img_url)
                    img = Image.open(BytesIO(img_response.content))
                    text = pytesseract.image_to_string(img, lang="eng").strip()
                    if text:
                        image_texts.append(text)
                except Exception:
                    continue

        # Step 5: Handle inline links, footnotes, and references
        inline_links = []
        for link in soup.find_all("a", href=True):
            href = link["href"]
            text = link.get_text(strip=True)
            if href and text:
                inline_links.append({"text": text, "url": href})

        # Step 6: Extract headings to maintain document structure
        headings = extract_headings(soup)

        # Step 7: Clean and preprocess the extracted text
        # Remove excessive whitespace, special characters, and ads-related sections
        cleaned_text = re.sub(r"\s+", " ", main_text)  # Normalize whitespace
        cleaned_text = re.sub(r"Advertisement|Sponsored", "", cleaned_text, flags=re.IGNORECASE)

        # Return all content in a structured format
        return {
            "text": cleaned_text,
            "image_texts": image_texts,
            "inline_links": inline_links,
            "metadata": metadata,
            "headings": headings,
        }

    except Exception as e:
        return {"error": str(e)}

# Helper function to extract headings
def extract_headings(soup):
    headings = []
    for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        headings.append((tag.name, tag.get_text(strip=True)))
    return headings

# Function to add hierarchical structure (headings) to content
def enhance_with_headings(headings, text_content):
    # Prepend the headings to the content for better context clarity
    for level, heading in headings:
        text_content = f"Heading ({level}): {heading}\n" + text_content
    return text_content

# <------------------------------------------------------------------>

# Function to detect tabular data in the chatbot response
def detect_table_in_response(answer):
    """
    Detects tables in free-form text, including approximate or inconsistent formats.
    Handles ranges, numeric/text values, and space-separated tables.
    """
    if isinstance(answer, dict) and "text" in answer:
        response_text = answer["text"]
    elif isinstance(answer, str):
        response_text = answer
    else:
        return None

    # Try to find tables with simple heuristics
    table_data = []
    lines = response_text.split("\n")
    for line in lines:
        # Match lines that look like table rows (tab or space-separated data)
        if re.match(r"^\s*\d{4}(\s+|\t)\S+", line):  # Line starts with a year (e.g., "2014")
            table_data.append(line.strip())

    if not table_data:
        return None  # No table-like data found

    # Extract headers (first line before table)
    headers = ["Year", "Wealth (USD Millions)"]
    data_rows = []

    for row in table_data:
        # Split by spaces or tabs
        parts = re.split(r"\s{2,}|\t", row.strip())  # Use 2+ spaces or tabs as separator
        if len(parts) >= 2:  # Ensure at least two columns exist
            year = parts[0]
            wealth = parts[1]

            # Normalize ranges like "1,500 - 2,000" into averages
            if "-" in wealth:
                numbers = [float(n.replace(",", "")) for n in wealth.split("-") if n.strip().replace(",", "").isdigit()]
                wealth = sum(numbers) / len(numbers) if numbers else None
            else:
                wealth = float(wealth.replace(",", "")) if wealth.replace(",", "").isdigit() else None

            data_rows.append([year, wealth])

    # Convert to DataFrame
    if data_rows:
        df = pd.DataFrame(data_rows, columns=headers)
        return df

    return None

# Function to generate Plotly graphs
def generate_graph(data, chart_type):
    if chart_type == "Bar Chart":
        return px.bar(data, x=data.columns[0], y=data.columns[1], title="Bar Chart")
    elif chart_type == "Line Chart":
        return px.line(data, x=data.columns[0], y=data.columns[1], title="Line Chart")
    elif chart_type == "Pie Chart":
        return px.pie(data, names=data.columns[0], values=data.columns[1], title="Pie Chart")
    elif chart_type == "Histogram":
        return px.histogram(data, x=data.columns[0], title="Histogram")
    return None

# <----------------------------------------------------------------->
def extract_and_preprocess(documents):
    # Combine all documents into a single text
    text = " ".join([doc.page_content for doc in documents])

    # More robust cleaning (handling special characters and more)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.lower()

    return text, documents[0].metadata  # Return text and metadata

# Semantic Chunking with Overlap
def chunk_text(text, chunk_size=500, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_text(text)

def embed_and_postprocess(chunks, embeddings_model):
    embedded_chunks = embeddings_model.embed_documents(chunks)
    embedded_chunks_np = np.array(embedded_chunks)
    normalized_embeddings = normalize(embedded_chunks_np, axis=1)  # Normalize
    return list(zip(chunks, normalized_embeddings.tolist()))

def create_vector_store():
    if st.session_state.scraped_data:
        embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # Initialize the embeddings model
        start_time = time.time()  # Start the timer
        # Convert scraped data (dict) into a list of Document objects
        documents = [
            Document(page_content=content, metadata={"url": url})
            for url, content in st.session_state.scraped_data.items()
        ]
        
        text, metadata = extract_and_preprocess(documents)
        chunks = chunk_text(text)
        text_embeddings = embed_and_postprocess(chunks, embeddings_model)  # Now includes (chunk, embedding) pairs

        # Generate embeddings and create FAISS vector store
        st.session_state.vector_store = FAISS.from_embeddings(
            text_embeddings=text_embeddings,  # Correct format
            embedding=embeddings_model
        )
        # Generate embeddings and create FAISS vector store
        # st.session_state.vector_store = FAISS.from_documents(splitted_docs, st.session_state.embeddings)
        end_time = time.time()  # End the timer
        elapsed_time = round(end_time - start_time, 2)
        st.success(f"Learning is successful in {elapsed_time} seconds!. Now you start asking anything related to provided content.")  # Display the timing
    else:
        st.warning("No scraped content available to Learn.")


def query_bot_with_context(question, context):
    """
    Handles user queries and retrieves context-aware responses from the AI model.
    Includes structured responses and gracefully handles edge cases.
    """
    # Check if vector store is initialized
    if "vector_store" not in st.session_state or not st.session_state.vector_store:
        st.error("Vector store is not initialized. Please scrape content and create the vector store first.")
        return {"text": "Vector store not initialized."}

    # Handle specific questions about the chatbot's identity
    if question.lower() in ["what is your name?", "who are you?", "what do i call you?"]:
        return {"text": "My name is CurioVeda. I'm here to assist you with your articles and queries!"}

    # Set up the retrieval chain
    retriever = st.session_state.vector_store.as_retriever()
    llm = ChatGroq(api_key=groq_api_key, model_name="Llama3-8b-8192")

    # Define a prompt for professional and structured responses
    prompt = ChatPromptTemplate.from_template(
        """
        You are an AI assistant skilled in generating structured, actionable, and professional responses. 
        Use the context and user query to generate the answer in the following format:
        1. Provide a detailed text response to address the user's query.
        2. If the response contains numerical data or lists, structure it in a table format (e.g., pandas DataFrame-like).
        3. Ensure that the numerical data is formatted and labeled clearly for visualization (e.g., column names like 'Year', 'Values').
        4. If somone ask about this type of question, like What is Your Name? then you can answer like "My name is CurioVeda. I'm here to assist you with your articles and queries!"
        **Guiding Principles**:
        - Always provide professional and well-structured text responses.
        - Include numerical data tables if relevant, formatted for easy visualization.
        - Avoid unnecessary data or verbose explanations.

        Use the context to generate the response:
        {context}

        Now, answer the user's question:
        {input}
        """
    )

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Retrieve answer
    try:
        response = retrieval_chain.invoke({'context': context, 'input': question})
    except Exception as e:
        return {"text": f"An error occurred while generating the response: {str(e)}"}

    # Handle cases where the response is empty or irrelevant
    if not response or "answer" not in response or response["answer"].strip() == "":
        return {"text": "I'm here to assist with knowledge-based queries. Could you please ask a question related to the provided information?"}

    return {"text": response.get("answer", "I'm sorry, I couldn't generate a response.")}


# Set the page configuration FIRST
st.set_page_config(page_title="CurioVeda: Interactive Article Insights", page_icon="static/assistant.png", layout="wide")

# <----------------------------------------------------------------->
# Function to get base64 of an image
def get_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
# Load custom CSS
with open("static/style.css", "r") as f:
    css = f.read()
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

logo_path = "static/logo-curioveda.png"
hero_image_path = "static/Chatbot-background.webp" # Background image for the hero section
# <----------------------------------------------------------------->


# Header Section with Login/Signup Button
logo_base64 = get_image_base64(logo_path) if os.path.exists(logo_path) else ""
st.markdown(f"""
    <div class="header">
        <div class="header-logo">
            <img src="data:image/png;base64,{logo_base64}" alt="Logo">
            <h1>CurioVeda</h1>
        </div>
        <div class="nav-links">
            <a href="#home">Home</a>
            <a href="#about-us">About Us</a>
            <a href="#contact-us">Contact Us</a>
            <a href="#login" class="login-btn">Login/Sign Up</a>
        </div>
        <div class="burger" onclick="toggleNav()">
            <div></div>
            <div></div>
            <div></div>
        </div>
    </div>
    <script>
        function toggleNav() {{
            var navLinks = document.getElementById("nav-links");
            navLinks.classList.toggle("active");
        }}
    </script>
""", unsafe_allow_html=True)

# Hero Section with Background Image
if os.path.exists(hero_image_path):
    hero_image_base64 = get_image_base64(hero_image_path)
    st.markdown(
        f"""
        <div class="section"style='
            background: url(data:image/png;base64,{hero_image_base64}) no-repeat center center;'>
            <h1>Get Insights from Articles with CurioVeda<br>At your fingertips</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.error("Hero image not found!")

# Main Content Section
st.markdown("<div class='section-header'>Enter Your Article URLs and Ask Questions</div>", unsafe_allow_html=True)

# Left Section: URL Management
with st.container():
    left_section, col , right_section = st.columns([1.2,0.5,2.3])

    # Left Section
    with left_section:
        st.markdown('<h3 class="center-text">Enter the Article URL</h3>', unsafe_allow_html=True)

        # Text input box styled using the class "input-box"
        url_input = st.text_input("Enter URL:", placeholder="Paste URL here...", key="url_input", label_visibility="collapsed")

        # Add URL button
        if st.button("Add URL"):
            if len(st.session_state.urls) >= 10:
                st.warning("You can only add up to 10 URLs.")
            elif url_input.strip():
                add_url(url_input)
            else:
                st.warning("Please enter a valid URL.")

        # Display added URLs
        st.write("#### Added URLs:")
        for idx, url in enumerate(st.session_state.urls):
            # Wrap the URL display
            cols = st.columns([8, 1])
            formatted_url = "\n".join(url[i:i+40] for i in range(0, len(url), 40))  # Split URL into 80-character chunks
            cols[0].write(f"{idx + 1}. {formatted_url}")
            if cols[1].button("❌", key=f"delete_{idx}"):
                delete_url(idx)

        # Middle Section
        # Button to scrape content from all added URLs
        if st.button("Scrape All Content"):
            for url in st.session_state.urls:
                if url not in st.session_state.scraped_data:
                    with st.spinner("Scraping the website..."):
                        st.session_state.scraped_data[url] = scrape_url(url)
            st.success("Scraping completed!")
            # st.write("#### Scraped Data:")
            # for url, data in st.session_state.scraped_data.items():
            #     st.write(f"URL: {url}")
            #     st.write(data[:5000] + "..." if len(data) > 500 else data)

        if st.button("Start Learning"):
            with st.spinner("Learning from the Scrapped data..."):
                create_vector_store()

        else:
            st.warning("No vector embeddings available. Please scrape and process content first.")

        # Display scraped content
        st.write("#### Scraped Content:")
        if st.session_state.scraped_data:
            # for url, content in st.session_state.scraped_data.items():
            #     st.write(f"#### URL: {url}")
            #     st.write(content[:500] + "..." if len(content) > 500 else content)
            st.write("Scraped content is now available for further processing.")
        else:
            st.write("No content scraped yet.")

        
        if st.button("Clear All Data"):
            st.session_state.urls = []
            st.session_state.scraped_data = {}
            st.session_state.vector_store = None
            st.success("All data cleared.")

    # Right Section: Chatbot Interface
    with right_section:
        st.markdown('<h3 class="center-text">Chatbot</h3>', unsafe_allow_html=True)

        # User input question
        user_question = st.text_input("Ask a question:", placeholder="Type your question here...")
        if user_question:
            if user_question.strip():
                context = "\n".join(
                    [f"User: {entry['question']}\nBot: {entry['answer']}" for entry in st.session_state.chat_history]
                )
                context += f"\nUser: {user_question}"

                answer = query_bot_with_context(user_question, context)

                if answer:
                    st.session_state.chat_history.insert(0, {"question": user_question, "answer": answer})

                    st.write(f"**CurioVeda:** {answer['text'] if isinstance(answer, dict) else answer}")

                    table_data = detect_table_in_response(answer)
                    if table_data is not None:
                        st.write("### Data Table")
                        st.dataframe(table_data)
                        st.session_state.visualization_data = {"ready_for_visualization": True, "data": table_data}
                    else:
                        st.session_state.visualization_data = {"ready_for_visualization": False, "data": None}
                else:
                    st.write("**CurioVeda:** Hmm, I couldn't find that in the knowledge base.")
            else:
                st.warning("Please enter a question.")

        # Display chat history
        st.write("#### Chat History")
        if st.session_state.chat_history:
            for chat in st.session_state.chat_history[1:]:
                st.markdown(f"**You:** {chat['question']}")
                st.markdown(f"**CurioVeda:** {chat['answer']}")
        else:
            st.info("Start asking questions to see the chat history!")

        # Visualization Section
        if st.session_state.visualization_data["ready_for_visualization"]:
            st.write("### Generate Visualization")
            data = st.session_state.visualization_data["data"]

            # Button state management
            if "selected_chart" not in st.session_state:
                st.session_state.selected_chart = None

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if st.button("Bar Chart"):
                    st.session_state.selected_chart = "Bar Chart"
            with col2:
                if st.button("Line Chart"):
                    st.session_state.selected_chart = "Line Chart"
            with col3:
                if st.button("Pie Chart"):
                    st.session_state.selected_chart = "Pie Chart"
            with col4:
                if st.button("Histogram"):
                    st.session_state.selected_chart = "Histogram"

            # Generate and display the chart
            if st.session_state.selected_chart and data is not None:
                chart = generate_graph(data, st.session_state.selected_chart)
                st.plotly_chart(chart)

        # Clear Chat History
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.visualization_data = {"ready_for_visualization": False, "data": None}
            st.success("Chat history cleared.")
 

# Footer Section
st.markdown("<div class='footer'>© 2024 QueryHub. All Rights Reserved.</div>", unsafe_allow_html=True)