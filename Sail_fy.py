# Libraries
import os
import fitz  # PyMuPDF for PDF extraction
import yaml
from tqdm import tqdm
from pprint import pprint
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Load OpenAI API Key
OPENAI_API_KEY = yaml.safe_load(open('credentials.yml'))['openai']

# Paths
pdf_directory = "data/pdf_folder"
chroma_db_path = "data/chroma_store"

# Ensure the ChromaDB directory exists
if not os.path.exists(chroma_db_path):
    os.makedirs(chroma_db_path)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text("text")
    return text

# Load and extract text from PDFs
documents = []
for filename in tqdm(os.listdir(pdf_directory)):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_directory, filename)
        text = extract_text_from_pdf(pdf_path)
        documents.append({"text": text, "source": filename})
print(documents)

# Initialize OpenAI embeddings
embedding = OpenAIEmbeddings(model='text-embedding-ada-002', api_key=OPENAI_API_KEY)


# Split text into manageable chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
split_texts, metadatas = [], []
for doc in documents:
    chunks = splitter.split_text(doc["text"])
    for chunk in chunks:
        split_texts.append(chunk)
        metadatas.append({'source': doc['source']})

# Create embeddings for the text chunks
embeddings = embedding.embed_documents(split_texts)

# Persist embeddings into ChromaDB
chroma = Chroma(embedding_function=embedding, persist_directory=chroma_db_path)
chroma.add_texts(texts=split_texts, embedding=embeddings, metadatas=metadatas)
chroma.persist()

# Similarity search
query = "What are the most important events in SAIL?"
result = chroma.similarity_search(query, k=4)

pprint(result[0].page_content)

from pprint import pprint

for i, res in enumerate(result):
    # Accessing the text and metadata directly
    pprint(res.text if hasattr(res, 'text') else 'No content available')
    print(f"Source: {res.metadata.get('source', 'Unknown source') if hasattr(res, 'metadata') else 'Unknown source'}\n")
