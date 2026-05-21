import os
from pypdf import PdfReader

def load_documents(directory_path="./data"):
    documents = []
    if not os.path.exists(directory_path): 
        os.makedirs(directory_path)
    
    # Line 9: Added terminal log for starting document loading
    print(f"Scanning directory '{directory_path}' for documents...")
    
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if filename.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                documents.append({"text": f.read(), "source": filename})
                print(f"Successfully loaded text file: {filename}")
        elif filename.endswith('.pdf'):
            reader = PdfReader(file_path)
            text = "".join([page.extract_text() or "" for page in reader.pages])
            documents.append({"text": text, "source": filename})
            print(f"Successfully loaded PDF file: {filename}")
            
    return documents

def chunk_documents(documents, chunk_size=500, chunk_overlap=50):
    chunks = []
    # Line 27: Added terminal log for text chunking process
    print("Splitting documents into text chunks...")
    
    for doc in documents:
        words = doc["text"].split()
        for i in range(0, len(words), chunk_size - chunk_overlap):
            chunk_text = " ".join(words[i:i + chunk_size])
            chunks.append({"text": chunk_text, "metadata": {"source": doc["source"]}})
            
    # Line 35: Added final count log
    print(f"Total chunks created: {len(chunks)}")
    return chunks