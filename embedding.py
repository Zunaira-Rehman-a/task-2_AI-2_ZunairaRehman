import chromadb
from sentence_transformers import SentenceTransformer

class VectorDBManager:
    def __init__(self):
        # Line 7: Persistent storage configuration
        self.client = chromadb.PersistentClient(path="./chroma_db")
        print("Initializing Embedding Model (all-MiniLM-L6-v2)...")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.collection = self.client.get_or_create_collection("rag_collection")
        print("ChromaDB persistent client and collection ready.")

    def add_documents(self, chunks):
        if not chunks: 
            print("No chunks provided for indexing.")
            return
            
        documents = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        ids = [f"id_{i}" for i in range(len(chunks))]
        
        # Line 21: Logging embedding generation
        print(f"Generating embeddings for {len(chunks)} text chunks...")
        embeddings = self.embedding_model.encode(documents).tolist()
        
        print("Storing vectors and metadata into ChromaDB...")
        self.collection.add(embeddings=embeddings, documents=documents, metadatas=metadatas, ids=ids)
        print("Database indexing completed successfully.")

    def query_db(self, query_text):
        # Line 30: Logging retrieval query
        print(f"Searching vector database for query: '{query_text}'")
        query_embedding = self.embedding_model.encode([query_text]).tolist()
        results = self.collection.query(query_embeddings=query_embedding, n_results=3)
        print("Matching context retrieved from ChromaDB.")
        return results