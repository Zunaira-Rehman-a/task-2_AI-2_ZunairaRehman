import streamlit as st
from src.ingestion import load_documents, chunk_documents
from src.embedding import VectorDBManager
from src.llm import generate_response

st.set_page_config(page_title="RAG Chatbot")
db_manager = VectorDBManager()

st.title("🤖 Local RAG Chatbot")

with st.sidebar:
    if st.button("🚀 Process & Index Documents"):
        docs = load_documents()
        if docs:
            chunks = chunk_documents(docs)
            db_manager.add_documents(chunks)
            st.success("Documents successfully processed and indexed into ChromaDB!")
        else:
            st.warning("Please place your PDF documents in the 'data/' folder to proceed.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if user_query := st.chat_input("Ask about your documents..."):
    st.chat_message("user").markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    with st.chat_message("assistant"):
        retrieved = db_manager.query_db(user_query)
        if retrieved and retrieved['documents'][0]:
            answer, sources = generate_response(user_query, retrieved)
            full_response = f"{answer}\n\n**📄 Sources:** {', '.join(sources)}"
        else:
           full_response = "Please process and index the documents from the sidebar first."
        
        st.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})