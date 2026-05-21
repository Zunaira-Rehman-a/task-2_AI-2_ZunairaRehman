import ollama

def generate_response(query, retrieved_results):
    sources = set()
    context_str = ""

    # ChromaDB results standard format keys: 'documents' and 'metadatas'
    if retrieved_results and 'documents' in retrieved_results and retrieved_results['documents']:
        # ChromaDB nested lists ki wajah se hum pehle index [0] ko nikalte hain
        documents = retrieved_results['documents'][0]
        metadatas = retrieved_results['metadatas'][0] if 'metadatas' in retrieved_results else None

        for i, doc in enumerate(documents):
            source_name = "Unknown"
            if metadatas and i < len(metadatas) and metadatas[i]:
                source_name = metadatas[i].get("source", "teyzix_policy.txt")
            
            sources.add(source_name)
            context_str += f"[Source: {source_name}]\n{doc}\n\n"

    # Agar ooper ka loop sahi na chale aur context khali reh jaye to plain string handling fallback
    if not context_str and retrieved_results:
        context_str = str(retrieved_results)
        sources.add("teyzix_policy.txt")

    system_prompt = (
        "You are a strict assistant.\n"
        "Answer the user's question ONLY from the provided context.\n"
        "If the answer is not present in the context, say:\n"
        "'I cannot find the answer in the provided documents.'\n\n"
        f"Context:\n{context_str}"
    )

    print("Formatting prompt and sending request to local LLM (Llama3.2)...")

    try:
        response = ollama.chat(
            model="llama3.2",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
        )
        answer = response["message"]["content"]
        return answer, list(sources)
    except Exception as e:
        print(f"Ollama Error: {e}")
        return f"Ollama execution error: {str(e)}", list(sources)