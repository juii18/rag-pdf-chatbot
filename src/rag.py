import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq


load_dotenv()


def create_llm():

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )

    return llm


def ask_question(vector_store, question, llm):

    # Retrieve relevant documents
    results = vector_store.similarity_search(
        question,
        k=4
    )

    # No relevant information
    if not results:

        return (
            "I could not find this information in the PDF.",
            []
        )

    # Combine retrieved context
    context = "\n\n".join(
        document.page_content
        for document in results
    )

    prompt = f"""
You are an AI assistant that answers questions about an uploaded PDF.

Use ONLY the information provided in the context.

Rules:
- Do not make up information.
- Do not use outside knowledge.
- If the answer is not present in the context, say:
  "I could not find this information in the PDF."
- Give a clear and helpful answer.
- Keep the answer concise unless detailed explanation is necessary.

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    # Extract source pages
    source_pages = []

    for document in results:

        page = document.metadata.get("page")

        if page is not None:

            page_number = page + 1

            if page_number not in source_pages:
                source_pages.append(page_number)

    return response.content, source_pages