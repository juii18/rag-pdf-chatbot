import os
import tempfile

import streamlit as st

from pdf_loader import load_pdf
from chunking import split_documents
from embeddings import get_embeddings
from vector_store import create_vector_store
from rag import create_llm, ask_question


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="PDF AI Chat",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background: #f7f9fc;
}

section[data-testid="stSidebar"] {
    background: white;
    border-right: 1px solid #e5e7eb;
}

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 6rem;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

/* Main title */

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    color: #172033;
    margin-top: 20px;
    margin-bottom: 8px;
}

.main-subtitle {
    text-align: center;
    font-size: 17px;
    color: #64748b;
    margin-bottom: 35px;
}

/* Feature cards */

.feature-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 25px;
    text-align: center;
    min-height: 145px;
    box-shadow: 0 3px 12px rgba(0,0,0,0.04);
}

.feature-icon {
    font-size: 30px;
    margin-bottom: 10px;
}

.feature-title {
    font-size: 18px;
    font-weight: 600;
    color: #172033;
}

.feature-text {
    color: #64748b;
    font-size: 14px;
    margin-top: 7px;
}

/* Sidebar */

.sidebar-heading {
    font-size: 24px;
    font-weight: 700;
    color: #172033;
}

.sidebar-description {
    color: #64748b;
    line-height: 1.5;
}

/* Chat */

div[data-testid="stChatMessage"] {
    border-radius: 16px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "page_count" not in st.session_state:
    st.session_state.page_count = 0

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0


# ============================================================
# CACHE MODELS
# ============================================================

@st.cache_resource
def get_embedding_model():
    return get_embeddings()


@st.cache_resource
def get_llm():
    return create_llm()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-heading">📚 PDF AI Chat</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-description">'
        'Upload a PDF and chat with your document using '
        'Retrieval-Augmented Generation.'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.subheader("📄 Upload Document")

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        help="Upload a PDF document to start chatting."
    )

    if uploaded_file:

        if uploaded_file.name != st.session_state.document_name:

            with st.spinner("Processing your PDF..."):

                try:

                    # ----------------------------------------
                    # Save uploaded PDF
                    # ----------------------------------------

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".pdf"
                    ) as temp_file:

                        temp_file.write(
                            uploaded_file.getbuffer()
                        )

                        temp_path = temp_file.name

                    # ----------------------------------------
                    # Load PDF
                    # ----------------------------------------

                    documents = load_pdf(temp_path)

                    # ----------------------------------------
                    # Create chunks
                    # ----------------------------------------

                    chunks = split_documents(documents)

                    # ----------------------------------------
                    # Create embeddings
                    # ----------------------------------------

                    embedding_model = get_embedding_model()

                    # ----------------------------------------
                    # Create vector store
                    # ----------------------------------------

                    vector_store = create_vector_store(
                        chunks,
                        embedding_model
                    )

                    # ----------------------------------------
                    # Save state
                    # ----------------------------------------

                    st.session_state.vector_store = vector_store

                    st.session_state.document_name = (
                        uploaded_file.name
                    )

                    st.session_state.page_count = (
                        len(documents)
                    )

                    st.session_state.chunk_count = (
                        len(chunks)
                    )

                    st.session_state.messages = []

                    # Remove temporary PDF

                    os.unlink(temp_path)

                    st.success(
                        "PDF processed successfully!"
                    )

                except Exception as e:

                    st.error(
                        f"PDF processing failed:\n\n{e}"
                    )


    # ========================================================
    # DOCUMENT STATUS
    # ========================================================

    st.divider()

    st.subheader("📊 Document Status")

    if st.session_state.vector_store:

        st.success("Document ready")

        st.write(
            f"📄 **{st.session_state.document_name}**"
        )

        st.write(
            f"📑 {st.session_state.page_count} pages"
        )

        st.write(
            f"🔎 {st.session_state.chunk_count} searchable sections"
        )

    else:

        st.info(
            "Upload a PDF to start chatting."
        )


    # ========================================================
    # EXAMPLES
    # ========================================================

    st.divider()

    st.subheader("💡 Try asking")

    st.write("📌 What is this document about?")

    st.write("🔍 Explain the main concepts.")

    st.write("📝 Summarize this document.")

    st.write("🎯 What are the important topics?")

    st.write("📚 Explain this topic in simple words.")


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="main-title">📚 Chat with your PDF</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-subtitle">'
    'Ask questions and get answers grounded in your document.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# EMPTY STATE
# ============================================================

if st.session_state.vector_store is None:

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            '<div class="feature-card">'
            '<div class="feature-icon">🔎</div>'
            '<div class="feature-title">Find Information</div>'
            '<div class="feature-text">'
            'Search through your PDF using AI.'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            '<div class="feature-card">'
            '<div class="feature-icon">🤖</div>'
            '<div class="feature-title">AI Answers</div>'
            '<div class="feature-text">'
            'Get natural answers grounded in your PDF.'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            '<div class="feature-card">'
            '<div class="feature-icon">📖</div>'
            '<div class="feature-title">Source Pages</div>'
            '<div class="feature-text">'
            'See which PDF pages support the answer.'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )

    st.info(
        "👈 Upload a PDF from the sidebar to start chatting."
    )


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            pages = message["sources"]

            st.caption(
                "📖 Sources: "
                + ", ".join(
                    f"Page {page}"
                    for page in pages
                )
            )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask anything about your PDF..."
)


if question:

    # --------------------------------------------------------
    # USER
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)


    # --------------------------------------------------------
    # AI
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching your PDF..."
        ):

            try:

                llm = get_llm()

                answer, sources = ask_question(
                    st.session_state.vector_store,
                    question,
                    llm
                )

                st.markdown(answer)

                if sources:

                    st.caption(
                        "📖 Sources: "
                        + ", ".join(
                            f"Page {page}"
                            for page in sources
                        )
                    )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    }
                )

            except Exception as e:

                error = (
                    "Sorry, I couldn't process your "
                    f"question.\n\nError: {e}"
                )

                st.error(error)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error,
                        "sources": []
                    }
                )