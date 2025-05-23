import streamlit as st
import os
import re
from PyPDF2 import PdfReader
import docx
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from streamlit_chat import message
from langchain.prompts import PromptTemplate
from langchain.document_loaders import WebBaseLoader
from langchain.schema import Document

# Static PromptTemplate - we will dynamically change template text later
PROMPT = None

def main():
    st.set_page_config(page_title="Chat with Organic Docs", page_icon="🌿", layout="wide")

    file_mapping = {
        "land preparation": "land preparation.pdf",
        "Online Store (pakorganic.com)": "web"
    }

    file_options = list(file_mapping.keys())

    # Query parameters
    query_params = st.experimental_get_query_params()
    param_selected = query_params.get("option", [file_options[0]])[0]
    trigger = query_params.get("run", ["false"])[0].lower() == "true"

    # Choose last processed file if available, else from query, else default
    if "last_processed_file" in st.session_state and st.session_state.last_processed_file in file_options:
        selected_label = st.session_state.last_processed_file
    else:
        selected_label = param_selected

    # Fixed Header UI
    st.markdown(
        f"""
        <style>
            .fixed-header {{
                position: fixed;
                top: 4px;
                left: 0;
                right: 0;
                width: 95%;
                margin: auto;
                background-color: #4CAF50;
                padding: 25px;
                z-index: 999;
                border-radius: 10px 10px 0 0;
            }}
            .dropdown-row {{
                margin-top: 10px;
                display: flex;
                justify-content: center;
                align-items: center;
                gap: 10px;
            }}
            .stApp {{
                margin-top: 160px;
            }}
        </style>
        <div class="fixed-header">
            <h2 style="color: white; text-align:center;">🌿🌱 AI-Powered Chatbot 🤖 for Transition to Organic Farming 🌿🌱</h2>
            <p style="color: white; text-align:center;">Get instant answers from organic farming in the context of Pakistan.</p>
            <div class="dropdown-row">
                <form action="" method="get">
                    <label style="color:white;">Select your area of interest:</label>
                    <select name="option">
                        {''.join([f'<option value="{opt}" {"selected" if selected_label == opt else ""}>{opt}</option>' for opt in file_options])}
                    </select>
                    <input type="hidden" name="run" value="true">
                    <button type="submit">Process</button>
                </form>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    # Centered instruction below the banner


    st.markdown(
    """
    <div style='text-align: center; margin-top: 5px; font-size: 18px; color: #333;'>
        👉 <strong>Please select your area of interest from the dropdown above and click "Process" to begin.</strong>
    </div>
    """,
    unsafe_allow_html=True
    )
    
    if "chat_history_messages" not in st.session_state:
        st.session_state.chat_history_messages = []

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
        st.session_state.processComplete = False

    if "last_processed_file" not in st.session_state:
        st.session_state.last_processed_file = None

    if trigger and selected_label != st.session_state.last_processed_file:
        with st.spinner("Processing..."):
            openai_key = st.secrets["OPENAI_API_KEY"]
            file_key = file_mapping[selected_label]

            # 💡 Create prompt depending on source
            if file_key == "web":
                template = """
You are a helpful assistant. Use the following context as your main reference to answer the question. You may infer or summarize when helpful.

If the answer is not contained in the context, say 'Sorry, this question is out of my knowledge domain. I cannot answer this question.'

Always include a "Source" line at the end with the source URL from the context if available , otherwise source line will not be included.

Context:
{context}

Question:
{question}
"""
                urls = [
                    "https://pakorganic.com/",
                    "https://pakorganic.com/page/2/",
                    "https://pakorganic.com/organic-farming-consultancy/",
                    "https://pakorganic.com/our-projects/"
                ]
                documents = get_web_documents(urls)
            else:
                template = """
You are a helpful assistant. Use ONLY the following context to answer the question.
If the answer is not contained in the context, say 'Sorry, this question is out of my knowledge domain.' don't include source line in that case.

Context:
{context}

Question:
{question}
"""
                file_path = os.path.join(os.getcwd(), file_key)
                text = get_file_text(file_path)
                documents = [Document(page_content=text)]

            # ✅ Dynamic prompt
            prompt = PromptTemplate(input_variables=["context", "question"], template=template)

            text_chunks = get_text_chunks_with_metadata(documents)
            vectorstore = get_vectorstore_with_metadata(text_chunks)
            st.session_state.conversation = get_conversation_chain(vectorstore, openai_key, prompt)
            st.session_state.processComplete = True
            st.session_state.last_processed_file = selected_label
            st.session_state.chat_history_messages = []
            st.success("You may now chat with the selected content.")

    if st.session_state.processComplete and st.session_state.conversation:
        user_question = st.chat_input("Ask a question about the selected content:")
        if user_question:
            lower_question = user_question.strip().lower()

            # Handle small talk manually
            if is_small_talk(lower_question):
                answer = get_small_talk_reply(lower_question)
            else:
                with st.spinner("Getting your answer..."):
                    response = st.session_state.conversation({'question': user_question})
                    answer = response.get('answer', "Sorry, I couldn't find an answer to that question.")

            st.session_state.chat_history_messages.append({"role": "user", "content": user_question})
            st.session_state.chat_history_messages.append({"role": "bot", "content": answer})

        for i, chat in enumerate(st.session_state.chat_history_messages):
            message(chat["content"], is_user=(chat["role"] == "user"), key=f"chat_{i}")

# ========== Helper Functions ==========

def get_file_text(file_path):
    text = ""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        text += get_pdf_text(file_path)
    elif ext == ".docx":
        text += get_docx_text(file_path)
    return text

def get_pdf_text(file_path):
    with open(file_path, "rb") as f:
        reader = PdfReader(f)
        return "".join([page.extract_text() for page in reader.pages if page.extract_text()])

def get_docx_text(file_path):
    doc = docx.Document(file_path)
    return " ".join([para.text for para in doc.paragraphs])

def get_web_documents(urls):
    loader = WebBaseLoader(urls)
    return loader.load()

def get_text_chunks_with_metadata(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200
    )
    return splitter.split_documents(documents)

def get_vectorstore_with_metadata(documents):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(documents, embeddings)

def get_conversation_chain(vectorstore, openai_api_key, prompt):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-3.5-turbo', temperature=0)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

# === Small Talk Handling ===

def is_small_talk(message):
    small_talk_patterns = [
        r"^thanks?$",
        r"^thank\s?you$",
        r"^thankyou$",
        r"^hello$",
        r"^helo$",
        r"^thx$",
        r"^tnx$",
        r"^thanx$",
        r"^ok$",
        r"^okay$",
        r"^great$",
        r"^awesome$",
        r"^nice$",
        r"^cool$",
        r"^that('?s)? (great|good)$",
        r"^that was helpful$"
    ]
    return any(re.match(pattern, message) for pattern in small_talk_patterns)

def get_small_talk_reply(message):
    reply_map = {
        "thank you": "You're welcome!",
        "thankyou": "You're welcome!",
        "thanks": "You're welcome!",
        "thx": "You're welcome!",
        "tnx": "You're welcome!",
        "thanx": "You're welcome!",
        "hello": "You're welcome! how can I help you!",
        "helo": "You're welcome! how can I help you!",
        "ok": "👍",
        "okay": "👍",
        "great": "Glad to hear that!",
        "awesome": "Glad to hear that!",
        "nice": "Glad to hear that!",
        "cool": "Glad to hear that!",
        "that is great": "Awesome!",
        "that's great": "Awesome!",
        "that's good": "👍",
        "that was helpful": "I'm glad I could help!"
    }

    for key in reply_map:
        if re.match(rf"^{key}$", message):
            return reply_map[key]
    return "You're welcome!"  # Default fallback

if __name__ == '__main__':
    main()
