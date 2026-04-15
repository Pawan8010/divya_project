import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_community.llms import HuggingFacePipeline

# ---------------- UI ----------------
st.set_page_config(page_title="PDF QA App", page_icon="📄")

st.title("📄 PDF Question Answering App")
st.write("Upload a PDF and ask questions!")

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

# ---------------- MODEL LOADING ----------------
@st.cache_resource
def load_llm():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def generate_text(prompt):
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        outputs = model.generate(input_ids, max_new_tokens=256)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    class CustomPipeline:
        def __call__(self, inputs, **kwargs):
            if isinstance(inputs, list):
                return [[{'generated_text': generate_text(i)}] for i in inputs]
            return [{'generated_text': generate_text(inputs)}]

    return HuggingFacePipeline(pipeline=CustomPipeline())

# ---------------- PROCESS PDF ----------------
@st.cache_resource
def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    return retriever

# ---------------- MAIN LOGIC ----------------
if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF uploaded successfully!")

    retriever = process_pdf("temp.pdf")
    llm = load_llm()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    # ---------------- USER INPUT ----------------
    query = st.text_input("Ask a question:")

    if st.button("Get Answer"):
        if query:
            with st.spinner("Thinking..."):
                result = qa_chain.run(query)
            st.success(result)
        else:
            st.warning("Please enter a question")