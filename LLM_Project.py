#!/usr/bin/env python
# coding: utf-8

# # Step 1: Import Libraries

# In[3]:


!pip install -q langchain langchain-community langchain-openai faiss-cpu pypdf sentence-transformers transformers accelerate

# In[ ]:


!pip install --upgrade transformers 

# In[3]:


!pip uninstall -y langchain langchain-core langchain-community
!pip install langchain==0.2.14 langchain-community==0.2.12 langchain-core==0.2.36

# # Step 2: Upload Document

# In[4]:


import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline
from langchain_community.llms import HuggingFacePipeline

from google.colab import files

# # Step 3: Upload Document

# In[34]:


uploaded = files.upload()

file_name = list(uploaded.keys())[0]
print("Uploaded:", file_name)

# # Step 4: Load Document

# In[35]:


loader = PyPDFLoader(file_name)
documents = loader.load()

print("Total pages:", len(documents))

# # Step 5: Text Chunking

# In[36]:


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

docs = text_splitter.split_documents(documents)
print("Total chunks:", len(docs))

# # Step 6: Create Embeddings + Vector Store

# In[37]:


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(docs, embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# # Step 7: Load LLM (FLAN-T5 BASE)

# In[38]:


import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

# Install compatible version (run once, then restart runtime if needed)
!pip install -q transformers==4.36.2 langchain langchain-community

# Imports
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.llms import HuggingFacePipeline

# Model name
model_name = "google/flan-t5-base"   # ✅ lightweight model

# Load tokenizer + model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Define a custom generation function to bypass transformers.pipeline task issues
def custom_generator(prompt, model, tokenizer, max_new_tokens, temperature):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        num_return_sequences=1,
        do_sample=True, # Allow sampling based on temperature
        top_k=50,       # For diverse outputs
        top_p=0.95      # For diverse outputs
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Create a minimal pipeline-like object that wraps the custom generator
class MinimalPipeline:
    def __init__(self, model, tokenizer, max_new_tokens, temperature):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.task = "text2text-generation" # Add the missing 'task' attribute

    def __call__(self, text_inputs, **kwargs):
        # HuggingFacePipeline expects a list of dictionaries with 'generated_text'
        if isinstance(text_inputs, list):
            results = []
            for prompt in text_inputs:
                generated_text = custom_generator(prompt, self.model, self.tokenizer, self.max_new_tokens, self.temperature)
                results.append([{'generated_text': generated_text}])
            return results
        else:
            generated_text = custom_generator(text_inputs, self.model, self.tokenizer, self.max_new_tokens, self.temperature)
            return [{'generated_text': generated_text}]

# Instantiate the minimal pipeline object
min_pipe = MinimalPipeline(
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    temperature=0.5
)

# Convert to LangChain LLM
llm = HuggingFacePipeline(pipeline=min_pipe)

# Test run
response = llm.invoke("Explain in simple terms: What is artificial intelligence?")
print(response)

# # Step 8: Build RAG QA System

# In[39]:


qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# # Step 9: Ask Questions

# In[40]:


def ask_question(query):
    result = qa_chain({"query": query})

    print("\n🟢 Answer:\n")
    print(result["result"])

    print("\n📚 Source Chunks:\n")
    for doc in result["source_documents"]:
        print("-", doc.page_content[:200], "...\n")

# # Step 10: Run Example Queries

# In[41]:


ask_question("Summarize this policy in simple language")
ask_question("What are the risks mentioned?")
ask_question("What are the key rules?")

# 
import sys

if __name__ == "__main__":
    user_input = sys.argv[1]

    # 🔥 CALL YOUR EXISTING FUNCTION HERE
    # Example:
    # result = qa_chain.run(user_input)

    result = "Replace with your model output"

    print(result)