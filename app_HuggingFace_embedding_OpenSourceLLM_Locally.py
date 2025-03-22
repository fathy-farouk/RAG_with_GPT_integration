# **** Load and Preprocess Data **** ---------------------------------------------------------------------------------------------
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.llms import CTransformers  # ✅ Local Open-Source LLM
import os
import shutil
import faiss
import numpy as np

# ✅ Initialize Open-Source LLM (Example: LLaMA, Mistral, GPT-J)
llm = CTransformers(model="TheBloke/Llama-2-7B-Chat-GGML", model_type="llama")

# ✅ Initialize Hugging Face Embeddings (Still used for FAISS)
from langchain.embeddings import HuggingFaceEmbeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ Load PDF File
pdf_path = "Fantastic family hotel_review.pdf"  # Replace with your file
loader = PyMuPDFLoader(pdf_path)
docs = loader.load()

# ✅ Split Text into Smaller Chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = text_splitter.split_documents(docs)

# ✅ Convert Extracted Text into `Document` Objects
doc_objects = [Document(page_content=doc.page_content) for doc in split_docs]

# ✅ Ensure FAISS Directory Exists
faiss_index_path = "faiss_index"
os.makedirs(faiss_index_path, exist_ok=True)

# ✅ Delete Existing FAISS Index to Avoid Old Data Conflicts
if os.path.exists(faiss_index_path):
    shutil.rmtree(faiss_index_path)  # Removes old FAISS index directory

os.makedirs(faiss_index_path, exist_ok=True)  # Recreate directory

# ✅ Create and Save FAISS Index
vector_db = FAISS.from_documents(doc_objects, embedding_model)
vector_db.save_local(faiss_index_path)

print("\033[1;32m✅ FAISS index saved correctly with both `index.faiss` and `index.pkl`!\033[0m")

# ✅ Load FAISS Vector Database Safely
vector_db = FAISS.load_local(faiss_index_path, embeddings=embedding_model, allow_dangerous_deserialization=True)
retriever = vector_db.as_retriever()

print("\033[1;32m✅ FAISS Index Loaded Successfully!\033[0m")

#######################################################################################################
# **Run a Query Using the FAISS Index** ------------------------------------------------------------------------------------------
# ✅ Example Query
query = "who is president of ahly egyptian club"

# ✅ Perform FAISS search and return **distance values**
retrieved_docs_with_scores = vector_db.similarity_search_with_score(query, k=1)  # ✅ Retrieve docs with scores

# ✅ Check if FAISS found **relevant** documents
retrieved_docs = []
scores = []
for doc, score in retrieved_docs_with_scores:
    similarity_score = 1 / (1 + score)  # ✅ Convert L2 distance to similarity
    retrieved_docs.append(doc)  # ✅ Store document
    scores.append(similarity_score)  # ✅ Store converted similarity score

# ✅ Define Score Threshold
score_threshold = 0.5

# ✅ Display Score Information and Decide the Answering Mode
if scores:
    score_value = scores[0]
    print(f"\n\033[1;34m🔹 Normalized Similarity Score for FAISS Retrieval: {score_value:.2f}\033[0m")

    if score_value >= score_threshold:
        print("\n\033[1;32m✅ High Match (Similarity > 0.5), FAISS (RAG) will answer your query.\033[0m")
        use_faiss = True
    else:
        print("\n\033[1;31m❌ Low Match (Similarity < 0.5), Local LLM will answer your query.\033[0m")
        use_faiss = False

#########################################################################################################################
# ✅ Define System Prompt to Improve Local LLM Accuracy
system_prompt = """
You are an AI assistant specialized in answering user queries based on retrieved documents.
Follow these instructions:
1. If FAISS provides relevant context, use only that information to generate a response.
2. If FAISS does not return relevant documents, rely on your general knowledge.
3. Be concise, clear, and provide factual answers only.
4. If the query is unclear, ask for clarification.
5. Do not make up information beyond the provided context.
"""

# ✅ Use Local LLM to Answer the Query Based on Retrieved Context
if not use_faiss:  # ✅ If FAISS is irrelevant, rely **only on Local LLM**
    prompt = f"Answer the following question:\n\nQuery: {query}\n\nAnswer:"
    gpt_response = llm.predict(prompt)
else:  
    gpt_response = retrieved_docs[0].page_content  # ✅ Direct FAISS answer, no LLM involved

# ✅ Print Final Answer Based on Decision
if use_faiss:
    # ✅ FAISS is relevant → Only FAISS answer
    print("\n\033[1;33m📌 **Best Retrieved Document Answer (FAISS):**\033[0m\n")
    best_doc = retrieved_docs[0]  # Select the most relevant document
    print(f"\033[1m{best_doc.page_content}\033[0m")  # Bold text output

else:
    # ✅ FAISS is not relevant → Only Local LLM answer
    print("\n\033[1;36m🤖 **Local LLM-Generated Answer:**\033[0m\n")
    print(f"\033[1m{gpt_response.strip()}\033[0m")  
