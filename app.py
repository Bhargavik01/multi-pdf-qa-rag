import os
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_community.vectorstores import Pinecone as LangChainPinecone
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from langchain.prompts import PromptTemplate

load_dotenv()

# Initialize Pinecone client with the new SDK
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "multipdfrag"
index_names = [index.name for index in pc.list_indexes()]
# Check if index exists, create if not
if index_name not in index_names:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Connect to the index
index = pc.Index(index_name)
embedding = OpenAIEmbeddings()

# Load and embed PDFs, add metadata 'source' for each chunk
def load_and_index_pdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    chunks = text_splitter.split_documents(docs)

    for doc in chunks:
        doc.metadata["source"] = os.path.basename(file_path)

    return chunks

# Load all PDFs and combine chunks into one list
all_chunks = []
pdf_files = [
    "documents/HR_Handbook.pdf",
    "documents/Sales_Playbook.pdf",
    "documents/Security_Protocol.pdf"
]

for pdf in pdf_files:
    all_chunks.extend(load_and_index_pdf(pdf))

# Upsert all chunks into Pinecone via LangChain wrapper (this will embed and upload)
vectorstore = LangChainPinecone.from_documents(all_chunks, embedding, index_name=index_name)

# Helper to query Pinecone vectorstore and format results
def get_answer(store, query: str) -> str:
    results = store.similarity_search(query, k=1)
    output = ""
    for doc in results:

        source = doc.metadata.get("source", "Unknown Source")
        output += f"[Source: {source}]\n{doc.page_content}\n\n"

    return output.strip()

# Setup OpenAI chat model
llm = ChatOpenAI(temperature=0, model='gpt-4o')

print("Ready for questions. Type 'exit' to quit.")

while True:
    query = input("\nAsk a Question: ")
    if query.lower() == "exit":
        break

    # Search relevant chunks
    answer_text = get_answer(vectorstore, query)

    # Prepare prompt with context + user question
    prompt_text = """
You are a helpful assistant. Use the context provided to answer the user question.
For each part of your answer, mention the source file from which the information came in the first line.
Only answer from the context given. Don't use your knowledge
Context:
{context}

Question:
{question}
"""

    prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_text)
    final_prompt = prompt.format(context=answer_text, question=query)

    # Get answer from LLM
    response = llm.invoke(final_prompt)

    print("\n Answer:\n", response.content)
