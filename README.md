# multi-pdf-qa-rag
LangChain + Pinecone + OpenAI Multiple PDF QA project 

# PDF QA Chatbot with LangChain, Pinecone, and OpenAI

This project loads multiple PDFs, splits and embeds their content, stores embeddings in Pinecone, and allows querying using GPT-4o via LangChain.

## 📂 Project Structure

```text
pdf-qa-chatbot/
├── documents/
│   ├── HR_Handbook.pdf
│   ├── Sales_Playbook.pdf
│   └── Security_Protocol.pdf
├── .env
├── requirements.txt
├── app.py
└── README.md

🔧 Setup
Clone the repo
git clone https://github.com/YOUR_USERNAME/pdf-qa-chatbot.git
cd pdf-qa-chatbot

Install dependencies
pip install -r requirements.txt

Set your API keys in a .env file 
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key

Add your PDF files to the documents/ folder
Run the chatbot
python main.py

✅ Features
Chunk and embed multiple PDFs
Store and search embeddings using Pinecone
Query using OpenAI GPT-4o with context
Display source file of the response

📌 Notes
Only the provided PDF chunks will be used for answering questions.
Happy Chatting! 💬

