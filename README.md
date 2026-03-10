# Query BOT – Voice Assistant with RAG

An **AI-powered voice and text assistant** designed to answer questions from **preloaded organizational data** using **Retrieval-Augmented Generation (RAG)**.

The assistant retrieves relevant information from a **preprocessed dataset (PDF knowledge base)** and generates contextual responses using an **LLM**. If the requested information is not available in the dataset, the system automatically falls back to a **general-purpose language model** to provide an answer.

The system also supports **voice input, text-to-speech output, and persistent chat history storage**.

## Features

### Knowledge-Based Question Answering
- Answers user queries using **preloaded organizational data**.
- Data is processed and stored as **vector embeddings** for efficient retrieval.

### Retrieval-Augmented Generation (RAG)
- Retrieves relevant information from the dataset using **FAISS vector similarity search**.
- Combines retrieved context with the **LLM** to generate accurate responses.

### LLM Fallback
- If no relevant information exists in the dataset, the assistant **queries the general LLM** to generate an answer.

### Voice Interaction
- **Speech-to-text** input for asking questions via microphone.
- **Text-to-speech** output for spoken responses.

### Chat History Storage
- Stores conversations using **SQLite database**.
- Allows viewing of previous interactions.

### Interactive Web Interface
- Built with **Streamlit** for real-time chat interaction.

## Tech Stack

**Programming Language**
- Python

**AI / ML**
- LangChain
- FAISS Vector Store
- Retrieval-Augmented Generation (RAG)

**Large Language Model**
- ChatGroq API

**Speech Processing**
- SpeechRecognition  
- pyttsx3

**Frontend**
- Streamlit

**Database**
- SQLite

## Installation
Install dependencies

pip install -r requirements.txt

## Environment Variables

Create a `.env` file in the project root. and iside that store the API key.

Example:

key=GROQ_API_KEY


## Running the Application

Start the Streamlit server: streamlit run main.py

The application will open in browser.

## How It Works
1. Organizational data is **preloaded and processed into vector embeddings**.
2. User asks a question using **text or voice input**.
3. The system performs **vector similarity search** to retrieve relevant information.
4. Retrieved context is passed to the **LLM for response generation**.
5. If no relevant information exists, the assistant **queries the general LLM**.
6. The answer is returned to the user and optionally **spoken using text-to-speech**.

## Possible Applications
- Organizational knowledge assistant  
- Institutional information bot  
- Voice-enabled helpdesk systems  
- Internal documentation chatbot  
- Domain-specific conversational assistants  

## Future Improvements

- Use **better embedding models**
- Add **user authentication**
- Deploy using **Docker or cloud platforms**
- Improve retrieval accuracy with **hybrid search**
