# 📄 PDFHelper (JavaScript Edition)

A **Node.js (Express)** application to process PDF documents, extract text, generate embeddings using **Google's Gemini**, store them in **MongoDB Atlas** with vector search, and answer questions using **Retrieval-Augmented Generation (RAG)**.

---

## 🚀 Overview

This JavaScript version replicates the core features of the Python-based pdfhelper project, fulfilling the **HackRX challenge** requirements.

- Downloads a PDF via URL  
- Extracts and chunks its content  
- Generates embeddings using Gemini  
- Stores data in MongoDB with vector search  
- Answers questions using **Gemini** or **Groq** LLMs

---

## 🔧 Features

- 📥 **PDF Processing**: Uses `pdf-parse` to extract text
- 🧩 **Text Chunking**: Overlapping chunks for better context
- 🧠 **Embedding Generation**: Uses `text-embedding-004` from Gemini (768 dimensions)
- 🗃️ **MongoDB Atlas Integration**: Stores chunks with vector indexing
- 🧠 **RAG Pipeline**: Answers via Gemini or Groq based on user query
- ⚡ **Express API Endpoint**: Single `/hackrx/run` endpoint

---

## 📋 Prerequisites

- **Node.js**: Version 16+
- **MongoDB Atlas**: Cluster with vector search enabled
- **API Keys**:
  - `GEMINI_KEY`: Google Gemini API key
  - `GROQ_API_KEY`: Groq API key (optional)
  - `MONGO_URI`: MongoDB Atlas connection URI

---

## 📦 Installation

```bash
git clone https://github.com/Parvsharma04/PDFHelper-js.git
cd PDFHelper-js
npm install
```

### Environment Variables

Create a `.env` file in the root folder:

```
GEMINI_KEY=your-gemini-key
GROQ_API_KEY=your-groq-key
MONGO_URI=your-mongo-uri
```

---

## ▶️ Run the App

```bash
node app.js
```

> The app runs on `http://localhost:3000`

---

## 📤 Example Request

### POST `/hackrx/run`

```json
{
  "documents": "https://example.com/sample.pdf",
  "questions": [
    "What is the main topic of the document?",
    "Who is the author of the document?"
  ]
}
```

### Sample Response

```json
{
  "answers": [
    "The document mainly discusses XYZ.",
    "The author is not specified in the context."
  ]
}
```

---

## 📡 API Endpoints

| Method | Endpoint       | Description                    |
|--------|----------------|--------------------------------|
| GET    | `/`            | Health check                   |
| POST   | `/hackrx/run`  | Process PDF and answer questions |

---

## 🧱 Key Dependencies

| Package | Purpose |
|---------|---------|
| `express` | HTTP server |
| `axios` | PDF download |
| `pdf-parse` | Text extraction |
| `@google/generative-ai` | Gemini embedding + LLM |
| `groq-sdk` | Groq LLM |
| `mongodb` | MongoDB integration |
| `dotenv` | Env vars |
| `uuid` | Document ID generation |

Install with:

```bash
npm install express axios pdf-parse @google/generative-ai groq-sdk mongodb dotenv uuid
```

---

## ⚙️ Configuration

- **Chunk Size**: 10000 characters  
- **Overlap**: 2000 characters  
- **Embedding Dimension**: 768  
- **Gemini LLM**: `gemini-1.5-flash`  
- **Embedding Model**: `text-embedding-004`  
- **Groq LLM**: `llama3-8b-8192`

---

## 🛠 Error Handling

- ❌ Invalid or broken PDF links
- 📄 Empty or scanned PDFs (OCR not supported)
- 🧠 Embedding errors from Gemini
- 🔌 MongoDB or connection issues

Returns meaningful JSON error responses.

---

## 🧩 Notes

- Gemini is **required** for embeddings (even if you use Groq for answers)
- Groq is **optional**
- Vector search must be enabled in MongoDB
- Only text-based PDFs are supported

---

## 🧰 Troubleshooting

| Issue | Fix |
|-------|-----|
| MongoDB errors | Check `MONGO_URI`, IP access, and index |
| No text extracted | Check if PDF is scanned |
| Embedding mismatch | Ensure model used is `text-embedding-004` |
| API key errors | Check `.env` file format |

---

## 📜 License

Licensed under the **MIT License**
