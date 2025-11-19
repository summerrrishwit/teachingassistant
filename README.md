# 🎥 Video QA App with LLM + RAG + Multimodal Reasoning

An interactive Streamlit app where users can:

* Upload a video
* Pause at any timestamp and ask questions
* Extract frames and transcript around that moment
* Use a Gemma-3 LLM to answer based on vision + language
* Generate key concept notes from the transcript

---

## 🛠️ Setup Instructions

### 1. **Install Ollama and Pull Model**

Install [Ollama](https://ollama.com/):

```bash
brew install ollama  # or follow instructions for Linux/Windows
```

Pull the Gemma 3 model:

```bash
ollama pull gemma3:4b
```

---

### 2. **Clone the Repository and Install Python Requirements**

```bash
git clone https://github.com/your-username/video-qa-app.git
cd video-qa-app
pip install -r requirements.txt
```

---

### 3. **Run the Application**

```bash
streamlit run app/main.py
```

---

## 💡 Features

* 🔼 Upload videos (`.mp4`, `.webm`, `.mov`)
* ⏱️ Enter timestamp in `HH:MM:SS`, `MM:SS`, or `SS` format
* 🧠 Ask LLM questions about that moment using:

  * Local transcript window (Whisper)
  * Global context via summarized or retrieved chunks (RAG)
  * Visual frames
* 📌 Multi-turn conversation memory
* 📝 Extract notes as:

  * Bullet points
  * Summaries
  * Q\&A pairs

---

## 📂 Project Structure

```
├── app/
│   ├── main.py                 # Streamlit frontend logic
│   ├── config.py               # Config paths (VIDEO_PATH, FRAME_DIR)
│   ├── video_utils.py          # Save and extract frames
│   ├── transcript_utils.py     # Whisper transcription, summarization
│   ├── llm_utils.py            # LLM prompt + response wrapper
│   └── rag_utils.py            # Chunking + FAISS vector search (if used)
├── requirements.txt
└── README.md
```

---

## 🤖 Model Details

* Uses `gemma3:4b` pulled via Ollama
* Incorporates both transcript and image frames for answering
* Prompt templates vary by task (QA, summarization, bullet points)

---

## 📋 Coming Soon

* ✅ Interactive pause capture (JS-integrated video player)
* ✅ Export notes to PDF/Markdown
* 🔍 Visual timeline navigation
* 🔁 Persistent RAG index across sessions
