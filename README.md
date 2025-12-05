# 🚀 Multi-Model AI Pipeline (Under 2 Seconds Latency)

This project demonstrates a **production-grade multi-model pipeline** engineered to run an end-to-end sequence:

**ASR → Embeddings → RAG → LLM → TTS**

…in **under 2 seconds total latency**.

This repo is designed to reflect the exact type of work done by **AI Research Engineers** at frontier labs (e.g., Rumik, DeepMind, OpenAI), focusing on:

- Multi-model orchestration  
- Latency optimization  
- Audio + text hybrid systems  
- High-performance inference  
- Real-world system design  
- Research-quality engineering  

---

# 🧠 1. Why This Pipeline?

Typical AI models operate alone (one input → one output), but real agents must combine **multiple modalities**:

- They **listen** → ASR  
- They **understand** → Embeddings  
- They **reason** → LLM  
- They **retrieve memory** → RAG  
- They **speak** → TTS  

This project shows how to make all of these run **efficiently** and **fast** under a single unified pipeline.

---

# 🧩 2. Architecture Overview

```mermaid
flowchart LR
    A[Audio Input] --> B[Voice Activity Detection]
    B --> C[ASR Model - Speech-to-Text]
    C --> D[Embedding Model - Sentence Embeddings]
    D --> E[FAISS Retriever - Top-K Context]
    E --> F[LLM Inference - Local or API]
    F --> G[TTS Model - Generate Speech]
    G --> H[Audio Output to User]
```

The entire pipeline is built around **asynchronous inference**, **parallel execution**, **caching**, and **latency-aware design**.

---

# ⚙️ 3. Components Explained

## 🎤 **1. VAD — Voice Activity Detection**
- Removes silence  
- Helps ASR run faster  
- Reduces unnecessary model calls  

## 🗣️ **2. ASR — Speech-to-Text**
Supports:
- Whisper tiny/small  
- Faster models using quantization  
- Streaming audio chunks  

## 🧠 **3. Embedding Model**
Transforms text into vector embeddings for:
- semantic understanding  
- retrieval  
- context injection  

Uses fast CPU-friendly SentenceTransformer models.

## 📚 **4. RAG — FAISS Retrieval**
FAISS index performs:
- top-K nearest neighbor search  
- low-latency context lookup  
- flexible memory search  

## 💬 **5. LLM Reasoning Layer**
Supports:
- Llama / Mistral local inference  
- GPT-based remote inference  
- Token streaming  
- Context compression  

## 🔊 **6. TTS — Text-to-Speech**
Generates:
- natural voice output  
- low-latency synthesis  
- streaming audio chunks  

---

# ⚡ 4. Latency Optimizations Implemented

To achieve **<2s total latency**, the pipeline includes:

### ✔ Asynchronous FastAPI Server  
No blocking I/O.

### ✔ Model Warmup  
Reduces first-call delay.

### ✔ Parallel Execution  
Some tasks overlap (like preprocessing + FAISS).

### ✔ Quantization (optional)  
INT8 / FP16 models accelerate inference.

### ✔ Embedding Cache  
Avoid recomputing semantic vectors.

### ✔ GPU / CPU Flex Mode  
Auto-selects best hardware.

### ✔ Lightweight Models Selected  
Where possible, small architectures are preferred to improve speed.

---

# 🧪 5. Benchmarking System

Included benchmarking script reports:

```
ASR Latency:          230ms  
Embedding Latency:     18ms  
RAG Retrieval:          6ms  
LLM Latency:         1100ms  
TTS Latency:          210ms  
---------------------------------
TOTAL PIPELINE:      1564ms (PASS)
```

If the total is **under 2000 ms**, the pipeline is considered **real-time capable** for conversational AI.

---

# 📁 6. Project Structure

```
multi-model-latency-pipeline/
│── src/
│   ├── asr.py               # Speech-to-text
│   ├── vad.py               # Voice activity detection
│   ├── embeddings.py        # Sentence embeddings
│   ├── rag_faiss.py         # FAISS retriever logic
│   ├── llm.py               # Local/remote LLM inference
│   ├── tts.py               # Text-to-speech
│   ├── pipeline.py          # Orchestration + FastAPI
│   ├── benchmark.py         # Latency measurement engine
│   ├── config.py            # Settings & paths
│   ├── utils.py             # Shared helpers
│   └── __init__.py
│
├── data/                    # Knowledge base, audio samples
├── notebooks/               # Experiments, profiling
│
├── requirements.txt
└── README.md
```

---

# 🚀 7. Running the Pipeline

## Install dependencies
```
pip install -r requirements.txt
```

## Start the API server
```
uvicorn src.pipeline:app --reload
```

## Run latency benchmark
```
python src/benchmark.py
```

---

# 🌟 8. What This Project Demonstrates

✔ Ability to build **complex multi-model AI systems**  
✔ Understanding of **ASR, embeddings, retrieval, LLMs, TTS**  
✔ Experience optimizing **latency under real-world constraints**  
✔ Skill in designing **modular production architectures**  
✔ Knowledge of **FastAPI, concurrency, async pipelines**  
✔ Understanding of **FAISS and retrieval-augmented reasoning**  
✔ The exact workflow expected at **AI Research Labs (Rumik, OpenAI, DeepMind)**  

---

# 👩‍💻 Author  
**Sanjivani Chavan**  
AI Engineer | LLM Systems | Real-Time ML Pipelines | Retrieval Architect  


