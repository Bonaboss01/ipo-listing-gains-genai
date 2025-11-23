# IPO Listing Gains Prediction – GenAI Edition (Version 2)

A complete **end-to-end ML + Generative AI project** that predicts IPO listing gains and explains the reasoning in plain language.

This is **Version 2** of my original IPO project.  
Version 1 focused on:
- Exploratory Data Analysis (EDA)
- Classical machine learning
- A simple deep learning model

Version 2 turns it into a **full GenAI system** with:
- Better-structured ML/DL pipeline
- LLM-powered explanations
- RAG (Retrieval-Augmented Generation) over IPO documents
- A simple app for interactive use

> ⚠️ **Disclaimer:** This project is for learning and demonstration only.  
> It is **not** financial advice and must not be used for real trading or investment decisions.

---

## 🎯 Project Goal

Given information about an IPO (issue price, issue size, oversubscription, sector, etc.), the system:

1. **Predicts whether the IPO is likely to give listing gains** (e.g. “Gain” vs “No Gain” or percentage gain range).
2. **Explains the prediction** in simple English using a Large Language Model (LLM).
3. **Answers questions about specific IPOs** (e.g. “What are the key risks?”) using a RAG pipeline over IPO-related documents.
4. Exposes this functionality through a **simple web app** (Streamlit or Flask).

---

## 🧱 Main Features

- 🧹 **Data preparation & feature engineering**
- 📊 **EDA & visual insights**
- 🤖 **Classical ML models** (Logistic Regression, Random Forest, XGBoost, etc.)
- 🧠 **Deep learning model** (Keras MLP for tabular data)
- 🧩 **LLM explanation layer** – converts numeric outputs into human-friendly narratives
- 📚 **RAG pipeline** – query IPO documents using embeddings + LLM
- 🧵 **LangChain agent** – routes queries to prediction or RAG tools
- 🌐 **App UI** – simple interface to upload IPO details, get predictions & ask questions
- 📄 **Clear documentation** – architecture, model card, limitations

---

## 📂 Repository Structure

```text
ipo-listing-gains-genai/
│
├── data/
│   ├── raw/                # Original IPO datasets (as obtained)
│   └── processed/          # Cleaned and feature-engineered datasets
│
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory Data Analysis
│   ├── 02_ml_models.ipynb          # Classical ML experiments
│   ├── 03_deep_learning.ipynb      # Keras / deep learning experiments
│   ├── 04_genai_explanations.ipynb # LLM-based explanation experiments
│   └── 05_rag_langchain.ipynb      # RAG + LangChain experiments
│
├── src/
│   ├── data_preparation.py   # Cleaning & feature engineering pipeline
│   ├── train_ml.py           # Train and persist classical models
│   ├── train_dl.py           # Train and persist deep learning model
│   ├── predict.py            # Unified prediction API for ML/DL models
│   │
│   ├── genai/
│   │   ├── explain_prediction.py   # LLM-based explanation of model output
│   │   ├── create_embeddings.py    # Build embeddings for IPO documents
│   │   ├── rag_pipeline.py         # Retrieval-Augmented Generation pipeline
│   │   └── agent_langchain.py      # LangChain agent wiring tools together
│   │
│   └── app/
│       └── app_streamlit.py        # Streamlit (or Flask) app entry point
│
├── models/
│   ├── ml_model.pkl         # Best classical ML model
│   ├── dl_model.h5          # Best deep learning model
│   └── preprocessor.pkl     # Scalers / encoders / transformers
│
├── vectorstore/             # FAISS / Chroma DB files for IPO documents
│
├── docs/
│   ├── architecture.md      # High-level system design & data flow
│   └── model_card.md        # Model description, assumptions, and limitations
│
├── requirements.txt         # Python dependencies
└── README.md                # You are here 🙂
