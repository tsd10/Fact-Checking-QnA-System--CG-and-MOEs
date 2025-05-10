# 🧠 Fact-Checking Q&A System using Completion Graphs and Mixture-of-Experts

![System Screenshot](https://github.com/tsd10/Fact-Checking-QnA-System---CG-and-MOEs/blob/main/System%20Working%20Screenshot.png)

This project presents a factual Q&A system built with **domain-specific expert LLMs**, a **question router**, and a **triple-based fact checker**. It is designed to answer natural language questions with high factual accuracy by leveraging a mixture-of-experts architecture and structured knowledge validation.

---

## 🔍 Idea Behind the Project

LLMs are powerful but prone to hallucination. To reduce factual errors and improve answer trustworthiness, we designed a **multi-stage architecture**:

1. **Domain Classification Router** (trained on Qwen2.5-7B):
   - Determines whether the input question belongs to `history`, `music`, or `politics`
2. **Expert Answering Module** (LoRA-finetuned Qwen2.5-3B models):
   - Answers questions using a domain-specific LLM
3. **Fact Verification Module** (Qwen2.5-3B classifier):
   - Validates factuality of the answer
4. **Triple Extraction + Matching**:
   - Extracts (head, relation, tail) triples from the answer and compares them to a known KG subset (Wikidata/CoDEx domain splits)

---

## 🏗️ System Architecture

```
User Query
    │
    ▼
🔍 Router (Qwen2.5-7B + LoRA)
    │
    ├──> 🧠 History Expert (Qwen2.5-3B)
    ├──> 🎵 Music Expert (Qwen2.5-3B)
    └──> 🗳️ Politics Expert (Qwen2.5-3B)
        │
        ▼
✅ Fact-Checker (Qwen2.5-3B Sequence Classifier)
        │
        ▼
🔗 Triple Matching (Wikidata-like KG subset)
        │
        ▼
💬 Final Response with Fact Check + Supporting Triples
```

---

## 🛠️ How to Run

### 📦 Requirements

```bash
conda create -n NLPproj python=3.8
conda activate NLPproj
pip install -r requirements.txt
```

### 🧬 Download Models
- Download [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen1.5-7B) or host it locally at `/home/tushard/ondemand/NLP Final Project/qwen2.5-7B`
- Place LoRA adapters:
  - Router: `/qwen2.5-7b-router-final`
  - Experts: `/qwen2.5-3b-history_finetuned`, `/qwen-2.5-3b-final-music`, `/qwen-2.5-3b-final-politics`
  - Verifier: `/qwen2.5-3b-lora-fact-verifier-final`

### 🚀 Launch the App

```bash
cd streamlit_app
streamlit run app.py --server.port 8501 --server.enableCORS false
```

---

## 🧪 Features

- **Natural Language Question Answering**
- **Domain-aware Routing**
- **Modular Expert Models**
- **LoRA-optimized Inference (3B Qwen)**
- **Triple-based Verification and Factuality Scoring**

---

## 📁 Project Structure

```bash
streamlit_app/
├── app.py                 # Main UI logic
├── utils/
│   ├── router.py          # Domain classifier (Qwen2.5-7B + LoRA)
│   ├── experts.py         # Expert answering logic
│   ├── fact_checker.py    # Verifies factuality
│   └── triple_verifier.py # Extracts & matches triples from answers
```

---

## 📚 Dataset
- Domain-specific triples derived from CoDEx or Wikidata
- Located at:
  - `domain_splits/history.txt`
  - `domain_splits/music.txt`
  - `domain_splits/politics.txt`

---

## 👨‍💻 Authors

Virginia Tech, NLP 5624 Spring 2025 Group 

---

## 📌 Future Work
- Add support for more domains
- Integrate real-time Wikidata lookups
- Visualize supporting knowledge triples interactively

---

## 📸 System Preview
![System Screenshot](https://github.com/tsd10/Fact-Checking-QnA-System---CG-and-MOEs/blob/main/System%20Working%20Screenshot.png)

---

Feel free to ⭐ star or fork the repo to build your own knowledge-grounded QA system!
