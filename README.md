# 🏥 Fine-Tuning IndicTrans with LoRA for Healthcare Translation (Tamil ↔ English)

## 📌 Project Overview
This project focuses on **domain-specific machine translation for healthcare conversations**, addressing the limitations of general-purpose translators in medical contexts.  
A **LoRA-fine-tuned IndicTrans model** is developed to accurately translate **Tamil healthcare sentences into English**, preserving clinical intent and medical terminology.

The system is designed as a **clinical support tool** and includes **confidence-based prediction selection** to ensure safer outputs.

---

## 🎯 Problem Statement
- India is a multilingual country where patients describe symptoms in regional languages.
- Doctors and medical records predominantly use English.
- General NMT systems often mistranslate medical terminology and long clinical sentences.

**Goal:**  
Build a **healthcare-aware translation system** that improves accuracy, reliability, and safety for Tamil–English medical translations.

---

## 🚀 Key Features
- LoRA fine-tuning on IndicTrans base model  
- Healthcare-specific synthetic dataset  
- Medical dictionary integration for terminology preservation  
- Confidence-score–based output selection (LoRA vs Base model)  
- Interactive UI deployed on Hugging Face Spaces  

---

## 🧠 Model Architecture
- **Base Model:** ai4bharat/indictrans2-indic-en-1B  
- **Fine-Tuning Method:** LoRA (PEFT)  
- **Tokenizer:** IndicTrans Tokenizer  

> LoRA is **not optional** — both base and LoRA models generate predictions, and the final output is selected using a confidence score.

---

## 📂 Repository Structure
```
├── Code/
│   ├── huggingface prediction code/
│   │   └── app.py
│   └── indictrans-lora-finetuning.ipynb
│
├── Dataset/
│   ├── medical term + sentences.csv
│   ├── sentences.csv
│   └── sentences_2.csv
│
├── Result/
│   ├── Training_loss.png
│   ├── Error_distribution.png
│   ├── Medical_confusion_matrix.png
│   ├── Precision_recall_specificity.png
│   ├── UI.png
│   └── Sample prediction/
│       ├── Sample_prediction_1.png
│       ├── Sample_prediction_2.png
│       └── Sample_prediction_3.png
│
└── README.md
```

---

## 📊 Dataset Description
- Fully **synthetic and anonymized**
- No real patient data used
- Simulates real hospital conversations

### Covered Scenarios
- Patient symptom descriptions  
- Diagnosis explanations  
- Medication and dosage instructions  
- Diagnostic test discussions  
- Follow-up and treatment advice  

### Dataset Statistics
| Split | Sentences |
|------|-----------|
| Training | 432,756 |
| Validation | 24,042 |
| Test | 24,042 |
| **Total** | **480,840** |

---

## ⚙️ Training Details
- Framework: Hugging Face Transformers  
- Fine-Tuning: LoRA via PEFT  
- Hardware: GPU (CUDA)  
- Loss Function: Cross-Entropy  

---

## 📈 Results & Evaluation
- Improved translation of medical terminology  
- Reduced hallucination  
- Better preservation of clinical meaning  

Evaluation visualizations are available in the `Result/` folder.

---

## 🖥️ Deployment
- Hosted on **Hugging Face Spaces**
- Built using **Gradio**
- Tamil input → English output
- Medical dictionary loaded securely via environment secrets

---

## 🔐 Ethical & Safety Considerations
- No real patient data used  
- Synthetic and anonymized dataset  
- Not a diagnostic tool  
- Human verification required  
- Confidence score supports safe interpretation  

---

## ⚠️ Limitations
- Rare medical terms may still fail  
- Limited to Tamil ↔ English  
- No speech input/output  
- Confidence score is probabilistic  

---

## 🧩 Challenges Faced
- Scarcity of Indian-language healthcare datasets  
- Complex medical terminology  
- Balancing fluency and accuracy  
- Computational constraints  

---

## 🔮 Future Work
- Support for more Indian languages  
- Speech-to-text integration  
- Larger-scale clinical validation  

---

## 👩‍💻 Author
**Dhivya Shreetha S**  
National AI Olympiad – Stage 2 Capstone Project  
Domain: Healthcare | NLP | Generative AI
