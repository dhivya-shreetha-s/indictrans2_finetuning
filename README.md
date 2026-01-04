# 🏥 Fine-Tuning IndicTrans with LoRA for Healthcare Translation (Tamil ↔ English)

## 📌 Project Overview
This project focuses on **domain-specific machine translation for healthcare conversations**, addressing the limitations of general-purpose translators in medical contexts.  
A **LoRA-fine-tuned IndicTrans model** is developed to accurately translate **Tamil healthcare sentences into English**, preserving clinical intent and medical terminology.

The system is designed as a **clinical support tool** and includes **confidence-based prediction selection** to ensure safer outputs.

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
│   └── sentences.csv
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

### Dataset Statistics
| Split | Sentences |
|------|-----------|
| Training | 4,92,763 |
| Validation | 27,376 |
| Test | 27,376 |
| **Total** | **5,47,515** |

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

## 👩‍💻 Author
**Dhivya Shreetha S**  
Mail Id: **dhivyashreetha07@gmail.com** 
