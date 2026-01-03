import gradio as gr
import torch
import numpy as np
import traceback
import os
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from IndicTransToolkit import IndicProcessor

# ---------------- CONFIG ----------------
MODEL_ID_INDIC_EN = "ai4bharat/indictrans2-indic-en-1B"
ADAPTER_PATH = "./final_model"
MODEL_ID_EN_INDIC = "ai4bharat/indictrans2-en-indic-1B"

# 🔒 LOAD API KEY FROM SECRETS
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- LOAD RESOURCES ----------------
print(f"Device: {DEVICE}")
ip = IndicProcessor(inference=True)

# --- 1. Load Tamil -> English Model & LoRA ---
print("Loading Model A: Tamil -> English...")
tokenizer_ie = AutoTokenizer.from_pretrained(MODEL_ID_INDIC_EN, trust_remote_code=True)
model_ie = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_ID_INDIC_EN, 
    trust_remote_code=True, 
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
)

LORA_ACTIVE = False
try:
    if os.path.exists(ADAPTER_PATH):
        model_ie = PeftModel.from_pretrained(model_ie, ADAPTER_PATH)
        model_ie.to(DEVICE)
        model_ie.eval()
        LORA_ACTIVE = True
        print("✅ LoRA Adapter attached to Indic-En model!")
    else:
        print(f"⚠️ LoRA path '{ADAPTER_PATH}' not found. Using Base Indic-En.")
        model_ie.to(DEVICE)
except Exception as e:
    print(f"⚠️ LoRA Load Error: {e}")
    model_ie.to(DEVICE)

# --- 2. Load English -> Tamil Model ---
print("Loading Model B: English -> Tamil...")
try:
    tokenizer_ei = AutoTokenizer.from_pretrained(MODEL_ID_EN_INDIC, trust_remote_code=True)
    model_ei = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_ID_EN_INDIC, 
        trust_remote_code=True, 
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    ).to(DEVICE)
    model_ei.eval()
except Exception as e:
    print(f"CRITICAL ERROR loading En-Indic model: {e}")
    tokenizer_ei = tokenizer_ie
    model_ei = model_ie 

# Configure AI Refiner
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-pro')
else:
    print("⚠️ WARNING: GEMINI_API_KEY not found in secrets. AI Refinement will be disabled.")

# ---------------- HELPER FUNCTIONS ----------------

def run_inference(model, tokenizer, text, src_lang, tgt_lang, use_lora_if_available=False):
    """Generic function to run either model."""
    try:
        batch = ip.preprocess_batch([text], src_lang=src_lang, tgt_lang=tgt_lang)
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to(DEVICE)

        class DummyContext:
            def __enter__(self): pass
            def __exit__(self, exc_type, exc_val, exc_tb): pass

        context = DummyContext()
        if isinstance(model, PeftModel) and not use_lora_if_available:
             if hasattr(model, "disable_adapter"):
                context = model.disable_adapter()
        
        with context:
            with torch.no_grad():
                forced_bos_id = tokenizer.convert_tokens_to_ids(f"<2{tgt_lang}>")
                outputs = model.generate(
                    **inputs, max_new_tokens=256, num_beams=5, use_cache=False,
                    forced_bos_token_id=forced_bos_id, output_scores=True, return_dict_in_generate=True 
                )

        decoded = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        final_text = ip.postprocess_batch(decoded, lang=tgt_lang)[0]

        try:
            transition_scores = model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )
            confidence = np.exp(transition_scores.cpu().numpy()).mean()
        except:
            confidence = 0.5
        
        return final_text, float(confidence)

    except Exception as e:
        print(f"Inference Error: {e}")
        return "", 0.0

# ---------------- 🔥 SMART AI-REFINER ----------------
def smart_medical_refinement(original_input, draft_translation, src_name, tgt_name):
    """Refines translation using AI as an editor."""
    try:
        if not GEMINI_API_KEY: 
            return draft_translation
            
        prompt = f"""
        Act as a strict Medical Translator Editor.
        Original {src_name}: "{original_input}"
        Draft {tgt_name}: "{draft_translation}"
        
        Task: 
        1. Review the draft for medical terminology accuracy.
        2. Fix ONLY incorrect medical terms.
        3. Do NOT rewrite the whole sentence unless meaning is wrong.
        4. Return ONLY the corrected {tgt_name} text.
        """
        
        response = gemini_model.generate_content(prompt)
        if response.text: 
            return response.text.strip()
        return draft_translation
        
    except Exception as e:
        print(f"AI Refinement Error: {e}")
        return draft_translation

# ---------------- MAIN LOGIC ----------------
def translate(text, direction, enable_ai, history):
    if not text.strip():
        # Return empty output but keep history same
        return gr.Textbox(value="", label="Best Translation"), history, history

    # 1. SETUP
    if direction == "Tamil → English":
        s_lang, t_lang = "tam_Taml", "eng_Latn"
        s_name, t_name = "Tamil", "English"
        active_model = model_ie
        active_tokenizer = tokenizer_ie
    else: 
        s_lang, t_lang = "eng_Latn", "tam_Taml"
        s_name, t_name = "English", "Tamil"
        active_model = model_ei
        active_tokenizer = tokenizer_ei

    # 2. RUN LOCAL
    use_lora = LORA_ACTIVE and (direction == "Tamil → English")
    source_label = "LoRA Fine-Tuned" if use_lora else "IndicTrans2 Base"
    
    draft_text, score = run_inference(
        active_model, active_tokenizer, text, s_lang, t_lang, use_lora_if_available=use_lora
    )
    
    final_output = draft_text
    final_label_extra = ""

    # 3. REFINE
    if enable_ai and draft_text:
        refined_text = smart_medical_refinement(text, draft_text, s_name, t_name)
        if refined_text != draft_text:
            final_output = refined_text
            final_label_extra = " + AI Refined"
        
    new_label = f"Best Translation (Source: {source_label}{final_label_extra})"
    
    # 4. UPDATE HISTORY
    # Insert new result at the top of the list: [Direction, Input, Output]
    history.insert(0, [direction, text, final_output])
    
    # Return: Output Box Update, History State Update, History Table Display Update
    return gr.Textbox(value=final_output, label=new_label), history, history

# ---------------- UI CONFIG ----------------
status_text = "System Ready (Both Models Loaded)" if LORA_ACTIVE else "LoRA Missing (Using Base Models)"

# Define Examples
examples = [
    ["நோயாளிக்கு கடுமையான நெஞ்சு வலி மற்றும் மூச்சுத் திணறல் உள்ளது.", "Tamil → English", True],
    ["The patient was diagnosed with acute renal failure and needs dialysis.", "English → Tamil", True],
    ["சிறுநீரக கற்கள் உள்ளதால் அவருக்கு அறுவை சிகிச்சை தேவைப்படுகிறது.", "Tamil → English", True],
    ["Take one tablet of Paracetamol 500mg after food for fever.", "English → Tamil", True]
]

# Custom CSS for polish
custom_css = """
.gradio-container {background-color: #f9fafb}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="Bi-Directional Medical Translation") as demo:
    
    # --- HEADER ---
    gr.Markdown("## 🏥 Bi-Directional Medical Translation")

    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("Accurate medical translation using **Fine-Tuned IndicTrans2** + **Smart AI Refinement**.")
        
        with gr.Column(scale=1):
            status_box = gr.Textbox(
                value=status_text,
                show_label=False, 
                interactive=False, 
                container=True,
                scale=1,
                text_align="center"
            )

    # --- MAIN INTERFACE ---
    with gr.Row():
        
        # LEFT COLUMN
        with gr.Column(scale=1):
            with gr.Group():
                direction = gr.Dropdown(
                    choices=["Tamil → English", "English → Tamil"], 
                    value="Tamil → English", 
                    label="Translation Direction",
                    filterable=False,
                    interactive=True
                )

            with gr.Group():
                input_text = gr.Textbox(
                    label="Input Text", 
                    lines=5, 
                    placeholder="Type or paste medical text here..."
                )
            
            enable_ai = gr.Checkbox(label="✨ Enable Smart Refinement", value=True)
            submit_btn = gr.Button("Translate Text", variant="primary")

        # RIGHT COLUMN
        with gr.Column(scale=1):
            output_text = gr.Textbox(
                label="Best Translation", 
                lines=9,
                show_copy_button=True
            )

    # --- HISTORY SECTION ---
    with gr.Row():
        history_display = gr.Dataframe(
            headers=["Direction", "Input Text", "Translated Output"],
            datatype=["str", "str", "str"],
            label="📜 Translation History",
            interactive=False,
            row_count=5,
            wrap=True
        )

    # --- EXAMPLES SECTION ---
    gr.Examples(
        examples=examples,
        inputs=[input_text, direction, enable_ai],
        label="Quick Test Examples"
    )

    # --- FOOTER / DETAILS ---
    with gr.Accordion("ℹ️ System Details", open=False):
        gr.Markdown(f"""
        - **Tamil → English Model:** `ai4bharat/indictrans2-indic-en-1B` + Custom LoRA
        - **English → Tamil Model:** `ai4bharat/indictrans2-en-indic-1B`
        - **Refinement:** Advanced AI Model
        - **Device:** {DEVICE}
        """)

    # --- STATE MANAGEMENT ---
    history_state = gr.State([]) # Stores the list of previous translations

    # --- LOGIC ---
    submit_btn.click(
        fn=translate,
        # Pass history_state in inputs
        inputs=[input_text, direction, enable_ai, history_state],
        # Update output text, history state, and history table display
        outputs=[output_text, history_state, history_display]
    )

demo.launch()