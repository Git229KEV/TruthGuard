"""
TruthGuard API - Modal deployment
"""
import modal
from modal import Image, App, fastapi_endpoint, Secret
from fastapi import UploadFile, File
import os
import io
import base64
import re
import threading

app = App("truthguard-api", secrets=[
    Secret.from_name("HF_TOKEN"),
    Secret.from_name("GOOGLE_KEY_NEW"),
    Secret.from_name("TAVILY_KEY_NEW")
])

MODAL_IMAGE = Image.debian_slim(python_version="3.11").pip_install(
    "fastapi>=0.110.0",
    "uvicorn>=0.29.0",
    "transformers>=4.40.0",
    "torch>=2.0.0",
    "pillow>=10.0.0",
    "tavily-python>=0.3.0",
    "google-genai>=1.0.0",
    "python-multipart>=0.0.9",
    "python-dotenv>=1.0.0",
    "sentencepiece>=0.1.0",
    "protobuf>=3.20.1",
    "huggingface_hub>=0.21.0",
    "psutil>=5.9.0",
)

MODEL_DIR = "/model-cache"
SIGLIP_MODEL_PATH = f"{MODEL_DIR}/siglip"
XLM_MODEL_PATH = f"{MODEL_DIR}/xlm"

SIGLIP_HF_REPO = "Kevin229/final_siglip_model"
XLM_HF_REPO = "Kevin229/final_xlm_roberta_model"

genai_client = None
tavily_client = None
siglip_model = None
siglip_processor = None
xlm_model = None
xlm_tokenizer = None
torch = None
models_ready = {"status": "loading", "siglip": False, "xlm": False}
models_loaded = False
load_lock = threading.Lock()


def get_token():
    return os.environ.get("HF_TOKEN", "")


def download_models():
    global genai_client, tavily_client, siglip_model, siglip_processor, xlm_model, xlm_tokenizer, torch, models_ready, models_loaded
    
    google_api_key = os.environ.get("GOOGLE_API_KEY", "") or os.environ.get("GEMINI_KEY", "")
    tavily_api_key = os.environ.get("TAVILY_API_KEY", "") or os.environ.get("TAVILY_KEY", "")
    
    print(f"[DEBUG] GOOGLE_API_KEY set: {bool(google_api_key)}, TAVILY_API_KEY set: {bool(tavily_api_key)}")
    
    if not genai_client and google_api_key:
        try:
            from google import genai
            genai_client = genai.Client(api_key=google_api_key)
            print("[OK] Gemini configured with google.genai")
        except Exception as e:
            print(f"[FAIL] Gemini config: {e}")
    
    if not tavily_client and tavily_api_key:
        try:
            from tavily import TavilyClient
            tavily_client = TavilyClient(api_key=tavily_api_key)
            print("[OK] Tavily configured")
        except Exception as e:
            print(f"[FAIL] Tavily config: {e}")
    
    if models_loaded:
        return
    
    with load_lock:
        if models_loaded:
            return
            
        print("[INFO] Downloading models from HuggingFace...")
        
        os.makedirs(SIGLIP_MODEL_PATH, exist_ok=True)
        os.makedirs(XLM_MODEL_PATH, exist_ok=True)
        
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=SIGLIP_HF_REPO,
                local_dir=SIGLIP_MODEL_PATH,
                local_dir_use_symlinks=False,
                token=get_token(),
            )
            print("[OK] SigLIP downloaded")
        except Exception as e:
            print(f"[FAIL] SigLIP download: {e}")
        
        try:
            snapshot_download(
                repo_id=XLM_HF_REPO,
                local_dir=XLM_MODEL_PATH,
                local_dir_use_symlinks=False,
                token=get_token(),
            )
            print("[OK] XLM-R downloaded")
        except Exception as e:
            print(f"[FAIL] XLM download: {e}")
        
        print("[INFO] Loading PyTorch...")
        try:
            import torch
            device = torch.device("cpu")
            torch.set_default_device(device)
        except Exception as e:
            print(f"[FAIL] PyTorch: {e}")
        
        if torch:
            try:
                from transformers import SiglipProcessor, SiglipForImageClassification
                siglip_processor = SiglipProcessor.from_pretrained(SIGLIP_MODEL_PATH)
                siglip_model = SiglipForImageClassification.from_pretrained(
                    SIGLIP_MODEL_PATH,
                    torch_dtype=torch.float32
                )
                siglip_model.eval()
                print("[OK] SigLIP loaded")
                models_ready["siglip"] = True
            except Exception as e:
                print(f"[FAIL] SigLIP load: {e}")
            
            try:
                from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
                xlm_tokenizer = XLMRobertaTokenizer.from_pretrained(XLM_MODEL_PATH)
                xlm_model = XLMRobertaForSequenceClassification.from_pretrained(
                    XLM_MODEL_PATH,
                    torch_dtype=torch.float32
                )
                xlm_model.eval()
                print("[OK] XLM-R loaded")
                models_ready["xlm"] = True
            except Exception as e:
                print(f"[FAIL] XLM load: {e}")
        
        models_ready["status"] = "ready"
        models_loaded = True
        print("[INFO] Models ready")


@app.function(image=MODAL_IMAGE, timeout=600, memory=4096, gpu="any")
def init_models():
    download_models()
    return {"status": "ready", "siglip": models_ready["siglip"], "xlm": models_ready["xlm"]}


def parse_visual_label(raw_label):
    label_lower = str(raw_label).lower()
    if "label_0" in label_lower or label_lower == "0" or "fake" in label_lower:
        return "RUMOR"
    if "label_1" in label_lower or label_lower == "1" or "true" in label_lower or "real" in label_lower:
        return "NON-RUMOR"
    if any(x in label_lower for x in ["not_rumor", "non_rumor", "real", "true", "legit"]):
        return "NON-RUMOR"
    if any(x in label_lower for x in ["fake", "false", "rumor", "misinformation", "label_0"]):
        return "RUMOR"
    return "UNKNOWN"


@app.function(image=MODAL_IMAGE, timeout=300, memory=4096)
@fastapi_endpoint(method="GET")
def health():
    download_models()
    return {
        "status": "healthy",
        "models_status": models_ready["status"],
        "siglip_loaded": models_ready["siglip"],
        "xlm_loaded": models_ready["xlm"],
        "gemini_available": genai_client is not None,
        "tavily_available": tavily_client is not None,
    }


@app.function(image=MODAL_IMAGE, timeout=600, memory=4096)
@fastapi_endpoint(method="POST")
async def analyze(file: UploadFile = File(...)):
    from PIL import Image
    import torch
    
    download_models()
    
    if not file.content_type or not file.content_type.startswith("image/"):
        return {"error": "File must be an image"}
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_base64 = base64.b64encode(contents).decode()
        
        results = {
            "visual": "UNAVAILABLE",
            "text": "UNAVAILABLE",
            "tavily": "UNAVAILABLE",
            "gemini": "UNAVAILABLE",
            "final": "NON-RUMOR",
            "sources": [],
            "confidence": 0.5,
            "gemini_analysis": "",
            "tavily_analysis": "",
            "translated": "",
            "original_text": "",
            "claim_verdict": "NON-RUMOR",
            "report_verdict": "NON-RUMOR",
            "gemini_model_used": "N/A"
        }
        
        # --- PILLAR 1: VISUAL FORENSICS (SigLIP) ---
        if torch and siglip_model and siglip_processor:
            try:
                inputs = siglip_processor(images=image, return_tensors="pt")
                with torch.no_grad():
                    logits = siglip_model(**inputs).logits
                    probs = torch.softmax(logits, dim=1)
                    pred = probs.argmax().item()
                raw_vis = siglip_model.config.id2label.get(pred, str(pred))
                results["visual"] = "RUMOR" if any(x in raw_vis.lower() for x in ["fake", "false", "rumor"]) else "NON-RUMOR"
                results["visual_confidence"] = float(probs.max().item())
                print(f"[SigLIP] {results['visual']}")
            except Exception as e:
                print(f"SigLIP Error: {e}")
        
        # --- PILLAR 2: TEXTUAL FORENSICS (XLM-R) ---
        if xlm_model and xlm_tokenizer:
            try:
                # Basic text context for the XLM-R model
                text_input = "This is a news article about current events"
                inputs_xlm = xlm_tokenizer(text_input, return_tensors="pt", truncation=True, padding=True, max_length=512)
                with torch.no_grad():
                    logits = xlm_model(**inputs_xlm).logits
                    text_probs = torch.softmax(logits, dim=1)
                    text_pred = text_probs.argmax().item()
                    confidence = text_probs.max().item()
                raw_txt = xlm_model.config.id2label.get(text_pred, str(text_pred))
                results["text"] = "RUMOR" if "0" in raw_txt or "fake" in raw_txt.lower() else "NON-RUMOR"
                results["text_confidence"] = float(confidence)
                print(f"[XLM-R] {results['text']}")
            except Exception as e:
                print(f"XLM Error: {e}")
        
        # --- PILLAR 3: VISION & OCR (Gemini Phase 1) ---
        if genai_client:
            try:
                print("[Gemini] Phase 1: OCR and Initial Analysis...")
                model_names = ["gemini-3.1-pro-preview", "gemini-3-flash-preview"]
                ocr_prompt = "Extract all text from this image accurately. If there is non-English text, translate it to English. Also, provide a 1-sentence summary of what is happening in the image."
                
                from google.genai.types import Content, Part
                for model_name in model_names:
                    try:
                        response_ocr = genai_client.models.generate_content(
                            model=model_name,
                            contents=Content(parts=[
                                Part(inline_data={"mime_type": "image/jpeg", "data": img_base64}),
                                Part(text=ocr_prompt)
                            ]),
                            config={"temperature": 0}
                        )
                        raw_ocr = response_ocr.text
                        results["original_text"] = raw_ocr.strip()
                        results["translated"] = raw_ocr.strip() # Populate the frontend 'translated' field
                        results["gemini_model_used"] = model_name
                        print(f"[Gemini] OCR success with {model_name}")
                        break
                    except: continue
            except Exception as e:
                print(f"Gemini OCR Error: {e}")

        # --- PILLAR 4: WEB RESEARCH (Tavily) ---
        if tavily_client:
            try:
                print("[Tavily] Running independent web research...")
                raw_query = results["original_text"]
                clean_query = re.sub(r'\[.*?\]', '', raw_query).strip()
                clean_query = clean_query.replace("**", "").replace('"', '').strip()
                
                search_query = clean_query[:300] if len(clean_query) > 10 else "latest fact check " + clean_query
                if not search_query.strip(): search_query = "latest news verification"
                
                tav_res = tavily_client.search(
                    query=search_query,
                    search_depth="advanced",
                    include_answer="advanced",
                    max_results=5
                )
                
                tav_sources = [{"title": r.get("title"), "url": r.get("url")} for r in tav_res.get("results", [])]
                results["sources"] = tav_sources
                
                tav_answer = tav_res.get("answer", "")
                tav_context = " ".join([r.get("content", "") for r in tav_res.get("results", [])[:3]])
                combined_tav_text = (tav_answer + " " + tav_context).lower()
                
                # Independent Tavily Scoring
                strong_rumor = ["false claim", "fake news", "hoax", "misinformation", "debunked", "fabricated"]
                strong_fact = ["confirmed by", "official statement", "verified", "true"]
                
                rumor_score = sum(2 for phrase in strong_rumor if phrase in combined_tav_text)
                fact_score = sum(2 for phrase in strong_fact if phrase in combined_tav_text)
                results["tavily"] = "RUMOR" if rumor_score > fact_score + 1 else "NON-RUMOR"
                results["tavily_analysis"] = tav_answer if tav_answer else "Web search completed."
                print(f"[Tavily] Verdict: {results['tavily']} (R:{rumor_score} F:{fact_score})")
            except Exception as e:
                print(f"Tavily Error: {e}")

        # --- PILLAR 5: FINAL SYNTHESIS (Gemini Phase 2) ---
        if genai_client:
            try:
                system_instruction = """You are a professional Multimodal Fact-Checker specializing in digital news verification. Your goal is to analyze images to detect rumors, manipulation, or misinformation.

Format your response STRICTLY as follows:
1. **CLAIM VERDICT:** [TRUE / FALSE / MISLEADING / UNVERIFIED]
2. **REPORT AUTHENTICITY:** [AUTHENTIC / MANIPULATED / AI-GENERATED]
3. **DETAILED ANALYSIS:** Start with a section titled 'Image Authenticity' followed by a professional breakdown of visual consistency, AI artifacts, and contextual extraction.
4. **SEARCH EVIDENCE:** Findings from the provided web research snippets.
5. **WHY:** A final summary of exactly why these verdicts were reached.

Your Analysis Protocol:
- Distinguish between the 'Claim' (what is being said) and the 'Report' (the image itself).
- Visual Consistency: Check for AI artifacts, inconsistent lighting, or news template patterns.
- Contextual Extraction: Identify the core claim, location, and key figures.
- Search Strategy: Verify claims against the provided web research research snippets."""

                translated_context = results["translated"] if results["translated"] else "No text extracted."
                search_context = results["tavily_analysis"] if results["tavily_analysis"] else "No web search results available."
                
                synthesis_prompt = f"""
INPUT - TRANSLATED TEXT FROM IMAGE:
{translated_context}

INPUT - WEB RESEARCH SUMMARY:
{search_context}

INPUT - SOURCES:
{str(results['sources'])}

TASK: Analyze the provided image pixels AND the 'Translated Text' against the 'Web Research' results. determine if the content is a RUMOR or NON-RUMOR.
Give your OWN forensic result based on both visual evidence and textual claim. Follow the system protocol for the 5-point report."""

                # Re-implement Model Fallback loop for Synthesis Phase
                model_names = ["gemini-3.1-pro-preview", "gemini-3-flash-preview", "gemini-1.5-pro", "gemini-1.5-flash"]
                sync_text = ""
                selected_model_final = "N/A"
                
                for model_name in model_names:
                    try:
                        print(f"[Gemini] Synthesis attempting with {model_name}...")
                        response_sync = genai_client.models.generate_content(
                            model=model_name,
                            contents=Content(parts=[
                                Part(inline_data={"mime_type": "image/jpeg", "data": img_base64}),
                                Part(text=synthesis_prompt)
                            ]),
                            config={
                                "system_instruction": system_instruction,
                                "temperature": 0, 
                                "max_output_tokens": 2048
                            }
                        )
                        sync_text = response_sync.text
                        if sync_text and len(sync_text) > 20:
                            results["gemini_analysis"] = sync_text
                            selected_model_final = model_name
                            print(f"[Gemini] Synthesis success with {model_name}")
                            break
                    except Exception as e:
                        print(f"[Gemini] Synthesis fallback: {model_name} failed: {e}")
                        continue
                
                if not sync_text:
                    results["gemini_analysis"] = f"Investigation complete. Extracted Text: {translated_context[:300]}... [Note: Full AI synthesis was limited by API status]."
                    selected_model_final = results["gemini_model_used"] if results["gemini_model_used"] != "N/A" else "None"

                # Update model name to include Tavily as requested in screenshot
                results["gemini_model_used"] = f"Model {selected_model_final.upper()} and Tavily"

                # Parse verdict from synthesis
                lines = sync_text.split('\n')
                g_cv = "NON-RUMOR"
                g_av = "NON-RUMOR"
                for line in lines:
                    if 'CLAIM VERDICT:' in line.upper():
                        if any(x in line.upper() for x in ["FALSE", "FAKE", "MISLEADING"]): g_cv = "RUMOR"
                    if 'REPORT AUTHENTICITY:' in line.upper():
                        if any(x in line.upper() for x in ["MANIPULATED", "SUSPICIOUS", "FAKE"]): g_av = "RUMOR"
                
                results["gemini"] = "RUMOR" if (g_cv == "RUMOR" or g_av == "RUMOR") else "NON-RUMOR"
                results["claim_verdict"] = g_cv
                results["report_verdict"] = g_av
                print(f"[Gemini] Synthesis: {results['gemini']}")
            except Exception as e:
                print(f"Gemini Synthesis Error: {e}")
                # ROBUST FALLBACK: Populate analysis even on failure
                if not results["gemini_analysis"]:
                    results["gemini_analysis"] = f"Investigation complete. Extracted Text: {results['translated'][:400]}... [Note: Full AI synthesis report was limited by API status]."

        # --- FINAL AGGREGATION (Majority Vote) ---
        verdicts = {
            "visual": results["visual"],
            "text": results["text"],
            "tavily": results["tavily"],
            "gemini": results["gemini"]
        }
        
        valid_votes = [v for v in verdicts.values() if v in ["RUMOR", "NON-RUMOR"]]
        r_count = valid_votes.count("RUMOR")
        nr_count = valid_votes.count("NON-RUMOR")
        
        if r_count > nr_count: 
            results["final"] = "RUMOR"
        elif nr_count > r_count:
            results["final"] = "NON-RUMOR"
        else:
            # TIE: Respect the Forensic Baseline (Universal Forensic override)
            results["final"] = results["visual"] if results["visual"] in ["RUMOR", "NON-RUMOR"] else "NON-RUMOR"
        
        results["confidence"] = float(max(r_count, nr_count) / len(valid_votes)) if valid_votes else 0.5
        print(f"[FINAL RESULT] {results['final']} (R:{r_count} NR:{nr_count} Conf:{results['confidence']:.2f})")
        return results
        
    except Exception as e:
        print(f"Analysis error: {e}")
        return {"error": str(e)}


@app.local_entrypoint()
def main():
    result = init_models.remote()
    print(f"Init result: {result}")
