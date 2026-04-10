from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import base64
import json
import os
import time
import threading
import re
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("[WARN] pytesseract not available, using fallback OCR")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(BASE_DIR, ".env")
load_dotenv(env_path, override=True)

print(f"[INFO] Loading .env from: {env_path}")
print(f"[INFO] GOOGLE_API_KEY: {'set' if os.getenv('GOOGLE_API_KEY') else 'NOT SET'}")
print(f"[INFO] TAVILY_API_KEY: {'set' if os.getenv('TAVILY_API_KEY') else 'NOT SET'}")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

default_siglip_path = os.path.join(BASE_DIR, "models", "final_siglip_model")
default_xlm_path = os.path.join(BASE_DIR, "models", "final_xlm_roberta_model")

SIGLIP_PATH = os.getenv("SIGLIP_MODEL_PATH")
if SIGLIP_PATH:
    SIGLIP_PATH = SIGLIP_PATH.strip('"').strip("'")
if not SIGLIP_PATH:
    SIGLIP_PATH = default_siglip_path

XLM_PATH = os.getenv("XLM_ROBERTA_MODEL_PATH")
if XLM_PATH:
    XLM_PATH = XLM_PATH.strip('"').strip("'")
if not XLM_PATH:
    XLM_PATH = default_xlm_path

print(f"[INFO] SIGLIP_PATH: {SIGLIP_PATH}")
print(f"[INFO] XLM_PATH: {XLM_PATH}")

def list_gdrive_folder_contents(folder_id):
    try:
        import gdown
        url = f"https://drive.google.com/drive/folders/{folder_id}"
        files = gdown.list_folder(url)
        return files
    except Exception as e:
        print(f"[WARN] Cannot list folder: {e}")
        return []

def download_from_huggingface(repo_id, dest_path):
    try:
        from huggingface_hub import snapshot_download
        os.makedirs(dest_path, exist_ok=True)
        print(f"[INFO] Downloading from HuggingFace: {repo_id}")
        
        kwargs = {"local_dir": dest_path, "local_dir_use_symlinks": False}
        if HF_TOKEN:
            kwargs["token"] = HF_TOKEN
        
        snapshot_download(repo_id=repo_id, **kwargs)
        
        print(f"[INFO] Contents after download:")
        for root, dirs, files in os.walk(dest_path):
            for f in files:
                full_path = os.path.join(root, f)
                try:
                    size = os.path.getsize(full_path) / (1024*1024)
                    print(f"  {os.path.relpath(full_path, dest_path)} ({size:.1f} MB)")
                except:
                    pass
        
        model_files = []
        for root, dirs, files in os.walk(dest_path):
            for f in files:
                if f in ['model.safetensors', 'pytorch_model.bin', 'config.json', 'tokenizer.json']:
                    model_files.append(os.path.join(root, f))
        
        if model_files:
            print(f"[INFO] Found model files: {model_files}")
        
        return len(model_files) > 0
    except Exception as e:
        print(f"[WARN] HuggingFace download failed: {e}")
        return False

SIGLIP_HF_REPO = os.getenv("SIGLIP_HF_REPO", "Kevin229/final_siglip_model")
XLM_HF_REPO = os.getenv("XLM_HF_REPO", "Kevin229/final_xlm_roberta_model")
HF_TOKEN = os.getenv("HF_TOKEN")

print(f"[INFO] SIGLIP_HF_REPO: {SIGLIP_HF_REPO}")
print(f"[INFO] XLM_HF_REPO: {XLM_HF_REPO}")

def check_model_valid(model_path, min_size_mb=50):
    for root, dirs, files in os.walk(model_path):
        for f in files:
            if f in ['model.safetensors', 'pytorch_model.bin']:
                size_mb = os.path.getsize(os.path.join(root, f)) / (1024*1024)
                print(f"[INFO] Found {f} at {root}: {size_mb:.1f} MB")
                if size_mb < min_size_mb:
                    print(f"[WARN] {f} is too small ({size_mb:.1f} MB), likely corrupted")
                    return False
    return True

siglip_exists = os.path.exists(SIGLIP_PATH) and os.listdir(SIGLIP_PATH)
xlm_exists = os.path.exists(XLM_PATH) and os.listdir(XLM_PATH)

print(f"[INFO] SigLIP dir exists: {os.path.exists(SIGLIP_PATH)}, has files: {siglip_exists}")
print(f"[INFO] XLM dir exists: {os.path.exists(XLM_PATH)}, has files: {xlm_exists}")

siglip_valid = check_model_valid(SIGLIP_PATH) if siglip_exists else False
xlm_valid = check_model_valid(XLM_PATH) if xlm_exists else False

if not siglip_exists or not siglip_valid:
    print(f"[INFO] SigLIP not found or invalid, downloading from HuggingFace...")
    import shutil
    if os.path.exists(SIGLIP_PATH):
        shutil.rmtree(SIGLIP_PATH)
    download_from_huggingface(SIGLIP_HF_REPO, SIGLIP_PATH)
    siglip_exists = os.path.exists(SIGLIP_PATH) and os.listdir(SIGLIP_PATH)
    siglip_valid = check_model_valid(SIGLIP_PATH)
    print(f"[INFO] SigLIP download result - exists: {siglip_exists}, valid: {siglip_valid}")

if not xlm_exists or not xlm_valid:
    print(f"[INFO] XLM not found or invalid, downloading from HuggingFace...")
    import shutil
    if os.path.exists(XLM_PATH):
        shutil.rmtree(XLM_PATH)
    download_from_huggingface(XLM_HF_REPO, XLM_PATH)
    xlm_exists = os.path.exists(XLM_PATH) and os.listdir(XLM_PATH)
    xlm_valid = check_model_valid(XLM_PATH)
    print(f"[INFO] XLM download result - exists: {xlm_exists}, valid: {xlm_valid}")

genai_client = None
tavily_client = None
tavily_cache = {}

if GOOGLE_API_KEY:
    try:
        import google.generativeai as genai_lib
        genai_lib.configure(api_key=GOOGLE_API_KEY)
        genai_client = genai_lib
        print("[OK] Gemini configured")
    except Exception as e:
        print(f"[FAIL] Gemini config error: {e}")

if TAVILY_API_KEY:
    try:
        from tavily import TavilyClient
        tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        print("[OK] Tavily configured")
    except Exception as e:
        print(f"[FAIL] Tavily config error: {e}")

device = "cpu"
torch = None
try:
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[OK] PyTorch loaded, device: {device}")
except Exception as e:
    print(f"[FAIL] PyTorch error: {e}")

siglip_model = None
siglip_processor = None
xlm_tokenizer = None
xlm_model = None

models_ready = {
    "status": "loading",
    "siglip": False,
    "xlm": False
}

def get_available_memory_mb():
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 * 1024)
    except:
        return 999999

def find_model_dir(base_path):
    if os.path.exists(base_path):
        for root, dirs, files in os.walk(base_path):
            print(f"[DEBUG] Checking dir: {root}, files: {files}")
            if 'config.json' in files and any(f in files for f in ['pytorch_model.bin', 'model.safetensors', 'model.bin']):
                print(f"[INFO] Found model files in: {root}")
                return root
    return base_path

def load_models_background():
    global siglip_model, siglip_processor, xlm_tokenizer, xlm_model, models_ready
    print("[INFO] Background thread starting model loading...")
    
    available_mem = get_available_memory_mb()
    print(f"[INFO] Available memory: {available_mem:.0f} MB")
    
    if not torch:
        models_ready["status"] = "partial"
        return
    
    if available_mem < 1500:
        print("[WARN] Low memory detected. Skipping local models (cloud-only mode).")
        models_ready["status"] = "cloud-only"
        return
        
    try:
        from transformers import (
            SiglipProcessor, SiglipForImageClassification,
            XLMRobertaTokenizer, XLMRobertaForSequenceClassification
        )
        
        if siglip_model is None:
            try:
                print(f"[INFO] SigLIP base path: {SIGLIP_PATH}")
                print(f"[INFO] SigLIP path exists: {os.path.exists(SIGLIP_PATH)}")
                if os.path.exists(SIGLIP_PATH):
                    print(f"[INFO] SigLIP contents: {os.listdir(SIGLIP_PATH)}")
                siglip_actual_path = find_model_dir(SIGLIP_PATH)
                print(f"[INFO] SigLIP actual path: {siglip_actual_path}")
                if siglip_actual_path and os.path.exists(siglip_actual_path):
                    print(f"[INFO] Loading SIGLIP from: {siglip_actual_path}")
                    siglip_model = SiglipForImageClassification.from_pretrained(
                        siglip_actual_path, 
                        low_cpu_mem_usage=True,
                        torch_dtype=torch.float32
                    )
                    siglip_processor = SiglipProcessor.from_pretrained(siglip_actual_path)
                else:
                    print(f"[WARN] SIGLIP model files not found in: {SIGLIP_PATH}")
                    raise FileNotFoundError(f"SigLIP model not found at {SIGLIP_PATH}")
                siglip_model = siglip_model.to(device)
                siglip_model.eval()
                print("[OK] SIGLIP model loaded")
                
                dummy_img = Image.new('RGB', (384, 384))
                with torch.no_grad():
                    inputs = siglip_processor(images=dummy_img, return_tensors="pt").to(device)
                    _ = siglip_model(**inputs)
                print("[OK] SIGLIP model warmed up")
            except Exception as e:
                print(f"[FAIL] SIGLIP error: {e}")
                siglip_model = None
        
        if xlm_model is None:
            try:
                print(f"[INFO] XLM base path: {XLM_PATH}")
                print(f"[INFO] XLM path exists: {os.path.exists(XLM_PATH)}")
                if os.path.exists(XLM_PATH):
                    print(f"[INFO] XLM contents: {os.listdir(XLM_PATH)}")
                xlm_actual_path = find_model_dir(XLM_PATH)
                print(f"[INFO] XLM actual path: {xlm_actual_path}")
                if xlm_actual_path and os.path.exists(xlm_actual_path):
                    print(f"[INFO] Loading XLM-RoBERTa from: {xlm_actual_path}")
                    xlm_tokenizer = XLMRobertaTokenizer.from_pretrained(xlm_actual_path)
                    xlm_model = XLMRobertaForSequenceClassification.from_pretrained(
                        xlm_actual_path, 
                        low_cpu_mem_usage=True,
                        torch_dtype=torch.float32
                    )
                else:
                    print(f"[WARN] XLM-RoBERTa model files not found in: {XLM_PATH}")
                    raise FileNotFoundError(f"XLM-RoBERTa model not found at {XLM_PATH}")
                xlm_model = xlm_model.to(device)
                xlm_model.eval()
                print("[OK] XLM-RoBERTa model loaded")
                
                with torch.no_grad():
                    inputs = xlm_tokenizer("warmup", return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
                    _ = xlm_model(**inputs)
                print("[OK] XLM-RoBERTa model warmed up")
            except Exception as e:
                print(f"[FAIL] XLM-R error: {e}")
                xlm_model = None
            
    except Exception as e:
        print(f"[FAIL] Model loading error: {e}")
        
    models_ready["siglip"] = siglip_model is not None
    models_ready["xlm"] = xlm_model is not None
    models_ready["status"] = "ready"
    print(f"[INFO] Models ready: {models_ready}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    thread = threading.Thread(target=load_models_background, daemon=True)
    thread.start()
    yield

app = FastAPI(title="TruthGuard API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "TruthGuard Rumor Detection API",
        "status": "online",
        "models_status": models_ready["status"]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_status": models_ready["status"],
        "siglip_loaded": siglip_model is not None,
        "xlm_loaded": xlm_model is not None,
        "gemini_available": genai_client is not None,
        "tavily_available": tavily_client is not None
    }

@app.get("/models-status")
async def models_status():
    return models_ready

@app.get("/gemini-models")
async def list_gemini_models():
    if not genai_client:
        return {"error": "Gemini not configured"}
    try:
        models = genai_client.list_models()
        return {"models": [m.name for m in models]}
    except Exception as e:
        return {"error": str(e)}

def parse_visual_label(raw_label):
    label_lower = str(raw_label).lower()
    # Explicit mapping for common model outputs
    if "label_0" in label_lower or label_lower == "0" or "fake" in label_lower:
        return "RUMOR"
    if "label_1" in label_lower or label_lower == "1" or "true" in label_lower or "real" in label_lower:
        return "NON-RUMOR"
    
    # Keyword fallback
    if any(x in label_lower for x in ["not_rumor", "non_rumor", "real", "true", "legit"]):
        return "NON-RUMOR"
    if any(x in label_lower for x in ["fake", "false", "rumor", "misinformation", "label_0"]):
        return "RUMOR"
    return "UNKNOWN"

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True
)
def gemini_generate_retry(model, content, config=None):
    return model.generate_content(content, generation_config=config)

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    MAX_SIZE = 10 * 1024 * 1024
    if file.size and file.size > MAX_SIZE:
        raise HTTPException(status_code=413, detail="Image too large. Max 10MB allowed.")
    
    if models_ready["status"] == "loading":
        raise HTTPException(status_code=503, detail="Models are still loading in the background. Please try again in 1-2 minutes.")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        file.file.seek(0)
        img_bytes = file.file.read()
        img_base64 = base64.b64encode(img_bytes).decode()
        
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
        visual_display = "UNAVAILABLE"
        if torch and siglip_model and siglip_processor:
            try:
                inputs = siglip_processor(images=image, return_tensors="pt").to(device)
                with torch.no_grad():
                    logits = siglip_model(**inputs).logits
                    probs = torch.softmax(logits, dim=1)
                    pred = probs.argmax().item()
                raw_vis = siglip_model.config.id2label[pred] if hasattr(siglip_model.config, 'id2label') else str(pred)
                visual_display = "RUMOR" if any(x in raw_vis.lower() for x in ["fake", "false", "rumor"]) else "NON-RUMOR"
                results["visual"] = visual_display
                results["visual_confidence"] = float(probs.max().item())
                print(f"[SigLIP] {visual_display}")
            except Exception as e:
                print(f"SigLIP Error: {e}")

        # --- PILLAR 2: TEXTUAL FORENSICS (XLM-R) ---
        text_display = "UNAVAILABLE"
        extracted_text_ocr = ""
        if OCR_AVAILABLE:
            try:
                extracted_text_ocr = pytesseract.image_to_string(image)
                extracted_text_ocr = re.sub(r'\s+', ' ', extracted_text_ocr).strip()[:500]
            except: pass
        
        text_input = extracted_text_ocr if extracted_text_ocr else "This is a news article about current events"
        if torch and xlm_model and xlm_tokenizer:
            try:
                inputs_xlm = xlm_tokenizer(text_input, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
                with torch.no_grad():
                    logits = xlm_model(**inputs_xlm).logits
                    text_probs = torch.softmax(logits, dim=1)
                    text_pred = text_probs.argmax().item()
                    confidence = text_probs.max().item()
                raw_txt = xlm_model.config.id2label[text_pred] if hasattr(xlm_model.config, 'id2label') else str(text_pred)
                text_display = "RUMOR" if "0" in raw_txt or "fake" in raw_txt.lower() else "NON-RUMOR"
                results["text"] = text_display
                results["text_confidence"] = float(confidence)
                print(f"[XLM-R] {text_display}")
            except Exception as e:
                print(f"XLM Error: {e}")

        # --- PILLAR 3: VISION & OCR (Gemini Phase 1) ---
        if genai_client:
            try:
                print("[Gemini] Phase 1: OCR and Initial Analysis...")
                model_names = ["gemini-3.1-pro-preview", "gemini-3-flash-preview"]
                ocr_prompt = "Extract all text from this image accurately. If there is non-English text, translate it to English. Also, provide a 1-sentence summary of what is happening in the image."
                
                selected_model = None
                for model_name in model_names:
                    try:
                        model = genai_client.GenerativeModel(model_name)
                        response = gemini_generate_retry(model, [image, ocr_prompt], config={"temperature": 0})
                        results["original_text"] = response.text.strip()
                        results["translated"] = response.text.strip() # Populate the frontend 'translated' field
                        results["gemini_model_used"] = model_name
                        selected_model = model
                        print(f"[Gemini] OCR success with {model_name}")
                        break
                    except: continue

                # --- PILLAR 4: WEB RESEARCH (Tavily) ---
                if tavily_client:
                    print("[Tavily] Running independent web research...")
                    raw_query = results["original_text"]
                    clean_query = re.sub(r'\[.*?\]', '', raw_query).strip()
                    clean_query = clean_query.replace("**", "").replace('"', '').strip()
                    search_query = clean_query[:300] if len(clean_query) > 10 else "latest fact check " + clean_query
                    if not search_query.strip(): search_query = "latest news verification"
                    
                    tav_res = tavily_client.search(query=search_query, search_depth="advanced", include_answer="advanced", max_results=5)
                    results["sources"] = [{"title": r.get("title"), "url": r.get("url")} for r in tav_res.get("results", [])]
                    results["tavily_analysis"] = tav_res.get("answer", "Web search completed.")
                    
                    # Independent Scoring
                    combined_tav = (results["tavily_analysis"] + " " + " ".join([r.get("content", "") for r in tav_res.get("results", [])[:3]])).lower()
                    r_score = sum(2 for p in ["false claim", "fake news", "hoax", "misinformation", "debunked", "fabricated"] if p in combined_tav)
                    f_score = sum(2 for p in ["confirmed by", "official statement", "verified", "true"] if p in combined_tav)
                    results["tavily"] = "RUMOR" if r_score > f_score + 1 else "NON-RUMOR"
                    print(f"[Tavily] Verdict: {results['tavily']}")

                # --- PILLAR 5: FINAL SYNTHESIS (Gemini Phase 2) ---
                if selected_model:
                    print("[Gemini] Phase 2: Synthesis Reporting...")
                    
                    system_instruction_local = """You are a professional Multimodal Fact-Checker specializing in digital news verification. Your goal is to analyze images to detect rumors, manipulation, or misinformation.

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
                    search_context = results["tavily_analysis"] if results["tavily_analysis"] else "No web search available."
                    
                    synthesis_prompt = f"""
INPUT - TRANSLATED TEXT FROM IMAGE:
{translated_context}

INPUT - WEB RESEARCH SUMMARY:
{search_context}

INPUT - SOURCES:
{str(results['sources'])}

TASK: Analyze the provided image pixels AND the 'Translated Text' against the 'Web Research' results. determine if the content is a RUMOR or NON-RUMOR.
Give your OWN forensic result based on both visual evidence and textual claim. Follow the system protocol for the 5-point report."""

                    # Note: Local SDK usage might differ slightly in how system_instruction is passed
                    # but for consistency with VeriLens style, we implement it in the prompt or config
                    response_sync = gemini_generate_retry(
                        selected_model, 
                        [system_instruction_local, image, synthesis_prompt], 
                        config={"temperature": 0}
                    )
                    sync_text = response_sync.text
                    
                    if sync_text and len(sync_text) > 20:
                        results["gemini_analysis"] = sync_text
                    else:
                        results["gemini_analysis"] = f"Investigation complete. Source context: {translated_context[:300]}..."
                    
                    # Update model name to include Tavily as requested
                    results["gemini_model_used"] = f"Model {results['gemini_model_used'].upper()} and Tavily"

                    g_cv, g_av = "NON-RUMOR", "NON-RUMOR"
                    for line in sync_text.split('\n'):
                        if 'CLAIM VERDICT:' in line.upper() and any(x in line.upper() for x in ["FALSE", "FAKE", "MISLEADING"]): g_cv = "RUMOR"
                        if 'REPORT AUTHENTICITY:' in line.upper() and any(x in line.upper() for x in ["MANIPULATED", "SUSPICIOUS", "FAKE"]): g_av = "RUMOR"
                    
                    results["gemini"] = "RUMOR" if (g_cv == "RUMOR" or g_av == "RUMOR") else "NON-RUMOR"
                    results["claim_verdict"], results["report_verdict"] = g_cv, g_av
                    print(f"[Gemini] Synthesis: {results['gemini']}")

            except Exception as e:
                print(f"Gemini/Tavily Pipeline Error: {e}")
                if not results["gemini_analysis"]:
                    results["gemini_analysis"] = f"Analysis limited by API capacity. Extracted content: {results['translated'][:400]}..."
                results["gemini"] = "ERROR"

        # --- FINAL AGGREGATION (Majority Vote with Forensic Tie-breaker) ---
        verdicts = {k: results[k] for k in ["visual", "text", "tavily", "gemini"]}
        valid_votes = [v for v in verdicts.values() if v in ["RUMOR", "NON-RUMOR"]]
        r_count, nr_count = valid_votes.count("RUMOR"), valid_votes.count("NON-RUMOR")
        
        if r_count > nr_count: results["final"] = "RUMOR"
        elif nr_count > r_count: results["final"] = "NON-RUMOR"
        else: results["final"] = results["visual"] if results["visual"] in ["RUMOR", "NON-RUMOR"] else "NON-RUMOR"
        
        results["confidence"] = float(max(r_count, nr_count) / len(valid_votes)) if valid_votes else 0.5
        print(f"[FINAL RESULT] {results['final']} (R:{r_count} NR:{nr_count})")
        return results
        
    except Exception as e:
        print(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
