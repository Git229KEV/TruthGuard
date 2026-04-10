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

MODELS = [
    "gemini-3.1-pro-preview",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite-preview",
]

def try_generate_content(params, model_index=0):
    if model_index >= len(MODELS):
        raise Exception("All Gemini models failed.")
    
    model_name = MODELS[model_index]
    try:
        # genai_client is defined globally
        response = genai_client.models.generate_content(
            model=model_name,
            contents=params.get("contents"),
            config=params.get("config")
        )
        return {"text": response.text or "", "model": model_name}
    except Exception as e:
        print(f"[WARN] Model {model_name} failed, trying next... {e}")
        return try_generate_content(params, model_index + 1)


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
        
        # --- NEW PILLAR 3: EXTRACTION (Gemini) ---
        extraction_data = {"headline": "N/A", "description": "N/A"}
        if genai_client:
            try:
                print("[Gemini] Extracting news data...")
                from google.genai.types import Content, Part, GenerateContentConfig
                
                extraction_prompt = "Analyze this news card image. Extract the headline and the description. Return the result in JSON format with keys \"headline\" and \"description\"."
                
                # Config with JSON schema
                config = GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema={
                        "type": "OBJECT",
                        "properties": {
                            "headline": {"type": "STRING"},
                            "description": {"type": "STRING"},
                        },
                        "required": ["headline", "description"],
                    },
                    temperature=0
                )

                extract_res = try_generate_content({
                    "contents": [
                        Part(inline_data={"mime_type": "image/jpeg", "data": img_base64}),
                        Part(text=extraction_prompt)
                    ],
                    "config": config
                })
                
                import json
                extraction_data = json.loads(extract_res["text"])
                results["original_text"] = extraction_data.get("headline", "")
                results["translated"] = extraction_data.get("description", "")
                results["gemini_model_used"] = extract_res["model"]
                print(f"[Gemini] Extracted: {extraction_data['headline'][:50]}...")
            except Exception as e:
                print(f"Extraction Error: {e}")

        # --- NEW PILLAR 4: INTERNAL ANALYSIS (Gemini) ---
        if genai_client and extraction_data["headline"] != "N/A":
            try:
                print("[Gemini] Internal Analysis...")
                internal_prompt = f"""You are a news fact-checker. Analyze the following news:
Headline: {extraction_data['headline']}
Description: {extraction_data['description']}

Determine if this is a rumor or non-rumor based on your internal knowledge.

CRITICAL DIRECTIVE: If your analysis identifies this as a "misconception", "false claim", "misinformation", "hoax", "fake", "debunked", or if there are any negative indicators regarding its truthfulness, you MUST classify it as a Rumor (isRumor: true).

Provide a clear reason for your conclusion.
Return the result in JSON format with keys "isRumor" (boolean) and "reason" (string)."""

                config_analysis = GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema={
                        "type": "OBJECT",
                        "properties": {
                            "isRumor": {"type": "BOOLEAN"},
                            "reason": {"type": "STRING"},
                        },
                        "required": ["isRumor", "reason"],
                    },
                    temperature=0
                )

                analysis_res = try_generate_content({
                    "contents": internal_prompt,
                    "config": config_analysis
                })
                
                import json
                analysis_data = json.loads(analysis_res["text"])
                results["gemini"] = "RUMOR" if analysis_data.get("isRumor") else "NON-RUMOR"
                results["gemini_analysis"] = analysis_data.get("reason", "")
                print(f"[Gemini] Internal Verdict: {results['gemini']}")
            except Exception as e:
                print(f"Internal Analysis Error: {e}")

        # --- NEW PILLAR 5: WEB RESEARCH (Tavily + Gemini) ---
        if tavily_client and extraction_data["headline"] != "N/A":
            try:
                print("[Tavily] Running web research...")
                search_query = f"{extraction_data['headline']} {extraction_data['description']}"[:300]
                tav_res = tavily_client.search(query=search_query, search_depth="advanced", include_answer="advanced", max_results=5)
                results["sources"] = [{"title": r.get("title"), "url": r.get("url")} for r in tav_res.get("results", [])]
                
                print("[Gemini] Analyzing Tavily results...")
                tavily_prompt = f"""You are a news fact-checker. Analyze the following news and the search results from Tavily:

News Headline: {extraction_data['headline']}
News Description: {extraction_data['description']}

Tavily Search Results:
{json.dumps(tav_res, indent=2)}

Based on these search results, determine if the news is a rumor or non-rumor.

CRITICAL DIRECTIVE: If the search results or your analysis contain terms like "misconception", "false claim", "misinformation", "hoax", "fake", "debunked", or any negative indicators that the claim is not true, you MUST classify it as a Rumor (isRumor: true).

Provide a clear reason for your conclusion based on the evidence in the search results.
Return the result in JSON format with keys "isRumor" (boolean) and "reason" (string)."""

                tav_analysis_res = try_generate_content({
                    "contents": tavily_prompt,
                    "config": config_analysis # Reuse the same schema config
                })
                
                import json
                tav_analysis_data = json.loads(tav_analysis_res["text"])
                results["tavily"] = "RUMOR" if tav_analysis_data.get("isRumor") else "NON-RUMOR"
                results["tavily_analysis"] = tav_analysis_data.get("reason", "")
                print(f"[Tavily] Research Verdict: {results['tavily']}")
            except Exception as e:
                print(f"Tavily Analysis Error: {e}")

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
