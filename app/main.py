from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, HttpUrl
from transformers import AutoTokenizer, AutoConfig, AutoProcessor
import torch
from typing import Dict, Any, List, Optional, Union
import uvicorn
import tempfile
import os
from huggingface_hub import snapshot_download
import multiprocessing
import requests
from urllib.parse import urlparse

app = FastAPI(title="Tarsier Model API")

# Model configuration
model_path = "omni-research/Tarsier-34b"

def create_device_map(config):
    """Create a balanced device map based on actual model configuration"""
    # Get number of layers from text config
    try:
        num_layers = config.text_config.num_hidden_layers
        vision_layers = config.vision_config.num_hidden_layers
    except AttributeError:
        print("Error: 'config' does not have 'text_config' or 'vision_config' attributes.")
        raise
    
    # Validate configuration
    print(f"\nModel Configuration:")
    print(f"- Text model layers: {num_layers} (expected: 60)")
    print(f"- Vision model layers: {vision_layers} (expected: 24)")
    print(f"- Hidden size: {config.text_config.hidden_size} (expected: 7168)")
    
    if num_layers != 60:
        print(f"WARNING: Unexpected number of text layers: {num_layers}")
    if vision_layers != 24:
        print(f"WARNING: Unexpected number of vision layers: {vision_layers}")
    
    # Calculate split point - put half on each GPU
    split_point = num_layers // 2
    
    device_map = {
        "language_model.model.embed_tokens": 0,
        "language_model.model.norm": 1,
        "language_model.lm_head": 1,
        "vision_model": 0,  # Vision processing on first GPU
        "mm_projector": 0,  # Projector on first GPU
        "multi_modal_projector": 0,  # Add multi_modal_projector to first GPU
        "multi_modal_projector.linear_1": 0,
        "multi_modal_projector.linear_2": 0,
        "vision_tower": 0,  # Add vision tower to first GPU
        "vision_tower.vision_model": 0,
        "vision_tower.vision_model.embeddings": 0,
        "vision_tower.vision_model.encoder": 0,
        "vision_tower.vision_model.layernorm": 0,
        "vision_tower.vision_model.pooler": 0
    }
    
    # Distribute layers evenly
    for i in range(num_layers):
        device_map[f"language_model.model.layers.{i}"] = 0 if i < split_point else 1
    
    print(f"\nDevice Mapping:")
    print(f"- GPU 0: {split_point} text layers + vision model ({vision_layers} layers)")
    print(f"- GPU 1: {num_layers - split_point} text layers")
    return device_map

# Global variables
model = None
tokenizer = None
processor = None

class GenerateRequest(BaseModel):
    instruction: str  # User must provide their own instruction/query
    max_new_tokens: int = 512
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    video_url: Optional[HttpUrl] = None  # Optional video URL - if not provided, treat as text-only request

async def download_video(url: str) -> str:
    """Download video from URL to temporary file."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get file extension from URL or default to .mp4
        parsed_url = urlparse(url)
        ext = os.path.splitext(parsed_url.path)[1] or '.mp4'
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    tmp.write(chunk)
            return tmp.name
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download video: {str(e)}")

@app.on_event("startup")
async def load_model():
    global model, tokenizer, processor
    try:
        print("Loading tokenizer and processor...")
        
        # Set memory optimization flags
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
        # Use snapshot_download for parallel downloading
        num_workers = min(16, multiprocessing.cpu_count() // 2)  # Use half of CPU cores, max 16
        print(f"Downloading model files with {num_workers} threads...")
        cache_dir = snapshot_download(
            model_path,
            max_workers=num_workers,
            resume_download=True
        )
        print("Download complete, loading model components...")
        
        tokenizer = AutoTokenizer.from_pretrained(cache_dir)
        processor = AutoProcessor.from_pretrained(cache_dir)
        
        print(f"Found {torch.cuda.device_count()} GPUs")
        
        # Let HuggingFace handle device placement automatically
        print("Loading model...")
        from transformers import LlavaForConditionalGeneration
        
        model = LlavaForConditionalGeneration.from_pretrained(
            cache_dir,
            device_map="auto",  # Let HF handle device placement
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            max_memory={0: "38GB", 1: "38GB", "cpu": "50GB"}
        )
        
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

@app.post("/generate")
async def generate(request: GenerateRequest) -> Dict[str, Any]:
    if model is None or tokenizer is None or processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Handle video if URL is provided
        if request.video_url:
            video_path = await download_video(str(request.video_url))
            try:
                # Process video and text together
                processor_output = processor(
                    text=request.instruction,
                    videos=video_path,
                    return_tensors="pt"
                )
                
                # Let the model handle device placement
                inputs = {
                    k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in processor_output.items()
                }
                
                # Generate
                with torch.inference_mode():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=request.max_new_tokens,
                        do_sample=request.do_sample,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        use_cache=True
                    )
                
                # Move output to CPU for decoding
                outputs = outputs.cpu()
                input_length = inputs["input_ids"].shape[1]
                
                # Decode only the new tokens
                generated_text = tokenizer.decode(
                    outputs[0][input_length:],
                    skip_special_tokens=True
                )
            finally:
                # Cleanup temporary video file
                os.unlink(video_path)
        else:
            # Text-only path
            processor_output = tokenizer(
                request.instruction,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            # Let the model handle device placement
            inputs = {
                k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                for k, v in processor_output.items()
            }
            
            # Generate
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=request.max_new_tokens,
                    do_sample=request.do_sample,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    use_cache=True
                )
            
            # Move output to CPU for decoding
            outputs = outputs.cpu()
            input_length = inputs["input_ids"].shape[1]
            
            # Decode only the new tokens
            generated_text = tokenizer.decode(
                outputs[0][input_length:],
                skip_special_tokens=True
            )
        
        return {
            "generated_text": generated_text,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(model.device) if model is not None else "not loaded"
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
