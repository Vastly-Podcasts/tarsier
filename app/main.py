from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from transformers import LlavaForConditionalGeneration
import torch
from typing import Dict, Any, Optional
import uvicorn
import tempfile
import os
import requests
from urllib.parse import urlparse

from dataset.processor import Processor
from dataset.utils import get_visual_type

app = FastAPI(title="Tarsier Model API")

# Model configuration
model_path = "omni-research/Tarsier-34b"

# Global variables
model = None
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
    global model, processor
    try:
        print("Loading model and processor...")
        
        # Set memory optimization flags
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
        # Initialize processor first
        processor = Processor(
            model_path,
            max_n_frames=8,
            do_image_padding=False
        )
        
        print(f"Found {torch.cuda.device_count()} GPUs")
        
        # First load model without device map
        model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )
        
        # Tie weights before device mapping
        print("Tying weights...")
        model.tie_weights()
        
        # Now set up device mapping
        print("Setting up device mapping...")
        model = model.to_device_map(
            device_map="auto",
            max_memory={0: "38GB", 1: "38GB", "cpu": "50GB"}
        )
        
        print("Model and processor loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

@app.post("/generate")
async def generate(request: GenerateRequest) -> Dict[str, Any]:
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Handle video if URL is provided
        if request.video_url:
            print(f"\nProcessing video from URL: {request.video_url}")
            video_path = await download_video(str(request.video_url))
            print(f"Video downloaded to: {video_path}")
            
            try:
                # Validate video file exists and has content
                if not os.path.exists(video_path):
                    raise HTTPException(status_code=400, detail="Video file not created")
                video_size = os.path.getsize(video_path)
                print(f"Video file size: {video_size / 1024:.2f} KB")
                
                if video_size == 0:
                    raise HTTPException(status_code=400, detail="Video file is empty")
                
                # Process video using Tarsier's processor
                try:
                    print("Processing with instruction:", request.instruction)
                    inputs = processor(
                        prompt=f"<video>\n{request.instruction}",
                        visual_data_file=video_path,
                        edit_prompt=True,
                        return_prompt=True
                    )
                    
                    if 'prompt' in inputs:
                        print(f"Processed prompt: {inputs.pop('prompt')}")
                    
                    print("\nProcessor output keys:", inputs.keys())
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor):
                            print(f"{k} shape: {v.shape}, dtype: {v.dtype}, device: {v.device}")
                    
                    # Move to model device
                    inputs = {k:v.to(model.device) for k,v in inputs.items() if v is not None}
                    
                    # Log input structure after device placement
                    print("\nModel inputs after device placement:")
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor):
                            print(f"{k} shape: {v.shape}, dtype: {v.dtype}, device: {v.device}")
                    
                    # Generate
                    print("\nGenerating with settings:", {
                        "max_new_tokens": request.max_new_tokens,
                        "temperature": request.temperature,
                        "top_p": request.top_p
                    })
                    
                    with torch.inference_mode():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=request.max_new_tokens,
                            do_sample=request.do_sample,
                            temperature=request.temperature,
                            top_p=request.top_p,
                            use_cache=True
                        )
                except Exception as e:
                    print(f"Error processing video: {str(e)}")
                    raise HTTPException(status_code=500, detail=str(e))
                
                print(f"\nGeneration output shape: {outputs.shape}")
                
                # Move output to CPU for decoding
                outputs = outputs.cpu()
                input_length = inputs["input_ids"].shape[1]
                print(f"Input length: {input_length}")
                
                # Decode only the new tokens
                generated_text = processor.tokenizer.decode(
                    outputs[0][input_length:],
                    skip_special_tokens=True
                )
                print(f"\nGenerated text: {generated_text}")
            finally:
                # Cleanup temporary video file
                print(f"Cleaning up temporary file: {video_path}")
                os.unlink(video_path)
        else:
            # Text-only path
            try:
                inputs = processor(
                    prompt=request.instruction,
                    edit_prompt=True,
                    return_prompt=True
                )
                
                if 'prompt' in inputs:
                    print(f"Processed prompt: {inputs.pop('prompt')}")
                
                # Move to model device
                inputs = {k:v.to(model.device) for k,v in inputs.items() if v is not None}
                
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
                generated_text = processor.tokenizer.decode(
                    outputs[0][input_length:],
                    skip_special_tokens=True
                )
            except Exception as e:
                print(f"Error processing text: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
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
        "device": str(next(model.parameters()).device) if model is not None else "not loaded"
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
