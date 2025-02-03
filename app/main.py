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

app = FastAPI(title="Tarsier Model API")

# Model configuration
model_path = "omni-research/Tarsier-34b"

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


@app.post("/generate")
async def generate(request: GenerateRequest) -> Dict[str, Any]:
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Handle video if URL is provided
        if request.video_url:
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
                    
                    # Format prompt with video tag
                    prompt = f"<video>\n{request.instruction}"
                    
                    # Process inputs
                    inputs = processor(
                        prompt=prompt,
                        visual_data_file=video_path,
                        edit_prompt=True,
                        return_prompt=True
                    )
                    
                    if 'prompt' in inputs:
                        processed_prompt = inputs.pop('prompt')
                        print(f"Processed prompt: {processed_prompt}")
                    
                    # Log processor outputs
                    print("\nProcessor output keys:", inputs.keys())
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor):
                            print(f"{k} shape: {v.shape}, dtype: {v.dtype}, device: {v.device}")
                    
                    # Move inputs to model device
                    inputs = {k:v.to(model.device) for k,v in inputs.items() if v is not None}
                    
                    # Generate with the specified parameters
                    with torch.inference_mode():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=request.max_new_tokens,
                            do_sample=request.do_sample,
                            temperature=request.temperature,
                            top_p=request.top_p,
                            use_cache=True
                        )
                    
                    # Get input length for proper text decoding
                    input_length = inputs["input_ids"].shape[1]
                    
                    # Decode only the new tokens
                    generated_text = processor.tokenizer.decode(
                        outputs[0][input_length:],
                        skip_special_tokens=True
                    )
                    print(f"\nGenerated text: {generated_text}")
                    
                except Exception as e:
                    print(f"Error processing video: {str(e)}")
                    raise HTTPException(status_code=500, detail=str(e))
            finally:
                # Cleanup temporary video file
                print(f"Cleaning up temporary file: {video_path}")
                os.unlink(video_path)
        else:
            # Text-only processing
            try:
                inputs = processor(
                    prompt=request.instruction,
                    edit_prompt=True,
                    return_prompt=True
                )
                
                if 'prompt' in inputs:
                    processed_prompt = inputs.pop('prompt')
                    print(f"Processed prompt: {processed_prompt}")
                
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
                
                # Get input length for proper text decoding
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
