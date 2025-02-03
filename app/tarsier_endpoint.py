from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
import torch
from typing import Dict, Any, Optional, List
import tempfile
import os
import requests
from urllib.parse import urlparse
import sys
import decord
from PIL import Image

# Add Tarsier to path
tarsier_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(tarsier_path)

# Import Tarsier modules
from tarsier.models.modeling_tarsier import TarsierForConditionalGeneration, LlavaConfig
from tarsier.dataset.processor import Processor
from tarsier.dataset.utils import get_visual_type, sample_frame_indices, sample_video, sample_gif, sample_image

app = FastAPI(title="Tarsier Original Implementation API")

# Model configuration
model_path = "omni-research/Tarsier-34b"

# Global variables
model = None
processor = None

def load_model_and_processor(model_name_or_path, max_n_frames=8):
    print(f"Load model and processor from: {model_name_or_path}; with max_n_frames={max_n_frames}")
    
    # Initialize processor with explicit max_n_frames
    processor = Processor(
        model_name_or_path,
        max_n_frames=max_n_frames,
        max_seq_len=None,  # Let it use model's max length
        add_sep=False,     # Don't add separator tokens
        do_image_padding=False  # Use resize instead of padding
    )
    
    # Load model config
    model_config = LlavaConfig.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    
    # Load model with specific settings
    model = TarsierForConditionalGeneration.from_pretrained(
        model_name_or_path,
        config=model_config,
        device_map='auto',
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    # Set model to eval mode
    model.eval()
    return model, processor

def debug_sample_frame_indices(start_frame, total_frames: int, n_frames: int):
    print(f"sample_frame_indices called with: start_frame={start_frame}, total_frames={total_frames}, n_frames={n_frames}")
    try:
        result = sample_frame_indices(start_frame, total_frames, n_frames)
        print(f"sample_frame_indices result: {result}")
        return result
    except Exception as e:
        print(f"Error in sample_frame_indices: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise

def process_one(model, processor, prompt, video_file, generate_kwargs):
    try:
        print(f"Inputs values: model={model}, processor={processor}, prompt={prompt}, video_file={video_file}, generate_kwargs={generate_kwargs}")
        
        # Ensure processor.max_n_frames is valid before proceeding
        if not hasattr(processor, 'max_n_frames') or not isinstance(processor.max_n_frames, int) or processor.max_n_frames < 1:
            processor.max_n_frames = 8  # Default to 8 frames as per Tarsier's default
        print(f"Using max_n_frames: {processor.max_n_frames}")
        
        # Get visual type
        visual_type = get_visual_type(video_file)
        print(f"Visual type: {visual_type}")
        
        # Load frames with explicit n_frames
        if visual_type == 'video':
            print(f"Loading video with n_frames={processor.max_n_frames}")
            try:
                # Load video
                vr = decord.VideoReader(video_file, num_threads=1, ctx=decord.cpu(0))
                total_frames = len(vr)
                fps = vr.get_avg_fps()
                print(f"Video loaded. Total frames: {total_frames}, FPS: {fps}")
                
                # Calculate frame indices
                start_frame = 0
                end_frame = total_frames - 1
                n_frames = processor.max_n_frames  # Now guaranteed to be valid
                print(f"Sampling parameters: start_frame={start_frame}, end_frame={end_frame}, n_frames={n_frames}")
                
                # Sample frames
                if n_frames == 1:
                    frame_indices = [0]
                else:
                    frame_indices = [round(i * (end_frame) / (n_frames - 1)) for i in range(n_frames)]
                print(f"Calculated frame indices: {frame_indices}")
                
                # Get frames
                frames = vr.get_batch(frame_indices).asnumpy()
                frames = [Image.fromarray(f).convert('RGB') for f in frames]
                print(f"Successfully sampled {len(frames)} frames from video")
                
            except Exception as e:
                print(f"Error in video sampling: {str(e)}")
                import traceback
                print(traceback.format_exc())
                raise
        elif visual_type == 'gif':
            frames = sample_gif(video_file, n_frames=processor.max_n_frames)  # Now guaranteed to be valid
        elif visual_type == 'image':
            frames = sample_image(video_file)
        else:
            raise ValueError(f"Unsupported visual type: {visual_type}")
            
        # Process with frames
        print("Processing frames with processor...")
        inputs = processor(prompt, images=frames, edit_prompt=True, return_prompt=True)
        
        if 'prompt' in inputs:
            print(f"Prompt: {inputs.pop('prompt')}")
            
        inputs = {k:v.to(model.device) for k,v in inputs.items() if v is not None}
        outputs = model.generate(
            **inputs,
            **generate_kwargs,
        )
        output_text = processor.tokenizer.decode(outputs[0][inputs['input_ids'][0].shape[0]:], skip_special_tokens=True)
        return output_text
    except Exception as e:
        print(f"Error processing video in process_one: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def download_video(url: str) -> str:
    """Download video from URL to temporary file."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        parsed_url = urlparse(url)
        ext = os.path.splitext(parsed_url.path)[1] or '.mp4'
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    tmp.write(chunk)
            return tmp.name
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download video: {str(e)}")

class GenerateRequest(BaseModel):
    instruction: str
    max_new_tokens: int = 512
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    video_url: Optional[HttpUrl] = None

@app.on_event("startup")
async def load_model():
    global model, processor
    try:
        print("Loading model and processor using Tarsier's implementation...")
        
        # Set memory optimization flags
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
        # Load model and processor using Tarsier's utility
        model, processor = load_model_and_processor(
            model_path,
            max_n_frames=8
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
        # Set up generation kwargs
        generate_kwargs = {
            "do_sample": request.do_sample,
            "max_new_tokens": request.max_new_tokens,
            "top_p": request.top_p,
            "temperature": request.temperature,
            "use_cache": True
        }
        
        # Handle video if URL is provided
        if request.video_url:
            video_path = await download_video(str(request.video_url))
            print(f"Video downloaded to: {video_path}")
            
            try:
                # Validate video file
                if not os.path.exists(video_path):
                    raise HTTPException(status_code=400, detail="Video file not created")
                video_size = os.path.getsize(video_path)
                print(f"Video file size: {video_size / 1024:.2f} KB")
                
                if video_size == 0:
                    raise HTTPException(status_code=400, detail="Video file is empty")
                
                # Process video using Tarsier's original implementation
                try:
                    print("Processing with instruction:", request.instruction)
                    # Format prompt according to Tarsier's expected format
                    prompt = f"USER: <video>\n{request.instruction} ASSISTANT: "
                    
                    # Use Tarsier's process_one function with explicit n_frames
                    generated_text = process_one(
                        model=model,
                        processor=processor,
                        prompt=prompt,
                        video_file=video_path,
                        generate_kwargs=generate_kwargs
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
                # Format prompt according to Tarsier's expected format
                prompt = f"USER: {request.instruction} ASSISTANT: "
                generated_text = process_one(
                    model=model,
                    processor=processor,
                    prompt=prompt,
                    video_file=None,
                    generate_kwargs=generate_kwargs
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
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 