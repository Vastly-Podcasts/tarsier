from transformers import LlavaForConditionalGeneration, AutoTokenizer
import torch
import sys

# Check if input text is provided
if len(sys.argv) < 2:
    print("Usage: python usage.py 'your text here'")
    sys.exit(1)

# Check if CUDA is actually available
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print(f"Using device: {device}, dtype: {dtype}")

# Initialize the model and tokenizer
model = LlavaForConditionalGeneration.from_pretrained(
    "model",
    device_map="auto" if device == "cuda" else None,
    torch_dtype=dtype
)

if device == "cpu":
    model = model.to(device)  # Explicitly move to CPU if needed

tokenizer = AutoTokenizer.from_pretrained("model")

# Get input text from command line argument
input_text = sys.argv[1]

# Prepare your input data with a shorter generation length
data = {
    "inputs": input_text,
    "parameters": {
        "max_new_tokens": 20,  # Reduced for faster generation
        "temperature": 0.7,    # Lower temperature for faster, more focused generation
        "top_p": 0.9,
        "do_sample": True
    }
}

# Tokenize input and move to same device as model
inputs = tokenizer(data["inputs"], return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}  # Move inputs to same device as model

# Generate with a timeout
try:
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=data["parameters"]["max_new_tokens"],
            temperature=data["parameters"]["temperature"],
            top_p=data["parameters"]["top_p"],
            do_sample=data["parameters"]["do_sample"]
        )
    
    # Decode and print the generated text
    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    print(generated_text)
except Exception as e:
    print(f"An error occurred: {e}") 