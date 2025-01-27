from transformers import LlavaForConditionalGeneration, AutoTokenizer
import torch

# Initialize the model and tokenizer
model = LlavaForConditionalGeneration.from_pretrained(
    "model",
    device_map="cpu",
    torch_dtype=torch.float32
)
tokenizer = AutoTokenizer.from_pretrained("model")

# Prepare your input data with a shorter generation length
data = {
    "inputs": "Say hello to the world",
    "parameters": {
        "max_new_tokens": 20,  # Reduced for faster generation
        "temperature": 0.7,    # Lower temperature for faster, more focused generation
        "top_p": 0.9,
        "do_sample": True
    }
}

# Tokenize input
inputs = tokenizer(data["inputs"], return_tensors="pt")

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