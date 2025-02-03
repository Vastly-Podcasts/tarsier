from transformers import LlavaForConditionalGeneration, AutoTokenizer
import torch

# Replace with your desired Tarsier model
model_path = "omni-research/Tarsier-34b"

# Define device map for multi-GPU distribution
device_map = "auto"  # or you can provide a dict with memory limits per GPU
max_memory = {
    0: "14GB",  # GPU 0
    1: "14GB",  # GPU 1
    2: "14GB",  # GPU 2
    3: "14GB",  # GPU 3
}

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Load model with device mapping and memory limits
model = LlavaForConditionalGeneration.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map=device_map,
    max_memory=max_memory
)

# Convert weights to FP16 to reduce memory usage (T4 GPUs support fp16)
model.half()

# A simple text prompt
prompt = """You are Tarsier-34B, a large language model. Please reply politely:
Human: Hello Tarsier, how are you doing today?
Assistant:"""

# Tokenize the prompt
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

# Generate
# Adjust these parameters to your liking
generation_output = model.generate(
    input_ids=input_ids,
    max_new_tokens=128,
    do_sample=False,
    temperature=0.7,
    top_p=0.9,
    num_beams=1
)

# Decode and print the model output
generated_text = tokenizer.decode(generation_output[0], skip_special_tokens=True)
print(generated_text)