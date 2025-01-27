from transformers import LlavaForConditionalGeneration, AutoTokenizer

model_name = "omni-research/Tarsier-7b"

# Download and save the model
model = LlavaForConditionalGeneration.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.save_pretrained("/dev/shm")
tokenizer.save_pretrained("/dev/shm")
