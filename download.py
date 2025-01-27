from transformers import LlavaForConditionalGeneration, AutoTokenizer

model_name = "omni-research/Tarsier-7b"

# Download and save the model
model = LlavaForConditionalGeneration.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.save_pretrained("model", max_shard_size="2GB")
tokenizer.save_pretrained("model")
