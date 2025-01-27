from typing import Dict, List, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class EndpointHandler():
    def __init__(self, path: str = ""):
        """
        The EndpointHandler is initialized once when the Inference Endpoint is started.
        `path` will be the path to your model directory on disk.
        """
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
        
        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        This method is called every time the Endpoint receives a request.

        Args:
            data (Dict[str, Any]):
                - data["inputs"]: The prompt string (or list of prompts) for text generation.
                - data["parameters"]: Optional dictionary of generation parameters 
                  (e.g. max_new_tokens, temperature, top_p, etc.)

        Returns:
            A list of dictionaries. Each dictionary is one generation result:
                [ {"generated_text": "..."} ]
        """
        # 1) Get the prompt(s). If none is provided, return an error.
        prompt = data.get("inputs", None)
        if prompt is None:
            return [{"error": "No 'inputs' field provided in request."}]

        # 2) Retrieve optional generation parameters
        #    Feel free to add or remove parameters based on your needs.
        params = data.get("parameters", {})
        max_new_tokens = params.get("max_new_tokens", 50)
        temperature = params.get("temperature", 1.0)
        top_p = params.get("top_p", 0.9)
        do_sample = params.get("do_sample", True)

        # 3) If the user provided a single string, wrap it in a list for consistent handling
        if isinstance(prompt, str):
            prompt = [prompt]

        # 4) Tokenize the input(s)
        #    Note: We call the tokenizer in a loop if there's more than one prompt
        #    so each is turned into a separate batch element.
        batch_encodings = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        batch_encodings = {k: v.to(self.device) for k, v in batch_encodings.items()}

        # 5) Generate output
        with torch.no_grad():
            outputs = self.model.generate(
                **batch_encodings,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample
            )

        # 6) Decode model outputs to strings
        #    We'll create a list of dicts: [{"generated_text": "..."}]
        generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return [{"generated_text": text} for text in generated_texts]
