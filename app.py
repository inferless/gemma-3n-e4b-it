import os, torch, inferless
from typing import Optional
from pydantic import BaseModel, Field
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

@inferless.request
class RequestObjects(BaseModel):
    prompt: str = Field(default="Describe this image in detail.")
    image_url: str = Field(default="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg")
    system_prompt: Optional[str] = "You are a helpful assistant."
    max_new_tokens: Optional[int] = 128
    temperature: Optional[float] = 0.1
    top_p: Optional[float] = 1.0

@inferless.response
class ResponseObjects(BaseModel):
    generated_text: str = Field(default="Test output")

class InferlessPythonModel:
    def initialize(self):
        model_id = "google/gemma-3n-E4B-it"
        self.model = Gemma3nForConditionalGeneration.from_pretrained(model_id,torch_dtype=torch.bfloat16,device_map="cuda").eval()
        self.processor = AutoProcessor.from_pretrained(model_id)

    def infer(self, inputs: RequestObjects) -> ResponseObjects:
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": inputs.system_prompt}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": inputs.image_url},
                    {"type": "text",  "text": inputs.prompt}
                ]
            }
        ]

        tensor_inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        input_len = tensor_inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            out = self.model.generate(
                **tensor_inputs,
                max_new_tokens=inputs.max_new_tokens,
                temperature=inputs.temperature,
                top_p=inputs.top_p,
                do_sample=(inputs.temperature > 0)
            )[0][input_len:]

        decoded = self.processor.decode(out, skip_special_tokens=True)
        return ResponseObjects(generated_text=decoded)

    def finalize(self):
        self.model = None
