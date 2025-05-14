import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, BitsAndBytesConfig
from config import PROMPT, FRAME_DIR
from singleton_class import Singleton
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

class QwenModel(Singleton):
    model = None
    tokenizer = None
    
    def __init__(self):
        pass

    @classmethod
    def load_model(cls):
        # Load model with 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        cls.model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen-VL-Chat",
            trust_remote_code=True,
        )
        cls.model.to("mps")
        cls.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)


def get_response(text, question):
    # Prepare the prompt
    prompt = PROMPT.format(text=text, question=question)
    QwenModel.load_model()

    prompt_inputs = []
    for path in os.listdir(FRAME_DIR):
        if path.endswith(".jpg"):
            image_path = os.path.join(FRAME_DIR, path)
            prompt_inputs.append({"image": image_path})

    prompt_inputs.append({'text': prompt.format(text=text, question=question)})  
    # print(prompt_inputs)       
    query = QwenModel.tokenizer.from_list_format(prompt_inputs)
    
    response, history = QwenModel.model.chat(QwenModel.tokenizer, query=query, history=None)
    
    return response