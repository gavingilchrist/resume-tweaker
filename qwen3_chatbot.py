#### DON'T USE - DOESN'T WORK ####
from transformers import pipeline, BitsAndBytesConfig, GenerationConfig
import torch
import os
import dotenv


class ChatBot(object):
    def __init__(self, initial_prompt):
        """
        Define model and reset chat history
        """
        if not "HF_TOKEN" in os.environ:
            dotenv.load_dotenv()
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        self.model = pipeline(task="text-generation", 
                              model="Qwen/Qwen3-4B-Instruct-2507",
                              dtype=torch.bfloat16, 
                              device_map="auto",
                              model_kwargs={"quantization_config": bnb_config})
        assert self.model.device.type == 'cuda'
        
        self.gen_config = GenerationConfig(max_new_tokens=512,
                                           do_sample=True,
                                           temperature=0.6,
                                           top_p=0.95)
        self.conversation = []
        self.send(initial_prompt)
        
    def send(self, prompt):
        """
        Submit prompt, update chat history, return response
        """
        self.conversation += [{"role": "user", "content": prompt}]
        self.conversation = self.model(self.conversation, 
                                       generation_config=self.gen_config)[0]["generated_text"]
        return self.conversation[-1]['content']
