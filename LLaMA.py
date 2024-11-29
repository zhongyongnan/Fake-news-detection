import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
import pandas as pd
from tqdm import tqdm
import torch.utils
import torch.utils.data
from transformers import AutoTokenizer
from datasets import Dataset, load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from transformers.pipelines.pt_utils import KeyDataset
import time
import transformers

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

class LLaMA():
    def __init__(self, model="meta-llama/Llama-2-7b-hf"):
        self.model = model
        self.device = torch.device("cuda")
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        # self.pipeline = transformers.pipeline(
        #     "text-generation",
        #     model=model,
        #     torch_dtype=torch.float16,
        #     device_map="auto",
        #     tokenizer=self.tokenizer
        # )
        print(f"Current CUDA device index: {torch.cuda.current_device()}")
        print(f"Current CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    def format_prompt(self, query):
        prompt = f"<s>{B_INST} {query.strip()} {E_INST} "
        return prompt

    def split_text(self, text, max_length):
        tokens = self.tokenizer.encode(text)
        return tokens[:max_length]

    def generate(self, q):
        prompt = self.format_prompt(q)
        prompt = self.tokenizer.decode(self.split_text(prompt, 4096), skip_special_tokens=True)

        sequences = self.pipeline(
            prompt,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=4096,
            truncation=True
        )
        # print(torch.cuda.memory_summary(device=None, abbreviated=False))

        # pdb.set_trace()
        output = ""
        for seq in sequences:
            # print(f"Result: {seq['generated_text']}")
            try:
                output += seq["generated_text"]
            except:
                output += seq[0]['generated_text']
            # print(f"response: {response}")
        # print(f"output:{output}")
        try:
            response = output.split(E_INST)[-1]
        except:
            for i in prompt:
                response = output.replace(i, "")
        # print(f"response:{response}")
        return response
    
    def find_string_after(self, target_string):
        specific_string = """Summary:"""
        start_pos = target_string.find(specific_string)
        
        if start_pos == -1:
            specific_string = "summary"
            start_pos = target_string.lower().find(specific_string)
            if start_pos == -1:
                return None  # Specific string not found
        
        # Calculate the starting position of the string after the specific string
        start_pos += len(specific_string)
        
        # Extract the string after the specific string
        result = target_string[start_pos:]
        
        return result


    def batch_generate(self, dataset, key, bs=1):
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        def collate_fn(batch):
            promps = [each[key] for each in batch]
            tokens = self.tokenizer(promps, padding=True, truncation=True, return_tensors="pt", add_special_tokens=False, max_length=1024)
            return tokens

        result = []
        loader = torch.utils.data.DataLoader(dataset, batch_size=bs, collate_fn=collate_fn)
        for batch in tqdm(loader, total=len(loader)):
            batch = batch.to(self.device)
            # 1.9300113341968912 bs=32
            # 1.7295081967213115 bs=4
            # 1.8747072599531616 bs=8
            # 每个token需要算更多的attention
            # print(len(batch['input_ids'][0]) * bs)
            with torch.no_grad():
                # print('=' * 10 + " before " + '=' * 10)
                # print(torch.cuda.memory_summary(device=None, abbreviated=False))
                # print('=' * 20)
                outputs = self.model.generate(**batch, max_new_tokens=256) # 这个outputs就是显存泄漏的来源
                # print('=' * 10 + " after " + '=' * 10)
                # print(torch.cuda.memory_summary(device=None, abbreviated=False))
                # print('=' * 20)
                try:
                    with open('./rec.txt', "r") as f:
                        result = eval(f.read())
                except Exception as e:
                    print(e)
                result += [self.find_string_after(self.tokenizer.decode(each, skip_special_tokens=True).strip()) for each in outputs]
                try:
                    with open('./rec.txt', "w") as f:
                        f.write(str(result))
                except Exception as e:
                    print(e)
                # print(result)
                # ===== 释放显存 =====
                del outputs
                del batch
                del result
                torch.cuda.empty_cache()
                # ==================
        try:
            with open('./rec.txt', "r") as f:
                result = eval(f.read())
        except Exception as e:
            print(e)
        return result