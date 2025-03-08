import json
import argparse
import os
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

class LLMProcessor:
    def __init__(self):
        self.args = self._parse_arguments()
        self.model = self._initialize_model()
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model)
        self.params = self._initialize_sampling_params()

    def _parse_arguments(self):
        parser = argparse.ArgumentParser(description="vllm_inference")
        parser.add_argument("--model", type=str, default='/mnt/bn/tokendance/models/Qwen2.5-72B-Instruct')
        parser.add_argument("--dtype", type=str, default='float16')
        parser.add_argument("--tensor_parallel", type=int, default=1)
        parser.add_argument("--temperature", type=float, default=0.7)
        parser.add_argument("--top_p", type=float, default=0.9)
        parser.add_argument("--repetition_penalty", type=float, default=1.0)
        parser.add_argument("--frequency_penalty", type=float, default=0.0)
        parser.add_argument("--max_tokens", type=int, default=4096)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--input_filepath", type=str, required=True, help="Path to input data file (txt format)")
        parser.add_argument("--output_filepath", type=str, required=True, help="Path to output file")
        parser.add_argument("--prompt_filepath", type=str, required=True, help="Path to prompt file")
        return parser.parse_args()

    def _initialize_model(self):
        return LLM(
            model=self.args.model,
            dtype=self.args.dtype,
            tensor_parallel_size=self.args.tensor_parallel,
            gpu_memory_utilization=0.95 if ("70b" in self.args.model.lower() or "72b" in self.args.model.lower()) else 0.9,
            trust_remote_code=True,
            enforce_eager=True if ("27b" in self.args.model.lower() or "70b" in self.args.model.lower() or "72b" in self.args.model.lower()) else False,
            disable_custom_all_reduce=True if ("27b" in self.args.model.lower() or "70b" in self.args.model.lower() or "72b" in self.args.model.lower()) else False
        )

    def _initialize_sampling_params(self):
        if "qwen2.5" in self.args.model.lower():
            stop_token_ids = [151643, 151644, 151645, 151646, 151647, 151648, 151649, 151650, 151651, 151652, 151653, 151654]
        elif "llama-3.1" in self.args.model.lower():
            stop_token_ids = [128009, 128001, 128006, 128007, 128008]
        else:
            stop_token_ids = []
        return SamplingParams(
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            repetition_penalty=self.args.repetition_penalty,
            frequency_penalty=self.args.frequency_penalty,
            max_tokens=self.args.max_tokens,
            stop_token_ids=stop_token_ids,
            seed=self.args.seed
        )

    def _apply_chat_template(self, query: str, system: str = None) -> str:
        if system is not None and len(system) > 0:
            chat = [{"role": "system", "content": system}, {"role": "user", "content": query}]
        else:
            chat = [{"role": "user", "content": query}]
        return self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    def _forward(self, query: str, system: str = None) -> str:
        prompt = self._apply_chat_template(query, system)
        response = self.model.generate([prompt], self.params)
        return response[0].outputs[0].text

    def process_dataset(self):
        with open(self.args.prompt_filepath, 'r', encoding='utf-8') as f:
            prompt_template = f.read()

        with open(self.args.input_filepath, 'r', encoding='utf-8') as f:
            dataset = f.readlines()

        processed_data = []
        
        for line in dataset:
            data = json.loads(line.strip())
            
            # 检查 lurker 字段，如果为 0 则跳过当前数据
            if data.get("lurker", 1) == 0:
                continue
            
            data.pop("response", None)

            input_str = ""
            for key, value in data.items():
                if isinstance(value, list):
                    value_str = json.dumps(value, ensure_ascii=False, indent=4)
                else:
                    value_str = str(value)
                input_str += f"{key}: {value_str}\n"
            
            query_template = prompt_template.replace("<INPUT_PLACEHOLDER>", input_str)

            history = data.get("history", [])
            if history:
                history_str = "\n".join([f"{msg}" for msg in history])
                query = f"{history_str}\nUser: {query_template}"
            else:
                query = f"User: {query_template}"
            
            system_message = "You are a helpful assistant."
            
            try:
                result = self._forward(query, system=system_message)
                data["predict"] = result.strip()
            except Exception as e:
                print(f"Model inference error: {e}")
                data["predict"] = None
            
            processed_data.append(data)
        
        with open(self.args.output_filepath, 'w', encoding='utf-8') as f:
            for item in processed_data:
                f.write(json.dumps(item, ensure_ascii=False, indent=4) + "\n")
        
        print(f"Processing complete. Output saved to {self.args.output_filepath}")

if __name__ == "__main__":
    processor = LLMProcessor()
    processor.process_dataset()