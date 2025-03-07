import json
import argparse
import os
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

class LocalLLM:
    def __init__(self, args):
        self.args = args
        self.model = LLM(
            model=args.model,
            dtype=args.dtype,
            tensor_parallel_size=args.tensor_parallel,
            gpu_memory_utilization=0.95 if ("70b" in args.model.lower() or "72b" in args.model.lower()) else 0.9,
            trust_remote_code=True,
            enforce_eager=True if ("27b" in args.model.lower() or "70b" in args.model.lower() or "72b" in args.model.lower()) else False,
            disable_custom_all_reduce=True if ("27b" in args.model.lower() or "70b" in args.model.lower() or "72b" in args.model.lower()) else False
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(args.model)
        
        # Sampling parameters
        if "qwen2.5" in args.model.lower():
            args.stop_token_ids = [151643, 151644, 151645, 151646, 151647, 151648, 151649, 151650, 151651, 151652, 151653, 151654]
        elif "llama-3.1" in args.model.lower():
            args.stop_token_ids = [128009, 128001, 128006, 128007, 128008]
        self.params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            frequency_penalty=args.frequency_penalty,
            max_tokens=args.max_tokens,
            stop_token_ids=args.stop_token_ids,
            seed=args.seed
        )
    
    def apply_chat_template(self, query: str, system: str=None) -> str:
        if system is not None and len(system) > 0:
            chat = [{"role": "system", "content": system}, {"role": "user", "content": query}]
        else:
            chat = [{"role": "user", "content": query}]
        prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        return prompt
    
    def forward(self, query: str, system: str=None) -> str:
        prompt = self.apply_chat_template(query, system)
        response = self.model.generate([prompt], self.params)
        return response[0].outputs[0].text

def parse_arguments():
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

def process_dataset(llm, input_filepath, output_filepath, prompt_filepath):
    with open(prompt_filepath, 'r', encoding='utf-8') as f:
        prompt_template = f.read()

    with open(input_filepath, 'r', encoding='utf-8') as f:
        dataset = f.readlines()

    processed_data = []
    
    for line in dataset:
        data = json.loads(line.strip())
        
        response = data.pop("response", None)

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
            result = llm.forward(query, system=system_message)
            data["predict"] = result.strip()
        except Exception as e:
            print(f"Model inference error: {e}")
            data["predict"] = None
        
        processed_data.append(data)
    
    with open(output_filepath, 'w', encoding='utf-8') as f:
        for item in processed_data:
            f.write(json.dumps(item, ensure_ascii=False, indent=4) + "\n")
    
    print(f"Processing complete. Output saved to {output_filepath}")

if __name__ == "__main__":
    args = parse_arguments()
    local_llm = LocalLLM(args)
    
    process_dataset(local_llm, args.input_filepath, args.output_filepath, args.prompt_filepath)
