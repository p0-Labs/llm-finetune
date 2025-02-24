import time
from human_eval.data import read_problems
from human_eval.execution import check_correctness
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

#import requests
#def gen_response_ollama(prompt: str, model_name: str='llama3.2:1b', endpoint='http://localhost:11434/api/generate'):
#    print(f'Generating code for prompt:\n{prompt}')
#    formatted_prompt = f'Complete the following Python function:\n\n{prompt}\n\nProvide only the function body, not the signature or docstring.'
#
#    payload = {
#        'model': model_name,
#        'prompt': formatted_prompt,
#        'stream': False,
#        'options': {
#            'temperature': 0.7,
#            'top_p': 0.9,
#            'max_tokens': 400
#        }
#    }
#
#    try:
#        response = requests.post(endpoint, json=payload, timeout=30)
#        response.raise_for_status()
#        result = response.json()
#        code = result.get('response', '').strip()
#        print(f'generated code:\n{code}')
#        return code
#    except requests.RequestException as e:
#        print(f'error calling ollama: {e}')
#        return ''

def load_local_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    d_opts = [('cuda', torch.cuda.is_available()), ('mps', torch.backends.mps.is_available()), ('cpu', True)]
    device = next(device for device, available in d_opts if available)
    print(f'using device: {device}')

    return tokenizer, model

def gen_response_local(prompt: str, tokenizer, model):
    formatted_prompt = f'Complete the following Python function:\n\n{prompt}\n\nProvide only the function body, not the signature or docstring.'
    inputs = tokenizer(formatted_prompt, return_tensors='pt')
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    code = code[len(formatted_prompt):].strip()
    print(f'generated code:\n{code}')
    return code

def benchmark(problems, tokenizer, model, use_canonical=False):
    n_problems = len(problems)
    solved = 0

    start_time = time.time()
    for task_id, problem in problems.items():
        print(f'processing problem {task_id}')
        try:
            prompt = problem['prompt']

            if use_canonical:
                generated_code = problem['canonical_solution']
            else:
                generated_code = gen_response_local(prompt, tokenizer, model)

            result = check_correctness(problem, generated_code, timeout=3.0)
            if result['passed']:
                solved += 1
            print(f'problem {task_id}: {'Solved' if result['passed'] else 'Not solved'}')
        except Exception as e:
            print(f'error processing problem {task_id}: {e}')
        return

    end_time = time.time()
    print(f'\ntotal problems: {n_problems}')
    print(f'solved problems: {solved}')
    print(f'success rate: {solved / n_problems * 100:.2f}%')
    print(f'total time: {end_time - start_time:.2f} seconds')

def main():
    tokenizer, model = load_local_model('../meta-llama/Llama-3.2-1B-Instruct')
    benchmark(read_problems(), tokenizer, model, use_canonical=False)

if __name__ == '__main__':
    main()
