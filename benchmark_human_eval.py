import time
import requests
from human_eval.data import read_problems
from human_eval.execution import check_correctness

def gen_response_local():
    pass

def gen_response_ollama(prompt, model_name='llama3.2:1b', endpoint='http://localhost:11434/api/generate'):
    print(f'Generating code for prompt:\n{prompt}')
    formatted_prompt = f'Complete the following Python function:\n\n{prompt}\n\nProvide only the function body, not the signature or docstring.'

    payload = {
        'model': model_name,
        'prompt': formatted_prompt,
        'stream': False,
        'options': {
            'temperature': 0.7,
            'top_p': 0.9,
            'max_tokens': 400
        }
    }

    try:
        response = requests.post(endpoint, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        code = result.get('response', '').strip()
        print(f'generated code:\n{code}')
        return code
    except requests.RequestException as e:
        print(f'error calling ollama: {e}')
        return ''

def benchmark(problems, gen_response, use_canonical=False):
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
                generated_code = gen_response(prompt)

            result = check_correctness(problem, generated_code, timeout=3.0)
            if result['passed']:
                solved += 1
            print(f'problem {task_id}: {'Solved' if result['passed'] else 'Not solved'}')
        except Exception as e:
            print(f'error processing problem {task_id}: {e}')

    end_time = time.time()
    print(f'\ntotal problems: {n_problems}')
    print(f'solved problems: {solved}')
    print(f'success rate: {solved / n_problems * 100:.2f}%')
    print(f'total time: {end_time - start_time:.2f} seconds')

def main():
    benchmark(read_problems(), gen_response_ollama, use_canonical=False)

if __name__ == '__main__':
    main()
