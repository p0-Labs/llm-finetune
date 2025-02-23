import time
from human_eval.data import read_problems
from human_eval.execution import check_correctness

def generate_code(prompt):
    '''
    Generate code for a given prompt using your LLM.
    Replace this implementation with your actual LLM interface.

    Args:
        prompt (str): The function signature and docstring from HumanEval.

    Returns:
        str: The generated code body to complete the function.

    Example using OpenAI API (uncomment and configure as needed):
        import openai
        response = openai.Completion.create(
            engine='code-davinci-002',
            prompt=prompt,
            max_tokens=200,
            temperature=0.0,
        )
        return response.choices[0].text
    '''
    raise NotImplementedError('Please implement this function to use your LLM')

use_canonical = True # set to False to use your LLM; True to test with canonical solutions

problems = read_problems()
num_problems = len(problems)
solved = 0

start_time = time.time()

for task_id, problem in problems.items():
    print(f'processing problem {task_id}')
    try:
        prompt = problem['prompt']

        if use_canonical:
            generated_code = problem['canonical_solution']
        else:
            generated_code = generate_code(prompt)

        result = check_correctness(problem, generated_code, timeout=3.0)
        if result['passed']:
            solved += 1
        print(f'problem {task_id}: {'Solved' if result['passed'] else 'Not solved'}')
    except Exception as e:
        print(f'error processing problem {task_id}: {e}')

end_time = time.time()
print(f'\ntotal problems: {num_problems}')
print(f'solved problems: {solved}')
print(f'success rate: {solved / num_problems * 100:.2f}%')
print(f'total time: {end_time - start_time:.2f} seconds')
