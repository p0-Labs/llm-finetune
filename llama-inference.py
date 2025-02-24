from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def get_device():
    d_opts = [('cuda', torch.cuda.is_available()), ('mps', torch.backends.mps.is_available()), ('cpu', True)]
    device = next(device for device, available in d_opts if available)
    print(f'using device: {device}')
    return device

def load_model(model_path: str, device):
    print('loading model...')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    print('model loaded!')
    return tokenizer, model

def generate_chat_response(tokenizer, model, prompt, device, max_length=100):
    print(f'prompt: {prompt}')
    print('generating response...\n')

    messages = [
        {'role': 'user', 'content': prompt}
    ]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_promp=True)
    inputs = tokenizer(formatted_prompt, return_tensors='pt', padding=True).to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_length,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

    generated_ids = outputs[0, inputs.input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response

def main():
    model_path = '../meta-llama/Llama-3.2-1B-Instruct'
    device = get_device()
    tokenizer, model = load_model(model_path, device)

    prompt = 'write a python function to print out the first n fibonacci numbers. write the function so it runs recursively. only give me the function.'
    response = generate_chat_response(tokenizer, model, prompt, device, max_length=200)
    print(response)

if __name__ == '__main__':
    main()
