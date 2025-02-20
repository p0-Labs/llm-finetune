# pip install torch transformers protobuf litgpt
# hugginface-cli login
# huggingface-cli download meta-llama/Llama-3.2-1B-Instruct --exclude "original/*" --local-dir meta-llama/Llama-3.2-1B-Instruct
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def get_device():
    d_opts = [('cuda', torch.cuda.is_available()), ('mps', torch.backends.mps.is_available()), ('cpu', True)]
    device = next(device for device, available in d_opts if available)
    print(f'using device: {device}')
    return device

def load_model(model_path: str):
    print('loading model...')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    print('model loaded!')
    return tokenizer, model

def generate_chat_response(tokenizer, model, conversation, max_length=100):
    print('generating response...')
    prompt = f"<s>[INST] {conversation} [/INST]"
    inputs = tokenizer(prompt, return_tensors='pt', padding=True).to(device)

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

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == '__main__':
    device = get_device()
    tokenizer, model = load_model('/Users/ln/dev/meta-llama/Llama-3.2-1B-Instruct')
    conversation = 'Write a function in python to detect the 13th Friday of a given month and year. The function should accept two parameters: the month (as a number) and the year (as a four-digit number). It should return True if the month contains a Friday the 13th, and False otherwise3.'
    response = generate_chat_response(tokenizer, model, conversation, max_length=400)
    print(response)
