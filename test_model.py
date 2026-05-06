from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print('Loading...')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-1.5B-Instruct',
    torch_dtype=torch.float32,
    device_map='cpu',
    low_cpu_mem_usage=True,
)

print('Generating...')
messages = [
    {'role': 'system', 'content': 'You are a helpful research assistant.'},
    {'role': 'user', 'content': 'What is influencer marketing? Answer in one sentence.'}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors='pt')

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
    )

decoded = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print('Output:', decoded)