from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_checkpoint = "openai-community/gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

model = AutoModelForCausalLM.from_pretrained(
    model_checkpoint,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
).to(device)