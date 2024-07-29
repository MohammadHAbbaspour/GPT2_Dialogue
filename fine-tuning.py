
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer
import torch
import numpy as np
from .utils import *
from .load_pretrained import *



tokenizer.pad_token = tokenizer.eos_token

tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['utterance'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|><|SYS|>\n' }}{% endif %}"

dataset = load_dataset('mohammadhabp/Dialogue_Bot')

dataset.update({'train': dataset['train'].select(range(7000))})
dataset.update({'validation': dataset['validation'].select(range(100))})

tokenized_dataset = dataset.map(prepare_data, remove_columns=dataset['train'].column_names)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, return_tensors='pt')

training_args = TrainingArguments(
    output_dir="chat_bot-model",
    evaluation_strategy="epoch",
    auto_find_batch_size=True,
    logging_dir='./logs',
    logging_steps=10,
    push_to_hub=True,
    hub_model_id='mohammadhabp/dialogsum_gpt2',
    hub_strategy='every_save',
    hub_private_repo=False,
    eval_steps=512,
    save_strategy='epoch',
    save_steps=512,
    save_total_limit=5,
    load_best_model_at_end=True,
    metric_for_best_model='rougeL',
    save_safetensors=True,
    group_by_length=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

torch.cuda.empty_cache()

trainer.evaluate(tokenized_dataset['test'].select(range(100)))

trainer.save_model()
trainer.push_to_hub('fine tuned model')