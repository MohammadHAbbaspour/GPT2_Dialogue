from datasets import load_metric
import re
from .load_pretrained import *


def inference(chat):
  inputs_text = tokenizer.apply_chat_template(chat, return_tensors='pt', tokenize=False)
  inputs = tokenizer(inputs_text, return_tensors='pt', truncation=True, padding=True, max_length=1024)
  inputs = {k: v.to(device) for k, v in inputs.items()}
  outputs = model.generate(**inputs, max_new_tokens=128, do_sample=True)
  return tokenizer.decode(outputs[0])


def convert_to_list(text):
  pattern = re.compile(r"\{'role': '(\w+)', 'utterance': \"(.*?)\"}|"
                         r"\{'role': '(\w+)', 'utterance': '(.*?)'\}", re.DOTALL)

  matches = pattern.findall(text)

  dialogs = []
  for match in matches:
      role = match[0] or match[2]
      utterance = match[1] or match[3]
      dialogs.append({'role': role, 'utterance': utterance})

  return dialogs


def prepare_data(record):
  context = record['context']
  response = record['response']

  context = convert_to_list(context)
  response = convert_to_list(response)

  context_text = ' '.join([f"{entry['role']}: {entry['utterance']}" for entry in context])
  response_text = response[0]['utterance']

  inputs = tokenizer(context_text, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
  labels = tokenizer(response_text, return_tensors='pt', truncation=True, padding='max_length', max_length=64).input_ids.to(device)

  inputs = {k: v.squeeze().to(device) for k, v in inputs.items()}
  labels = labels.squeeze().to(device)

  return {**inputs, 'labels': labels}


def compute_metrics(eval_preds):
    rouge_metric = load_metric("rouge")
    
    logits, labels = eval_preds.predictions, eval_preds.label_ids
    labels[labels == -100] = tokenizer.pad_token_id
    predictions = logits.argmax(axis=-1)

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    rouge = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {
        "rouge1": rouge["rouge1"].mid.fmeasure,
        "rouge2": rouge["rouge2"].mid.fmeasure,
        "rougeL": rouge["rougeL"].mid.fmeasure,
    }