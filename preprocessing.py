from datasets import load_dataset
import re
import pandas as pd
import numpy as np


class PreProcessDataset:
  def __init__(self, dataset, seed=42):
    self.train_dataset = dataset['train']
    self.test_dataset = dataset['test']
    self.validation_dataset = dataset['validation']
    self.preprocessed_train_dataset = None
    self.preprocessed_test_dataset = None
    self.preprocessed_validation_dataset = None
    self.seed = seed

  def _split_dialogue(self, record):
    dialogue = record['dialogue']
    splited_dialogue = re.split('#Person1#:|#Person2#:', dialogue)

    role_utterance_dialogue = []
    i = 0
    for u in splited_dialogue:
      if u.strip():
        if i % 2 == 0:
          role_utterance_dialogue.append({'role': 'USR', 'utterance': u.strip()})
        else:
          role_utterance_dialogue.append({'role': 'SYS', 'utterance': u.strip()})
        i += 1

    return role_utterance_dialogue

  def get_context_response(self, record):
    np.random.seed(self.seed)
    dialogue = self._split_dialogue(record)
    sys_utterance_ids = [i for i, d in enumerate(dialogue) if d['role'] == 'SYS']
    response_id = np.random.choice(sys_utterance_ids)
    context = dialogue[:response_id]
    response = dialogue[response_id]
    return {'context': context, 'response': response, 'turns number': response_id}

  def call(self):
    self.preprocessed_train_dataset = self.train_dataset.map(self.get_context_response)
    self.preprocessed_test_dataset = self.test_dataset.map(self.get_context_response)
    self.preprocessed_validation_dataset = self.validation_dataset.map(self.get_context_response)

    self.preprocessed_train_dataset = self.preprocessed_train_dataset.remove_columns(["dialogue", "summary", "topic"])
    self.preprocessed_test_dataset = self.preprocessed_test_dataset.remove_columns(["dialogue", "summary", "topic"])
    self.preprocessed_validation_dataset = self.preprocessed_validation_dataset.remove_columns(["dialogue", "summary", "topic"])


  def save(self, root_path):
    self.preprocessed_train_dataset.to_csv(root_path + '/train.csv')
    self.preprocessed_test_dataset.to_csv(root_path + '/test.csv')
    self.preprocessed_validation_dataset.to_csv(root_path + '/validation.csv')

  @property
  def columns(self):
    return {
        'train': self.preprocessed_train_dataset.column_names,
        'validation': self.preprocessed_validation_dataset.column_names,
        'test': self.preprocessed_test_dataset.column_names
    }


dataset = load_dataset('./dialogsum')

pre_processed_ds = PreProcessDataset(dataset)
pre_processed_ds.call()
pre_processed_ds.save(root_path='./data')
print(pre_processed_ds.columns)

data_root_path = './data/'
data_pathes = {
    'train': data_root_path + 'train.csv',
    'validation': data_root_path + 'validation.csv',
    'test': data_root_path + 'test.csv'
}

dataset = load_dataset('csv', data_files=data_pathes)
dataset.push_to_hub("mohammadhabp/Dialogue_Bot")