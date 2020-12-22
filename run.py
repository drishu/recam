# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Install huggingface transformers.
# !pip install transformers

import torch
import json
import numpy as np
import sys
from tensorflow import keras
from transformers import DistilBertTokenizerFast, DistilBertForMultipleChoice, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class RecamDataset(torch.utils.data.Dataset):
  def __init__(self, encodings, labels):
    self.encodings = encodings
    self.labels = labels

  def __getitem__(self, idx):
    item = {
      'input_ids': self.encodings[idx]['input_ids'],
      'attention_mask': self.encodings[idx]['attention_mask'],
      'labels': self.labels[idx],
    }
    return item

  def __len__(self):
    return len(self.labels)

def load_data(path):
    """ Load data and tokenize """
    local_path = keras.utils.get_file("data.json", path)

    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name, cache_dir=cache_dir)

    f = open(local_path, 'r')
    Lines = f.readlines()
    f.close
    encodings = []
    labels = []

    for line in Lines:
      item = json.loads(line)
      prompt = [
        item['article'],
        item['article'],
        item['article'],
        item['article'],
        item['article']
      ]
      choices = [
        item['question'].replace('@placeholder', item['option_0']),
        item['question'].replace('@placeholder', item['option_1']),
        item['question'].replace('@placeholder', item['option_2']),
        item['question'].replace('@placeholder', item['option_3']),
        item['question'].replace('@placeholder', item['option_4'])
      ]
      encoding = tokenizer(prompt, choices, return_tensors='pt', truncation=True, padding='max_length', max_length=512)
      encodings.append(encoding)
      labels.append(item['label'])

    return encodings, labels

def compute_metrics(pred):
    labels = pred.label_ids
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
    }
    
def main():
    encodings, labels = load_data(sys.argv[2])

    train_data, eval_data, train_labels, eval_labels = train_test_split(encodings, labels, test_size=.2)

    train_dataset = RecamDataset(train_data, train_labels)
    val_dataset = RecamDataset(eval_data, eval_labels)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    trainer.train()
    trainer.save_model(model_dir)
    trainer.evaluate()
    
if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model_name = sys.argv[1]
    cache_dir = './recam/cache/' + model_name
    model_dir = './recam/models/' + model_name
    model = DistilBertForMultipleChoice.from_pretrained(model_name, cache_dir=cache_dir)
    model.to(device)
    main()


