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

    tokenizer = load_tokenizer()

    f = open(local_path, 'r')
    Lines = f.readlines()
    f.close
    encodings = []
    labels = []
    max_length = 512

    for line in Lines:
      item = json.loads(line)
      prompt = item['article']
      choice0 = item['question'].replace('@placeholder', item['option_0'])
      choice1 = item['question'].replace('@placeholder', item['option_1'])
      choice2 = item['question'].replace('@placeholder', item['option_2'])
      choice3 = item['question'].replace('@placeholder', item['option_3'])
      choice4 = item['question'].replace('@placeholder', item['option_4'])
      encoding = tokenizer([prompt, prompt, prompt, prompt, prompt], [choice0, choice1, choice2, choice3, choice4], return_tensors='pt', truncation=True, padding='max_length', max_length=max_length)
      encodings.append(encoding)
      labels.append(item['label'])

    return encodings, labels

def load_model():
    return DistilBertForMultipleChoice.from_pretrained('distilbert-base-uncased')

def load_tokenizer():
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

def compute_metrics(pred):
    labels = pred.label_ids
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
    }

def train_model():
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    trainer.train()

    # trainer.save_model('where?')

    return trainer
    
def main():
    # Check for GPU.
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    encodings, labels = load_data(sys.argv[1])

    train_data, eval_data, train_labels, eval_labels = train_test_split(encodings, labels, test_size=.2)

    #train_dataset = RecamDataset(train_data, train_labels)
    #val_dataset = RecamDataset(eval_data, eval_labels)

    #model = load_model()

    #trainer = train_model()

    #trainer.evaluate()
    
if __name__ == "__main__":
    main()


