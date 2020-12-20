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
!pip install transformers

import torch
import json
import numpy as np
import sys
from tensorflow import keras
from transformers import DistilBertTokenizerFast, DistilBertForMultipleChoice, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

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

def load_data(path, max_length, tokenizer):
    """ Load data and tokenize """
    local_path = keras.utils.get_file("data.json", path)

    f = open(local_path, 'r')
    Lines = f.readlines()
    f.close
    encodings = []
    labels = []

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

def load_model(path):
    return DistilBertForMultipleChoice.from_pretrained(path)

def load_tokenizer():
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

def main():
    # Check for GPU.
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Parse parameters.
    args = sys.argv.pop(0)

    # Load data, prepare features and tokenize
    tokenizer = load_tokenizer()
    encodings, labels = load_data(args[0], args[1], tokenizer)

    dataset = RecamDataset(encodings, labels)

    model = load_model(args[3])
    
if __name__ == "__main__":
    main()


