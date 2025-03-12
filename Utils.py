from transformers import BertTokenizer
from datasets import load_dataset

dataset = load_dataset('glue', 'sst2')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def sample_dataset(dataset, fraction=0.1):
    return dataset.shuffle(seed=42).select(range(int(len(dataset) * fraction)))

def tokenize(sample):
  return tokenizer(sample['sentence'], truncation=True, padding='max_length', max_length=128)

train_dataset = sample_dataset(dataset['train'], fraction=0.1).map(tokenize, batched=True).remove_columns(['sentence'])
val_dataset = sample_dataset(dataset['validation'], fraction=0.1).map(tokenize, batched=True).remove_columns(['sentence'])
train_dataset.set_format('torch')
val_dataset.set_format('torch')

