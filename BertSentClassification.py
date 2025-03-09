import torch.nn as nn

class BertSentClassification(nn.Module):
  def __init__(self, bert, num_labels):
    super().__init__()
    self.bert = bert
    self.clf = nn.Linear(bert.config.hidden_size, num_labels)

  def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
    outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).pooler_output
    logits = self.clf(outputs)

    loss = None
    if labels is not None:
      loss_fct = nn.CrossEntropyLoss()
      loss = loss_fct(logits, labels)

    return (loss, logits) if loss is not None else logits

