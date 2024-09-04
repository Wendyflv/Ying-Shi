# 使用mengziT5做微调
import os
import platform
import torch
import torch.nn as nn
from deal_data import DatatLoader
from datasets import load_from_disk
from transformers import AdamW
from transformers import T5ForConditionalGeneration,T5Config
from rouge import Rouge
from trainer import T5Trainer
from transformers import BertTokenizer
import matplotlib.pyplot as plt

# class MengziT5Model(nn.Module):
#     def __init__(self):
#         # 加载预训练模型
#         self.model = T5ForConditionalGeneration.from_pretrained("./mengzi-t5-base")

#     def forward(self, inputs, labels=None):
#         input_ids = inputs['input_ids']
#         attention_mask = inputs['attention_mask']

#         if labels is not None:
#             # decoder的labels
#             train_labels = labels['input_ids'].contiguous()
#             train_labels_mask = labels['attention_mask']

#             # decoder的inputs
#             decoder_input_ids = 

batch_size = 64
epochs = 20
tokenizer = BertTokenizer(vocab_file=os.path.join("./", "vocab.txt"), eos_token="[EOS]")
train_dataset = DatatLoader(batch_size, tokenizer,"./datasets/CCPC/ccpc_test_v1.0.json" )
#t5model = T5ForConditionalGeneration.from_pretrained("./mengzi-t5-base")
config = T5Config(
    vocab_size= tokenizer.vocab_size,
    d_model = 768,
    d_kv = 64,
    num_layers=12,
    decoder_start_token_id=tokenizer.cls_token_id,
    pad_token_id=tokenizer.pad_token_id,
    bos_token_id=tokenizer.cls_token_id,
    eos_token_id=tokenizer.sep_token_id
)
t5model = T5ForConditionalGeneration(config=config)
t5_trainer = T5Trainer(
    model = t5model,
    dataset=train_dataset,
    epochs = epochs
)

results = t5_trainer.train()
with open("output.txt", "w") as file:
    for item in results:
        file.write(f"{item}\n")

# 绘制损失曲线
plt.plot(list(range(1, len(results)+1)), results)
plt.title("Loss of Each Train Epochs")
plt.show()




