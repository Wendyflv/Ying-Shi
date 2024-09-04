import torch
from torch.optim import AdamW
import torch.nn as nn
from datetime import datetime
from tqdm import tqdm
from transformers import GPT2LMHeadModel, get_linear_schedule_with_warmup, T5ForConditionalGeneration

from deal_data import DatatLoader


class T5Trainer:
    def __init__(self, model:T5ForConditionalGeneration, dataset:DatatLoader, epochs: int =10, max_grad_norm: float = 1.0,lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, gradient_accumulation: float = 1,
                 warmup_steps=10000, log_freq: int = 100):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.gradient_accumulation  = gradient_accumulation 
        self.max_grad_norm = max_grad_norm
        self.log_freq = log_freq
        self.epochs = epochs
        # print(self.model)

        # 训练集
        self.train_data = dataset
        # 忽略padding带来的影响
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        self.optim = AdamW(self.model.parameters(), lr = lr, betas=betas, weight_decay= weight_decay)
        total_step = dataset.steps / gradient_accumulation * epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optim, num_warmup_steps=warmup_steps, num_training_steps=total_step)
        

    def train(self):
        str_code = "train"

        self.model.train()
        avg_loss, overall_step = 0.0, int(0)
        epoch_losses = []

        for epoch in range(self.epochs):
            epoch_loss = 0.0

            data_iter = tqdm(
                range(len(self.train_data)), desc="EP_%s:%d" % (str_code, epoch + 1),
                total=len(self.train_data), bar_format="{l_bar}{r_bar}")
            
            for i in data_iter:
                tokens, targets, attn_x, attn_y = self.train_data[i]
                tokens, targets = tokens.to(self.device), targets.to(self.device)
                attn_x, attn_y = attn_x.to(self.device), attn_y.to(self.device)

                # 前向传播
                outputs = self.model.forward(input_ids=tokens, labels=targets)

                loss, _ = outputs[:2]
                epoch_loss += loss.item()

                if (overall_step + 1) % self.gradient_accumulation == 0:
                    avg_loss += loss.item()
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()
                    self.scheduler.step()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)

                if (overall_step + 1) % self.log_freq == 0:
                    print('now time: {}:{}. Step {} of of epoch {}, loss {}'.format(
                        datetime.now().hour,
                        datetime.now().minute,
                        i + 1,
                        epoch + 1,
                        avg_loss * self.gradient_accumulation / (self.log_freq / self.gradient_accumulation)
                    ))
                    avg_loss = 0
                overall_step += 1

            # 一轮结束后, 检验是否需要保存模型
            if len(epoch_losses) == 0 or epoch_loss / self.train_data.steps < min(epoch_losses):
                self.save(epoch + 1)
            epoch_losses.append(epoch_loss)

        return epoch_loss
    
    def save(self, epoch):
        print("Epoch {:03d} model saved on {}".format(epoch, "./pretrained/"))
        self.model.save_pretrained("./pretrained/") 





    


