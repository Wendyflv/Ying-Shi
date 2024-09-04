import torch
from transformers import BertTokenizer
from tqdm import tqdm
import random
import json
from data_process.vocab import WordVocab
import os 

    


# text = []
# with open("./datasets/CCPC/ccpc_train_v1.0.json", "r", encoding="utf-8")  as f:
#     for line in f.readlines():
#         dict_line = json.loads(line)
#         text.append("关键词：" + dict_line["keywords"] + dict_line["content"])
#     vocab = WordVocab(texts=text)
#     vocab.save_vocab_txt(os.path.join("./", "vocab.txt"))

# tokenizer = BertTokenizer(vocab_file=os.path.join("./", "vocab.txt"), eos_token="[EOS]")


class DatatLoader:
    def __init__(self, batch_size, tokenizer, data_path):
        self.lines = []
        self.data_path = data_path
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self._read_data()
        random.shuffle(self.lines)
        self.steps = len(self.lines) // batch_size
        

    def __len__(self):
        return self.steps   

    def __getitem__(self,item):
        # 取 [item * bs: (item + 1) * bs ]的文本数据
        data = self.lines[item * self.batch_size : (item + 1) * self.batch_size]
        input_len = max([ len( s[0].replace(" ","").replace("[EOS]", "")) for s  in data]) + 6
        output_len = max([ len(s[1].replace(" ","").replace("[EOS]", "")) for s in data]) +3
        

        tokens, targets, attens_x, attens_y = [], [], [], []

        for i in range(self.batch_size):
            # 处理输入
            # 对不满足最长长度的文本后面补上PAD
            x = self.tokenizer.encode(data[i][0])
            # 有效字符为1
            atten_x = [1 for _ in range(len(x))]
            # 填充部分为0
            atten_x += [0 for _ in range(input_len - len(x))]
            x.extend([self.tokenizer.pad_token_id for _ in range(input_len - len(x))])

            # 处理输出
            y = self.tokenizer.encode(data[i][1])
            atten_y =[1 for _ in range(len(y))]
            atten_y += [0 for _ in range(output_len - len(y))]
            y.extend([self.tokenizer.pad_token_id for _ in range(output_len - len(y))])


            tokens.append(x)
            targets.append(y)
            attens_x.append(atten_x)
            attens_y.append(atten_y)
        tokens, targets = torch.tensor(tokens, dtype=torch.int64), torch.tensor(targets, dtype=torch.int64)
        attens_x, attens_y = torch.tensor(atten_x, dtype=torch.int64), torch.tensor(atten_y, dtype=torch.int64)

        return tokens, targets, attens_x, attens_y   

    def _read_data(self):
        # 读取数据
        with open(self.data_path, "r", encoding="utf-8") as f:
            self.lines = []
            total_lines = sum(1 for _ in open(self.data_path, "r",encoding="utf-8"))
            for line in tqdm(f, desc="Loading Data", total=total_lines):
                json_line = json.loads(line)
                keywords=  json_line["keywords"].strip()
                poems = json_line["content"].strip().split("|")
                for i in range(len(poems)):
                    x = "关键词："+ keywords + " [EOS] " 
                    line = (x + " [EOS] ".join(poems[:i]) + " [EOS] ", poems[i] + " [EOS] ")
                    
                    self.lines.append(line)
                    
                

if __name__ == '__main__':

    


    batch_size = 32
    tokenizer = BertTokenizer(vocab_file=os.path.join("./", "vocab.txt"), eos_token="[EOS]")
    dataloader = DatatLoader(batch_size= batch_size, tokenizer=tokenizer, data_path="./datasets/CCPC/ccpc_train_v1.0.json")
    for batch in dataloader:
        tokens, targets, attns_x, attns_y = batch
        print("Tokens:", tokens)
        print("Targets:", targets)
        print("Attention X:", attns_x)
        print("Attention Y:", attns_y)
        break  # Exit after printing the first batch


    


       





        






    
