import torch
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
from transformers import BertTokenizer, T5ForConditionalGeneration, T5Config
from transformers import GPT2LMHeadModel
from PIL import Image
from tqdm import tqdm
import os


def cosine_similarity(x, y):
    return torch.sum(x * y) / (torch.sqrt(torch.sum(pow(x, 2))) * torch.sqrt(torch.sum(pow(y, 2))))





class geneate_poem_from_img:
    def __init__(self, clip_model : ChineseCLIPModel, clip_processor :ChildProcessError, 
                 keyword_path, keyword_dict_path, top_k,
                 tokenizer : BertTokenizer,
                 LanguageModel):
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        self.keyword_path = keyword_path
        self.keyword_dict_path = keyword_dict_path
        self.top_k = top_k
        self.tokenizer = tokenizer
        self.LMModel = LanguageModel

        if self.keyword_dict_path is None:
            self.keyword_dict = self._create_keyword_dict()
        else:
            self.keyword_dict = torch.load("./datasets/keywords_dict.pt")

        


    # 构造关键词:关键词编码向量 的词典
    def _create_keyword_dict(self):
        keyword_name= []
        text_feature = []
        with open("./datasets/keywords.txt", "r", encoding="utf-8") as f:
            data = f.readlines()
            for line in tqdm(data, total = len(data)):
                keyword_name.append(line.strip())
                feature = clip_processor(text = line.strip(), return_tensors="pt")
                text_feature.append(clip_model.get_text_features(**feature))
        text_feature = torch(text_feature, dim=0)
        text_feature = text_feature / text_feature.norm(p=2, dim=-1, keepdim=True)
        keyword_dict = {keyword_name[i]:text_feature[i:i+1] for i in range(len(keyword_name))}
        torch.save(keyword_dict, "./datasets/keywords_dict1.pt")
        return keyword_dict




    def top_k_keywords(self, top_k, img_feature):
        text_features = torch.cat(list(self.keyword_dict.values()), dim=0)
        similarity = torch.tensor([
            cosine_similarity(text_features[i], img_feature) for i in range(text_features.shape[0])
        ])
        top_k = torch.topk(similarity, k = self.top_k)
        # 返回相似度最高的前K个的关键词的索引
        return top_k.indices.squeeze(0).tolist()

    def generte_tokens(self, image_path):
        # 把图像编码
        image = Image.open(image_path)
        image = self.clip_processor(images = image, return_tensors = "pt")
        img_feature = self.clip_model.get_image_features(**image)
        img_feature = img_feature / img_feature.norm(p=2, dim=-1, keepdim=True)

        top_keyword_index = self.top_k_keywords(top_k=self.top_k, img_feature=img_feature)
        top_keywords = [ list(self.keyword_dict.keys())[i] for i in top_keyword_index]

        prompt = "关键词："+ " ".join(top_keywords) + " [EOS] "
        #print("prompt",prompt)

        input_ids = self.tokenizer.encode(prompt)
        #print("input_ids",input_ids)
        for _ in range(4):
            inputs = {"input_ids" : torch.tensor([input_ids])}
            outputs = self.LMModel.generate(**inputs, max_length=8, num_beams=5,
                                            no_repeat_ngram_size=1, early_stopping=True, do_sample=True)
            print(outputs)
            
            input_ids = input_ids [:-1] + outputs.squeeze(0)[1:-1].tolist() + input_ids[-1: ]

        return top_keywords, input_ids
    
    
    def generate_poem(self, image_path, epochs=10):
        
        def check_each_poem(path):
            keys, output_ids = self.generte_tokens(path)
            output = self.tokenizer.decode(output_ids)
            #print(output)
            output = output.replace(" ", "").replace("[CLS]", "").replace("[SEP]", "")
            poet = output.split("[EOS]")[1:-1]
            #print(poet)
            # if len(set([len(sentence) for sentence in poet])) > 1:
            #     return "", 0
            poet = "，".join(poet) + "。"
            

            # 检查诗与关键词的匹配性
            poet_feature = self.clip_processor(text=poet, return_tensors="pt")  # 构造诗的向量
            poet_feature = self.clip_model.get_text_features(**poet_feature)
            similarity = sum([cosine_similarity(self.keyword_dict[k], poet_feature) for k in keys]) / len(keys)
            print("poet: ",poet)
            print("similarity: ", similarity)
            return poet, similarity.item()
        
        candidate = [check_each_poem(image_path) for _ in tqdm(range(epochs), total=epochs)]
        return candidate
    
if __name__ == "__main__":

    # 加载模型
    clip_model = ChineseCLIPModel.from_pretrained("./Chinese_CLIP")
    clip_processor = ChineseCLIPProcessor.from_pretrained("./Chinese_CLIP")
    # 加载模型配置
    config = T5Config.from_json_file("./pretrained/config.json")

    # 初始化模型
    model = T5ForConditionalGeneration(config)

    # 加载状态字典
    model.load_state_dict(torch.load("./pretrained/model_epoch_041.pth" ,map_location=torch.device('cpu')))
    # 将模型移动到适当的设备
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    generator = geneate_poem_from_img(
        tokenizer= BertTokenizer(vocab_file= os.path.join("./", "vocab.txt"), eos_token="[EOS]"),
        LanguageModel= model,
        clip_model= clip_model,
        clip_processor= clip_processor,
        keyword_path= "./datasets/keywords.txt",
        keyword_dict_path= "./datasets/keywords_dict.pt",
        top_k= 8

    )

    candidates = generator.generate_poem("./examples/chun.jpg", 10)
    candidates = sorted(candidates, key=lambda x: x[1], reverse=True)     # 根据相似度高低排序
    for p, v in candidates[:10]:
        print("诗句: ", p, "\t 相似度评分: ", v)
    








