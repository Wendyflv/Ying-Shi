from transformers import ChineseCLIPProcessor, ChineseCLIPModel
from PIL import Image
import torch



# 加载模型
model = ChineseCLIPModel.from_pretrained("./Chinese_CLIP")
processor = ChineseCLIPProcessor.from_pretrained("./Chinese_CLIP")

# 对图像编码
imag = Image.open('D:/NLP/CLIPForPoems/Image2Poem/datasets/images/chun.jpg')
inputs = processor(images=imag, return_tensors="pt")
image_features = model.get_image_features(**inputs)
print(image_features.shape)

# 对关键词编码
key_words = ['余晖', '樱花', '春色','晚霞', '夏日', '沙漠', '旅人']
text_features = []
for keyword in key_words:
    feature = processor(text=keyword, return_tensors="pt")
    text_features.append(model.get_text_features(**feature))
    

def cosine_similarity(x, y):
    return torch.sum(x * y) / (torch.sqrt(torch.sum(pow(x, 2))) * torch.sqrt(torch.sum(pow(y, 2))))


# 将图片和关键词编码做余弦相似度计算
i = 0
for text_feature in text_features:
    similar = cosine_similarity(text_feature, image_features)
    print(key_words[i], similar)
    i +=1