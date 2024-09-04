from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import json
import os
import torch
from PIL import Image
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
from transformers import BertTokenizer, T5ForConditionalGeneration, T5Config
from generate_porms import geneate_poem_from_img


app = Flask(__name__)

UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

clip_model = ChineseCLIPModel.from_pretrained("./Chinese_CLIP")
clip_processor = ChineseCLIPProcessor.from_pretrained("./Chinese_CLIP")
config = T5Config.from_json_file("./pretrained/config.json")
model = T5ForConditionalGeneration(config)
model.load_state_dict(torch.load("./pretrained/model_epoch_046.pth", map_location=torch.device('cpu')))
model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

tokenizer = BertTokenizer(vocab_file=os.path.join("./", "vocab.txt"), eos_token="[EOS]")

generator = geneate_poem_from_img(
    tokenizer=tokenizer,
    LanguageModel=model,
    clip_model=clip_model,
    clip_processor=clip_processor,
    keyword_path="./datasets/keywords.txt",
    keyword_dict_path="./datasets/keywords_dict.pt",
    top_k=8
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # 生成诗句
        candidates = generator.generate_poem(file_path, 1)
        candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
        print(candidates)
        result = {"poem": candidates[0][0], "similarity": candidates[0][1]}
        # print(candidates[0][0])
        # print(json.dumps(result, ensure_ascii=False))

        # else:
        #result = {"poem": "没有生成诗句", "similarity": 0}

        return jsonify(result)

if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
