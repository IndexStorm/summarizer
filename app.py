import json

from flask import Flask, request
from flask_cors import CORS, cross_origin
from prometheus_flask_exporter import PrometheusMetrics
from transformers import pipeline
from transformers import BertTokenizerFast, EncoderDecoderModel
import torch

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
# tokenizer = BertTokenizerFast.from_pretrained('mrm8488/bert-small2bert-small-finetuned-cnn_daily_mail-summarization')
# model = EncoderDecoderModel.from_pretrained('mrm8488/bert-small2bert-small-finetuned-cnn_daily_mail-summarization')

# def generate_summary(text):
#     # cut off at BERT max length 512
#     inputs = tokenizer([text], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
#     input_ids = inputs.input_ids
#     attention_mask = inputs.attention_mask
#     output = model.generate(input_ids, attention_mask=attention_mask)

#     return tokenizer.decode(output[0], skip_special_tokens=True)

metrics = PrometheusMetrics(app)


@app.route("/", methods=['GET'])
def welcome():
    return "Welcome to summarization closed beta 0.1"


@app.route("/summarize", methods=['POST'])
@cross_origin()
def similarity_route():
    request_data = request.get_json()
    text = request_data.get("text")
    if text is None or type(text) is not str:
        return "Wrong args", 400
    

    num_of_words = len(text.split())
    if num_of_words < 30:
        return "Less than 30 words", 400
    
    ### TODO: 
    # - split text + merge output

    if num_of_words > 1000:
        return "Only support 1000 words now", 400
    else:
        try:
            result = summarizer(text, max_length=max(130, num_of_words // 3) , min_length=30, do_sample=False)
            # result = generate_summary(text)
            # result = {'summary': result}
        except:
            return "Failed to summarize", 500

    response = app.response_class(
        response=json.dumps(result),
        status=200,
        mimetype='application/json'
    )
    return response


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3001, debug=True)
