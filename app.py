import os
import sys
from src.textSummarizer.pipeline.prediction import PredictionPipeline
from flask import Flask,render_template,request,jsonify
from transformers import AutoTokenizer
from transformers import pipeline
from src.textSummarizer.config.configuration import ConfigurationManager


app = Flask(__name__)

class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()

    
@app.route("/")
def index():
    return render_template('index.html')

@app.route("/training", methods=['GET'])
async def training():
    try:
        os.system("python main.py") 
        return jsonify('training done')
    except Exception as e:
        return f"Error Occurred during training: {e}"

@app.route("/", methods=['POST','GET'])  
async def prediction():
    if request.method=="POST":
        text = request.form['textarea']
        #load
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        #prediction
        gen_kwargs = {"length_penalty": 0.8, "num_beams": 8, "max_length": 128}

        pipe = pipeline("summarization", model="google/flan-t5-small",tokenizer=tokenizer)

        output = pipe(text, **gen_kwargs)[0]["summary_text"]
        print(output)
    return render_template('index.html',text=text,output=output)
    
    
if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True,port=5000)
