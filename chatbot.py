from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from flask import Flask,request,render_template
from flask_cors import CORS
import json

#Model using Facebook's Blenderbot
model_name = "facebook/blenderbot-400M-distill"

#Initialization of the model:
tokenizer=AutoTokenizer.from_pretrained(model_name)
model=AutoModelForSeq2SeqLM.from_pretrained(model_name)
app=Flask(__name__)
CORS(app)
conversation_history=[]
@app.route('/')
def home():
    return render_template("index.html")
@app.route('/chatbot',methods=["POST"])
def handle_prompt():
    data=json.loads(request.get_data(as_text=True))
    InputText=data["prompt"]
    historyText="\n".join(conversation_history)
    fullInput=historyText+"\n"+InputText
    #Tokenization:
    Input=tokenizer.encode_plus(fullInput,return_tensors="pt")

    #Ouput Generation tokenised:
    output=model.generate(**Input)

    #Output Detokenization:
    response=tokenizer.decode(output[0],skip_special_tokens=True).strip()

    #Updating History:
    conversation_history.append(InputText)
    conversation_history.append(response)

    return response


if __name__=="__main__":
    app.run()