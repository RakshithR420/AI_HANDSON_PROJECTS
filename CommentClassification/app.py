import torch 
from transformers import BertModel
from transformers import BertTokenizer
import transformers
from flask import Flask, request, jsonify
from flask_cors import CORS
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.l2 =torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 6)
    def forward(self,ids,mask,token_type_ids):
        output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        pooled_output = output_1.pooler_output
        dropout_output = self.l2(pooled_output)
        output = self.l3(dropout_output)
        return output

model = BERTClass()

model.load_state_dict(torch.load('bert_toxic_classifier.pt',map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def predict(text, model, tokenizer, max_len=200, threshold=0.5, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        ids = inputs['input_ids'].to(device)
        mask = inputs['attention_mask'].to(device)
        token_type_ids = inputs['token_type_ids'].to(device)

        outputs = model(ids, mask, token_type_ids)  # shape: (1, 6) or (1, 7)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]
        predicted = (probs >= threshold).astype(int)

        return predicted, probs

# labels = ['IsToxic', 'IsAbusive', 'IsProvocative', 'IsObscene', 'IsHatespeech', 'IsRacist']
# text = " good work done"
# pred, prob = predict(text, model, tokenizer)

# for label, p, prob_score in zip(labels, pred, prob):
#     print(f"{label}: {p} (Confidence: {prob_score:.2f})")
    
    

app = Flask(__name__)
CORS(app)

@app.route("/classify", methods = ["POST"])
def predict_comment():
    comment = request.json["comment"]
    labels = ['IsToxic', 'IsAbusive', 'IsProvocative', 'IsObscene', 'IsHatespeech', 'IsRacist']  # use only 6 if your model outputs 6
    pred, prob = predict(comment, model, tokenizer)

    response = {
        "prediction": [
            {
                "label": label, "value": int(p), "confidence": round(float(score),2)
            }
            for label,p,score in zip(labels,pred,prob)
        ]
    }
    # for label,p,prob_score in zip(labels,pred,prob):
    #     print(f"{label}: {p} (Confidence: {prob_score:.2f})")
    return jsonify(response) 
        


if __name__ =="__main__":
    app.run(debug=True, port=5000)