# app.py
import torch
from transformers import BertTokenizer
from model import load_trained_model

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

        outputs = model(ids, mask, token_type_ids)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]
        predicted = (probs >= threshold).astype(int)

        return predicted, probs

# Load model and tokenizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = load_trained_model("bert_toxic.pt", device=device)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Example usage
text = "I hate you so much"
prediction, probabilities = predict(text, model, tokenizer)
print("Prediction:", prediction)
print("Probabilities:", probabilities)
