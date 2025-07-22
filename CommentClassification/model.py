import pandas as pd
import numpy as np
from sklearn import metrics
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig

file_path = "dataset/youtoxic_english_1000.csv"

df = pd.read_csv(
    file_path,
    encoding='utf-8',
    on_bad_lines='skip'  # In case of any malformed rows
)
df=df.drop(['CommentId','VideoId','IsRadicalism','IsHomophobic','IsNationalist','IsSexist','IsReligiousHate','comment_length','IsThreat'],axis=1, errors='ignore')

df['Text'] = df['Text'].str.lower()
df['Text'] = df['Text'].str.replace(r'[^a-zA-Z\s]','',regex=True)

df['list'] = df[df.columns[1:]].values.tolist()


def manual_convert(val):
    result = []
    for item in val:
        if item is True:
            result.append(1)
        elif item is False:
            result.append(0)
        else:
            result.append(0)  # fallback in case of unexpected values
    return result

df['list'] = df['list'].apply(manual_convert)
new_df = df[['Text','list']]


max_len= 200
train_batch_size = 8
valid_batch_size = 4
epochs =9
learning_rate = 1e-05
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer=tokenizer
        self.data = dataframe
        self.comment_text = dataframe['Text']
        self.targets = dataframe['list']
        self.max_len = max_len
    def __len__(self):
        return len(self.comment_text)
    def __getitem__(self, index):
        comment_text = str(self.comment_text[index])
        comment_text = " ".join(comment_text.split())
        
        inputs =  self.tokenizer.encode_plus(
            comment_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
        )
        
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids= inputs['token_type_ids']
        return {
            'ids': torch.tensor(ids),
            'mask': torch.tensor(mask),
            'token_type_ids': torch.tensor(token_type_ids),
            'targets': torch.tensor(self.targets[index])}

train_size = 0.8
train_dataset=new_df.sample(frac=train_size,random_state=200).reset_index(drop=True)
test_dataset=new_df.drop(train_dataset.index).reset_index(drop=True)


training_set = CustomDataset(train_dataset,tokenizer,max_len)
testing_set = CustomDataset(test_dataset,tokenizer,max_len)

train_params = {
    'batch_size': train_batch_size,
    'shuffle': True
}
test_params = {
    'batch_size': valid_batch_size,
    'shuffle': True
}
training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)





class BERTClass(torch.nn.Module):
  def __init__(self):
    super(BERTClass,self).__init__()
    self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased')
    self.l2 = torch.nn.Dropout(0.3)
    self.l3 = torch.nn.Linear(768, 6)
  def forward(self,ids,mask,token_type_ids):
    output_1= self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids)
    pooled_output = output_1.pooler_output  # [batch_size, hidden_size]
    dropout_output = self.l2(pooled_output)
    output = self.l3(dropout_output)  # No softmax, use BCEWithLogitsLoss
    return output


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model=BERTClass()
model.to(device)
print(torch.__version__)
 
def loss_fn(outputs,targets):
  return torch.nn.BCEWithLogitsLoss()(outputs,targets)

optimizer = torch.optim.Adam(params =  model.parameters(), lr=learning_rate)



def train(epoch):
    model.train()
    for _,data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.float)

        outputs = model(ids, mask, token_type_ids)

        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        if _%5000==0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
for epoch in range(epochs):
    train(epoch)
torch.save(model.state_dict(), 'bert_toxic_classifier.pt')
    
# def predict(text, model, tokenizer, max_len=200, threshold=0.5, device='cuda' if torch.cuda.is_available() else 'cpu'):
#     model.eval()
#     with torch.no_grad():
#         inputs = tokenizer.encode_plus(
#             text,
#             None,
#             add_special_tokens=True,
#             max_length=max_len,
#             padding='max_length',
#             truncation=True,
#             return_token_type_ids=True,
#             return_attention_mask=True,
#             return_tensors='pt'
#         )

#         ids = inputs['input_ids'].to(device)
#         mask = inputs['attention_mask'].to(device)
#         token_type_ids = inputs['token_type_ids'].to(device)

#         outputs = model(ids, mask, token_type_ids)  # shape: (1, 6) or (1, 7)
#         probs = torch.sigmoid(outputs).cpu().numpy()[0]
#         predicted = (probs >= threshold).astype(int)

#         return predicted, probs
# labels = ['IsToxic', 'IsAbusive', 'IsProvocative', 'IsObscene', 'IsHatespeech', 'IsRacist']  # use only 6 if your model outputs 6

# text = "Great work done"
# pred, prob = predict(text, model, tokenizer)

# for label, p, prob_score in zip(labels, pred, prob):
#     print(f"{label}: {p} (Confidence: {prob_score:.2f})")