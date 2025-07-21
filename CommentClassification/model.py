import torch
import transformers

class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 6)  # or 7 based on your dataset

    def forward(self, ids, mask, token_type_ids):
        output = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        pooled_output = output.pooler_output
        output = self.l2(pooled_output)
        return self.l3(output)

def load_trained_model(model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = BERTClass()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model
