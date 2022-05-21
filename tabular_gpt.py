from tracemalloc import start
from tqdm import tqdm
import torch
from torch import nn, relu
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class TabGPT(nn.Module):
    def __init__(self, model_name, tokenizer_name, num_loss_weight):
        super().__init__()
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
        self.gpt2 = GPT2LMHeadModel.from_pretrained(model_name)
        self.gpt2.parallelize()

        self.gpt2_tokenizer.pad_token = self.gpt2_tokenizer.eos_token
        self.lm_loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.l2_loss = nn.MSELoss()
        self.num_loss_weight = num_loss_weight
        
        num_added_toks = self.gpt2_tokenizer.add_tokens(['[NUM]'], special_tokens=True)
        self.gpt2.resize_token_embeddings(len(self.gpt2_tokenizer))
        self.num_token_id = self.gpt2_tokenizer('[NUM]', return_tensors='pt').input_ids.squeeze().item()

        self.num_regression = nn.Sequential(
            nn.Linear(1024, 400),
            nn.ReLU(),
            nn.Linear(400,1)
        ).to('cuda:3')

    def forward(self, text_input, numbers):
        tokenized_text = self.gpt2_tokenizer(text_input, truncation=True, padding=True, max_length=1024, return_tensors='pt').input_ids.to('cuda:0')
        model_output = self.gpt2(tokenized_text, output_hidden_states=True)
        num_indices = torch.eq(tokenized_text, self.num_token_id)

        lm_loss = self.lm_loss(model_output, tokenized_text, num_indices)      
        num_loss = self.numeric_loss(model_output, numbers, num_indices)

        ovr_loss = lm_loss + self.num_loss_weight * num_loss.to(f'cuda:{lm_loss.get_device()}')
        #print('lm loss', lm_loss)
        #print('num loss', num_loss)
        #print('ovr loss', ovr_loss)

        return ovr_loss, lm_loss, num_loss
        

    def lm_loss(self, model_output, targets, num_indices):
        logits = model_output.logits.permute(0,2,1)
        loss = self.lm_loss_fn(logits, targets)
        #print('loss', loss)
        #print(num_indices)
        loss = torch.where(~num_indices, loss, torch.tensor(0).float().to('cuda:0')) # this applies language modeling loss of zero to places where we have [NUM] tokens

        return torch.mean(loss)

    def numeric_loss(self, model_output, numbers, num_indices):
        numeric_value_indices = (num_indices.flatten() == True).nonzero(as_tuple=False)
        representations = torch.flatten(model_output.hidden_states[-1], start_dim=0, end_dim=1)
        #print(representations.shape)
        numeric_representations = torch.index_select(representations, 0, numeric_value_indices.flatten().to(f'cuda:{representations.get_device()}')).to(f'cuda:3')
        #print(numeric_representations.shape)
        predictions = self.num_regression(numeric_representations)
        #print(numbers)
        loss = self.l2_loss(predictions, numbers[0].to(f'cuda:{predictions.get_device()}').float())

        return loss
            

