from torch import optim
from torch import nn
import torch
from torch.utils import data
from tabular_gpt import TabGPT
from tabular_dataset import TableDataset
import argparse
import pandas as pd
from tqdm import tqdm


def run(file, columns, data_types, prompt, epochs, model_name, tokenizer_name, save_path, num_loss_weight, learning_rate, batch_size):
    df = pd.read_csv(file)
    column_type_dict = dict()
    for name, type in zip(columns, data_types):
        column_type_dict[name] = type

    dataset = TableDataset(df, columns, column_type_dict)
    data_loader = data.DataLoader(dataset, shuffle=True, batch_size=batch_size)
    model = TabGPT(model_name, tokenizer_name, num_loss_weight)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in tqdm(range(epochs)):
        print(f'starting epch {epoch}')
        epoch_loss_overall = 0
        epoch_loss_lm = 0
        epoch_loss_num = 0
        for text, numbers in data_loader:
            overall_loss, lm_loss, num_loss = model(text, numbers)
            
            overall_loss.backward()
            optimizer.step()

            epoch_loss_overall += overall_loss.item()
            epoch_loss_lm += lm_loss.item()
            epoch_loss_num += num_loss.item()
        
        torch.save(model.state_dict(), f'{save_path}_epoch_{epoch}')
        print('epoch loss overall', epoch_loss_lm)
        print('epoch loss lm', epoch_loss_lm)
        print('epoch loss num', epoch_loss_num)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type = str, default='pet_dataset.csv')
    parser.add_argument('--columns', type=str, nargs='+')
    parser.add_argument('--prompt', type=str)
    parser.add_argument('--data-types', type=str)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--model', type=str, default='gpt2-xl')
    parser.add_argument('--tokenizer', type=str, default='gpt2-xl')
    parser.add_argument('--model-save-path', type=str, default='table_model')
    parser.add_argument('--num-loss-weight')
    parser.add_argument('--learning-rate')
    parser.add_argument('--batch-size')



    args = parser.parse_args()

    run(
        file = args.file, 
        columns = args.columns,
        data_types = args.data_types,
        prompt = args.prompt,
        epochs = args.epochs,
        model = args.model,
        tokenizer = args.tokenizer,
        save_path = args.model_save_path,
        num_loss_weight = args.num_loss_weight,
        learning_rate = args.learning_rate,
        batch_size = args.batch_size
        )
