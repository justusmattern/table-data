from torch import optim
from torch import nn
import torch
from torch.utils import data
from tabular_gpt import TabGPT
import argparse
import pandas as pd
from tqdm import tqdm


def run(model_name, prompt, num_generations, column_names, column_types):
    model = TabGPT('gpt2-large', 'gpt2-large', 0)
    model.load_state_dict(torch.load(model_name))

    generated = []
    for i in tqdm(range(num_generations)):
        generated.append(model.sample(prompt, column_names, column_types))

    print('generation done')
    print(generated)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type = str)
    parser.add_argument('--prompt', type = str)
    parser.add_argument('--num-generations', type=int)
    parser.add_argument('--column-names', type=str, nargs='+')
    parser.add_argument('--column-types', type=str, nargs='+')

    args = parser.parse_args()

    run(
        model_name=args.model_name,
        prompt=args.prompt,
        num_generations=args.num_generations,
        column_names=args.column_names,
        column_types=args.column_types
    )
