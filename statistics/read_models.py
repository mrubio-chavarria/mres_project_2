#!/home/mario/anaconda3/envs/project2_venv/bin python

"""
DESCRIPTION:
This file implements scripts to load the Pytorch 
models.
"""

import os
import torch
import pandas as pd
import numpy as np

# Function
def extract_decoder_weights(route, device):
    model = torch.load(route, map_location=device)
    decoder_0_weight = model['decoder.model.0.weight']
    decoder_0_bias = model['decoder.model.0.bias']
    decoder_1_weight = model['decoder.model.1.weight']
    decoder_1_bias = model['decoder.model.1.bias']
    decoder_weight = decoder_1_weight @  decoder_0_weight
    decoder_bias = decoder_1_weight @  decoder_0_bias + decoder_1_bias
    return decoder_weight, decoder_bias

if __name__ == "__main__":

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # 3xr6
    # Dataset
    database_dir = "/home/mario/Documentos/Imperial/Project_2/output_experiments/chiron_basic/3xr6"

    # Obtain model routes
    folders = [database_dir + '/' + folder for folder in os.listdir(database_dir)]
    decoder_data = []
    for folder in folders:
        model_route = [folder + '/' + file for file in os.listdir(folder) if file.endswith('pt')][0]
        decoder_weight, decoder_bias = extract_decoder_weights(model_route, device)
        decoder_data.append((model_route.split('/')[-1], decoder_weight.numpy(), decoder_bias.numpy()))

    # Format data
    columns = [0, 1, 2, 3, 4]
    index = 0
    store_dir = "/home/mario/Projects/project_2/statistics/models/3xr6"
    for model, weights, bias in decoder_data:
        fileroute = store_dir + '/' + 'weights_' + model[:-3] + '.tsv'
        weights = np.transpose(weights)
        pd.DataFrame(data=weights, columns=columns).to_csv(fileroute, sep='\t')
        bias = np.transpose(bias).reshape((1, -1))
        fileroute = store_dir + '/' + 'bias_' + model[:-3] + '.tsv'
        pd.DataFrame(data=bias, columns=columns).to_csv(fileroute, sep='\t')
        index += 1

    # ap
    # Dataset
    database_dir = "/home/mario/Documentos/Imperial/Project_2/output_experiments/chiron_basic/ap"

    # Obtain model routes
    folders = [database_dir + '/' + folder for folder in os.listdir(database_dir)]
    decoder_data = []
    for folder in folders:
        model_route = [folder + '/' + file for file in os.listdir(folder) if file.endswith('pt')][0]
        decoder_weight, decoder_bias = extract_decoder_weights(model_route, device)
        decoder_data.append((model_route.split('/')[-1], decoder_weight.numpy(), decoder_bias.numpy()))

    # Format data
    columns = [0, 1, 2, 3, 4]
    index = 0
    store_dir = "/home/mario/Projects/project_2/statistics/models/ap"
    for model, weights, bias in decoder_data:
        fileroute = store_dir + '/' + 'weights_' + model[:-3] + '.tsv'
        weights = np.transpose(weights)
        pd.DataFrame(data=weights, columns=columns).to_csv(fileroute, sep='\t')
        bias = np.transpose(bias).reshape((1, -1))
        fileroute = store_dir + '/' + 'bias_' + model[:-3] + '.tsv'
        pd.DataFrame(data=bias, columns=columns).to_csv(fileroute, sep='\t')
        index += 1

    # Random
    model_route = "/home/mario/Documentos/Imperial/Project_2/output_experiments/chiron_basic/random_initial.pt"
    weights, bias = extract_decoder_weights(model_route, device)
    weights = np.transpose(weights.numpy())
    bias = np.transpose(bias.numpy()).reshape((1, -1))
    fileroute = "/home/mario/Projects/project_2/statistics/models/random_weights.tsv"
    pd.DataFrame(data=weights, columns=columns).to_csv(fileroute, sep='\t')
    fileroute = "/home/mario/Projects/project_2/statistics/models/random_bias.tsv"
    pd.DataFrame(data=bias, columns=columns).to_csv(fileroute, sep='\t')
    print()
    
