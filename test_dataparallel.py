
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sys


class Model(nn.Module):
    # Our model

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("\tIn Model: input size", input.size(),
              "output size", output.size())

        return output


class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


if __name__ == "__main__":

    print(torch.cuda.device_count())
    print(torch.cuda.current_device())

    print(type(sys.argv[1]))
    print(sys.argv[1])

    print(type(sys.argv))
    print(sys.argv)
    print('PREVIO')
    print('---------------------------')
    print('SCRIPT')

    # Read available devices
    available_gpus = sorted([int(device_id) for device_id in sys.argv[1].split(',')])

    device = torch.device(f"cuda:{available_gpus[0]}")
    
    # Parameters and DataLoaders
    input_size = 5
    output_size = 2
    batch_size = 30
    data_size = 100

    rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                         batch_size=batch_size, shuffle=True)
    
    model = Model(input_size, output_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model, device_ids=available_gpus)
    else:
        print('No GPU detected')

    model.to(device)

    for data in rand_loader:
        input = data.to(device)
        output = model(input)
        print("Outside: input size", input.size(),
            "output_size", output.size())
    
    # Save the model
    # # When locally
    # path = "/home/mario/Projects/project_2/saved_models/model.pickle"
    # When HPC
    path = "/rds/general/user/mr820/home/project_2/saved_models/model.pickle"
    torch.save(model.module.state_dict(), path)