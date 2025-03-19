# https://github.com/facebookresearch/torchbeast/blob/master/torchbeast/core/environment.py

import numpy as np
import cv2
cv2.ocl.setUseOpenCL(False)
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import functools
import os
import re
from torch.utils.tensorboard import SummaryWriter
import torch
import random
import tyro
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from networks import Simple, ImpalaCNNSmall, get_network

@dataclass
class Args:
    env_id: str = "monsters"
    """the id of the environment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """"""
    lr: float = 0.001
    """the learning rate of the optimizer"""
    rollout_steps: int = 10000
    """"""
    epochs: int = 100
    """num train epochs"""
    num_actions: int = 5
    """num actions"""
    data_dir: str = "data"
    """"""
    model_dir: str = "models"
    """"""
    weight_decay: float = 0.00001
    """weight decay"""
    batch_size:int = 64
    """batch_size"""
    network_type: str = "ImpalaCNNLarge"
    """ImpalaCNNSmall ImpalaCNNLarge"""



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MyDataset(Dataset):
    def __init__(self, inputs, labels, transform=None):
        self.inputs, self.labels = inputs, labels
        #self.labels = torch.LongTensor(labels)
        self.transform = transform

    def __getitem__(self, index):
        inputs, labels = self.inputs[index], self.labels[index]
        return inputs, labels

    def __len__(self):
        return len(self.inputs)

def train(writer, model, dataset, epochs, lr, weight_decay):
    running_loss = 0.0
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-5, weight_decay=weight_decay)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    running_loss_history = []
    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        for (input, labels) in dataloader: 
            out =model(input.to(device))
            loss = F.binary_cross_entropy_with_logits(out, labels.to(device).float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        running_loss = running_loss/len(dataloader)
        running_loss_history.append(running_loss)
        print(f"Epoch: {epoch}, Loss: {running_loss}")
        writer.add_scalar("losses/reg_loss", running_loss, epoch)

    return np.mean(running_loss_history)

def load_data(model_dir, env_id):
    def do_reduce(accum, e):
        accum[0].append(e[0])
        accum[1].append(e[1])
        return accum
    pattern = re.compile(fr"{env_id}_.*\.pt")
    result = []
    for filename in os.listdir(model_dir):
        if pattern.match(filename):
            result.append(torch.load(os.path.join(model_dir, filename)))
    obs, actions = functools.reduce(do_reduce, result, ([], []))
    obs, actions = torch.cat(obs), torch.cat(actions)
    return obs, actions


if __name__ == "__main__":

    args = tyro.cli(Args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    isExist = os.path.exists(args.model_dir)
    if not isExist:
        os.makedirs(args.model_dir)

    exp_name = os.path.basename(__file__)[: -len(".py")]
    run_name = f"{exp_name}__{args.env_id}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    obs, actions = load_data(args.data_dir, args.env_id)
    print(f"Training data consist of {len(obs)} frames.")
    dataset = MyDataset(obs, actions)
    model = get_network(args.network_type, 3, args.num_actions).to(device)
    train(writer, model, dataset, args.epochs, args.lr, args.weight_decay)
    episode_rewards = []
    torch.save(model, f"{args.model_dir}/{args.env_id}_{args.network_type}_{args.epochs}.pt")
    print("Finished training")


