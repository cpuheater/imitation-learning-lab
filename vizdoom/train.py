# https://github.com/facebookresearch/torchbeast/blob/master/torchbeast/core/environment.py

import numpy as np
from collections import deque
import gymnasium
import cv2
cv2.ocl.setUseOpenCL(False)
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import os
from torch.utils.tensorboard import SummaryWriter
import torch
import random
import tyro
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from model import Simple

@dataclass
class Args:
    env_id: str = "basic"
    """the id of the environment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """"""
    lr: float = 0.01
    """the learning rate of the optimizer"""
    rollout_steps: int = 10000
    """"""
    epochs: int = 30
    """num train epochs"""
    epochs: int = 50
    """num train epochs"""
    num_actions: int = 3
    """num actions"""
    data_file: str = "data/data.pt"
    """"""
    model_dir: str = "models"
    """"""


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MyDataset(Dataset):
    def __init__(self, data, transform=None):
        self.inputs, self.labels = data
        #self.labels = torch.LongTensor(labels)
        self.transform = transform

    def __getitem__(self, index):
        inputs, labels = self.inputs[index], self.labels[index]
        return inputs, labels

    def __len__(self):
        return len(self.inputs)

def train(writer, model, dataset, epochs=50, lr=0.01):
    running_loss = 0.0
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-5)
    dataloader = DataLoader(dataset, batch_size=64)
    running_loss_history = []
    for epoch in range(epochs):
        running_loss = 0.0
        for (input, labels) in dataloader:
            input = input.permute(0, 3, 2, 1)
            out =model(input)
            loss = F.binary_cross_entropy_with_logits(out, labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        running_loss = running_loss/len(dataloader)
        running_loss_history.append(running_loss)
        print(f"Epoch: {epoch}, Loss: {running_loss}")
        writer.add_scalar("losses/reg_loss", running_loss, epoch)

    return np.mean(running_loss_history)

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
    data = torch.load(args.data_file)
    dataset = MyDataset(data)
    model = Simple(3, args.num_actions)
    train(writer, model, dataset, args.epochs, args.lr)
    episode_rewards = []
    torch.save(model, f"{args.model_dir}/{run_name}_{args.epochs}.pt")
    print("Finished training")


