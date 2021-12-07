import torch.nn as nn
import torch
from models.basics import Generator
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--num_rblock', type=int, default=16)
args = parser.parse_args()

device = torch.device('cuda:0')

input = torch.rand(1,3,24,24)
input = input.to(device)
gen = Generator(args)
gen = gen.to(device)


output = gen(input)
print(output.shape)


del gen