# -*- coding:utf-8 -*-
import sys
import torch
from tensorboardX import SummaryWriter
import pickle

writer = SummaryWriter()
with open(sys.argv[1], "rb") as f:
    idolvecs = pickle.load(f)
labels = idolvecs[0]
weights = idolvecs[1]

writer.add_embedding(torch.FloatTensor(weights), metadata=labels)
