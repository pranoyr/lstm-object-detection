import torch.nn as nn
import torch
import numpy as np
from typing import List, Tuple
import torch.nn.functional as F
from vision.nn.conv_lstm import ConvLSTM


from collections import namedtuple
from torch.nn import ModuleList

GraphPath = namedtuple("GraphPath", ['s0', 'name', 's1'])  #

LSTM_list = ModuleList([ConvLSTM(input_size=(38, 38),
								input_dim=512,
								hidden_dim=[64, 64, 512],
								kernel_size=(3, 3),
								num_layers=3,
								batch_first=True,
								bias=True,
								return_all_layers=False),
                                
                            ConvLSTM(input_size=(19, 19),
								input_dim=1024,
								hidden_dim=[64, 64, 1024],
								kernel_size=(3, 3),
								num_layers=3,
								batch_first=True,
								bias=True,
								return_all_layers=False),

                            ConvLSTM(input_size=(10, 10),
								input_dim=512,
								hidden_dim=[64, 64, 512],
								kernel_size=(3, 3),
								num_layers=3,
								batch_first=True,
								bias=True,
								return_all_layers=False),

                            ConvLSTM(input_size=(5, 5),
								input_dim=256,
								hidden_dim=[64, 64, 256],
								kernel_size=(3, 3),
								num_layers=3,
								batch_first=True,
								bias=True,
								return_all_layers=False),

                            ConvLSTM(input_size=(3, 3),
								input_dim=256,
								hidden_dim=[64, 64, 256],
								kernel_size=(3, 3),
								num_layers=3,
								batch_first=True,
								bias=True,
								return_all_layers=False),

                            ConvLSTM(input_size=(1, 1),
								input_dim=256,
								hidden_dim=[64, 64, 256],
								kernel_size=(3, 3),
								num_layers=3,
								batch_first=True,
								bias=True,
								return_all_layers=False)])

print(LSTM_list[0])