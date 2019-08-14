import torch
from torch.nn import Conv2d, Sequential, ModuleList, ReLU, BatchNorm2d
from vision.nn.vgg import vgg

from vision.ssd.ssd import SSD
from vision.ssd.predictor import Predictor
from vision.ssd.config import vgg_ssd_config as config
from vision.ssd.vgg_ssd import create_vgg_ssd

import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from vision.nn.conv_lstm import ConvLSTM


class DecoderRNN(nn.Module):
	def __init__(self, CNN_embed_dim=300, h_RNN_layers=3, h_RNN=256, h_FC_dim=128, drop_p=0.3, num_classes=2):
		super(DecoderRNN, self).__init__()

	
		self.ConvLSTM_23 = ConvLSTM(input_size=(38, 38),
								input_dim=512,
								hidden_dim=[64, 64, 512],
								kernel_size=(3, 3),
								num_layers=3,
								batch_first=True,
								bias=True,
								return_all_layers=False)

		self.ConvLSTM_final = ConvLSTM(input_size=(19, 19),
								input_dim=1024,
								hidden_dim=[64, 64, 1024],
								kernel_size=(3, 3),
								num_layers=3,
								batch_first=True,
								bias=True,
								return_all_layers=False)

	def forward(self, x_RNN):
    		
		x_RNN_23 , x_RNN_final = x_RNN[0], x_RNN[1]
		
		#self.ConvLSTM.flatten_parameters()
		RNN_out_23, _ = self.ConvLSTM_23(x_RNN_23, None)  
		""" h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """ 
		""" None represents zero initial hidden state. RNN_out has shape=(time_step, batch, output_size) """

		#self.ConvLSTM.flatten_parameters()
		RNN_out_final, _ = self.ConvLSTM_final(x_RNN_final, None)  
		""" h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """ 
		""" None represents zero initial hidden state. RNN_out has shape=(time_step, batch, output_size) """
	

		# FC layers
		#x = self.fc1(RNN_out[-1, : , :])   # choose RNN_out at the last time step
		# print(x_RNN.shape)
		# print(RNN_out[0].shape)
		x_23 = F.relu(RNN_out_23[0])
		x_final = F.relu(RNN_out_final[0])
		# x = F.dropout(x, p=self.drop_p, training=self.training)
		# x = self.fc2(x)

		return x_23, x_final



if __name__ == "__main__":
    
    encoder = EncoderCNN()
    decoder =  DecoderRNN()

    x = torch.Tensor(32,2,3,300,300)

    out_enc_23, out_enc_final = encoder(x)



    out_dec_23, out_dec_final  = decoder([out_enc_23, out_enc_final])



    print(out_dec_23.size())

    print(out_dec_final.size())



    # net = create_vgg_ssd("21")




    # confidence, locations = net(images)
    # regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)  # TODO CHANGE BOXES
    # loss = regression_loss + classification_loss



    # encoder = EncoderCNN()
    # decoder =  DecoderRNN()

    # x = torch.Tensor(32,2,3,300,300)

    # out_enc_23, out_enc_final = encoder(x)



    # print(out_enc_23.size())

    # print(out_enc_final.size())