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


vgg_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512]
base_net = Sequential(*vgg(vgg_config))

# print(base_net)

# x = torch.Tensor(32,3,150,150)

# output = base_net(x)
# print(output.shape)

# for layer in base_net:
#     x = layer(x)
#     print(x.size())


# model = ConvLSTM(input_size=(height, width),
#                  input_dim=channels,
#                  hidden_dim=[64, 64, 128],
#                  kernel_size=(3, 3),
#                  num_layers=3,
#                  batch_first=True
#                  bias=True,
#                  return_all_layers=False)


class EncoderCNN(nn.Module):
	def __init__(self, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, CNN_embed_dim=300):
		"""Load the pretrained ResNet-152 and replace top fc layer."""
		super(EncoderCNN, self).__init__()

		# self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
		# self.drop_p = drop_p

		#resnet = models.resnet152(pretrained=False)
		#modules = list(resnet.children())[:-1]      # delete the last fc layer.
		#self.resnet = nn.Sequential(*modules)
		self.vgg = base_net
		# self.fc1 = nn.Linear(1024*9*9, fc_hidden1)
		# self.bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.01)
		# self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
		# self.bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.01)    
		# self.fc3 = nn.Linear(fc_hidden2, CNN_embed_dim)

	  
	def forward(self, x_5d):
		cnn_seq_23 = []
		cnn_seq_final= []

		for t in range(x_5d.size(1)):
			# ResNet CNN
			#with torch.no_grad():
			#x_final = self.vgg(x_3d[:, t, :, :, :])        # ResNet
			# x = x.view(x.size(0), -1)             # flatten output of conv
			#x = self.vgg[:end_layer](x_3d[:, t, :, :, :])   

			x = x_5d[:, t, :, :, :]
			for i,layer in enumerate(self.vgg):
				x = layer(x)
				if i == 22:
					x_23 = x
				if i == len(self.vgg) - 1:
					x_final = x

				# # FC layers
				# x = self.bn1(self.fc1(x))
				# x = F.relu(x)
				# x = self.bn2(self.fc2(x))
				# x = F.relu(x)
				# x = F.dropout(x, p=self.drop_p, training=self.training)
				# x = self.fc3(x)

			cnn_seq_23.append(x_23)
			cnn_seq_final.append(x_final)

			# swap time and sample dim such that (sample dim, time dim, CNN latent dim)
		cnn_seq_23 = torch.stack(cnn_seq_23, dim=0)
		cnn_seq_final = torch.stack(cnn_seq_final, dim=0)

			# cnn_embed_seq: shape=(batch, time_step, input_size

		return cnn_seq_23, cnn_seq_final


class DecoderRNN(nn.Module):
	def __init__(self, CNN_embed_dim=300, h_RNN_layers=3, h_RNN=256, h_FC_dim=128, drop_p=0.3, num_classes=2):
		super(DecoderRNN, self).__init__()

		# self.RNN_input_size = CNN_embed_dim
		# self.h_RNN_layers = h_RNN_layers   # RNN hidden layers
		# self.h_RNN = h_RNN                 # RNN hidden nodes
		# self.h_FC_dim = h_FC_dim
		# self.drop_p = drop_p
		# self.num_classes = num_classes


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

		# self.LSTM = nn.LSTM(
		#     input_size=self.RNN_input_size,
		#     hidden_size=self.h_RNN,        
		#     num_layers=h_RNN_layers,       
		#            # input & output will has batch size as 1s dimension. e.g. (time_step, batch, time_step input_size)
		# )

		# self.fc1 = nn.Linear(self.h_RNN, self.h_FC_dim)
		# self.fc2 = nn.Linear(self.h_FC_dim, self.num_classes)

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