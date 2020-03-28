import torch.nn as nn
import torch
import numpy as np
from typing import List, Tuple
import torch.nn.functional as F
# from ..utils import box_utils
import box_utils
from torch.nn import Conv2d, Sequential, ModuleList, ReLU, BatchNorm2d
from conv_lstm import ConvLSTMCell
from conv_lstm import BottleNeckLSTM
import mobilenetv1_ssd_config as config
from torchvision.models import resnet101

# borrowed from "https://github.com/marvis/pytorch-mobilenet"


def conv_dw(inp, oup, stride):
	return nn.Sequential(
		nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
		nn.BatchNorm2d(inp),
		nn.ReLU(inplace=True),

		nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
		nn.BatchNorm2d(oup),
		nn.ReLU(inplace=True),
	)


def conv_dw_1(inp, oup, kernel_size=3, padding=0, stride=1):
	return nn.Sequential(
		nn.Conv2d(inp, inp, kernel_size, stride,
				  padding, groups=inp, bias=False),
		nn.ReLU(inplace=True),

		nn.Conv2d(inp, oup, 1, 1, 0, bias=False),

	)


class ResNetLSTM(nn.Module):
	def __init__(self, num_classes, is_test=False, config=None, device=None, num_lstm=5):
		"""Compose a SSD model using the given components.
		"""
		super(ResNetLSTM, self).__init__()

		# alpha = 1
		# alpha_base = alpha
		# alpha_ssd = 0.5 * alpha
		# alpha_lstm = 0.25 * alpha


		resnet = resnet101(pretrained=True)
		all_modules = list(resnet.children())
		modules = all_modules[:-4] 
		self.base_net = nn.Sequential(*modules)

		modules = all_modules[6:7]
		self.conv_final = nn.Sequential(*modules)
		
		self.num_classes = num_classes
		self.is_test = is_test
		self.config = config

		lstm_layers = [BottleNeckLSTM(1024, 256),
					   BottleNeckLSTM(256, 64),
					   BottleNeckLSTM(64, 16),
					   ConvLSTMCell(16, 16),
					   ConvLSTMCell(16, 16)]
		
	
		self.lstm_layers =  nn.ModuleList([lstm_layers[i] for i in range(num_lstm)])

		self.extras = ModuleList([
			Sequential(
				Conv2d(in_channels=256, out_channels=128, kernel_size=1),
				ReLU(),
				conv_dw_1(inp=128, oup=256, kernel_size=3,
						  stride=2, padding=1),
				ReLU()
			),
			Sequential(
				Conv2d(in_channels=64, out_channels=32, kernel_size=1),
				ReLU(),
				conv_dw_1(inp=32, oup=64, kernel_size=3, stride=2, padding=1),
				ReLU()
			),
			Sequential(
				Conv2d(in_channels=16, out_channels=8, kernel_size=1),
				ReLU(),
				conv_dw_1(inp=8, oup=16, kernel_size=3, stride=2, padding=1),
				ReLU()
			),
			Sequential(
				Conv2d(in_channels=16, out_channels=8, kernel_size=1),
				ReLU(),
				conv_dw_1(inp=8, oup=16, kernel_size=3, stride=2, padding=1),
				ReLU()
			)
		])



		self.regression_headers = ModuleList([
			conv_dw_1(inp=512, oup=4 * 4, kernel_size=3, padding=1),
			conv_dw_1(inp=256, oup=6 * 4, kernel_size=3, padding=1),
			conv_dw_1(inp=64, oup=6 * 4, kernel_size=3, padding=1),
			conv_dw_1(inp=16, oup=6 * 4, kernel_size=3, padding=1),
			conv_dw_1(inp=16, oup=4 * 4, kernel_size=3, padding=1),
			conv_dw_1(inp=16, oup=4 * 4, kernel_size=3, padding=1, stride=2),
		])

		self.classification_headers = ModuleList([
			conv_dw_1(inp=512, oup=4 * num_classes, kernel_size=3, padding=1),
			conv_dw_1(inp=256, oup=6 * num_classes, kernel_size=3, padding=1),
			conv_dw_1(inp=64, oup=6 * num_classes, kernel_size=3, padding=1),
			conv_dw_1(inp=16, oup=6 * num_classes, kernel_size=3, padding=1),
			conv_dw_1(inp=16, oup=4 * num_classes, kernel_size=3, padding=1),
			conv_dw_1(inp=16, oup=4 * num_classes, kernel_size=3, padding=1, stride = 2),
		])


		if device:
			self.device = device
		else:
			self.device = torch.device(
				"cuda:0" if torch.cuda.is_available() else "cpu")
		if is_test:
			self.config = config
			self.priors = config.priors.to(self.device)

	def forward(self, x):
		confidences = []
		locations = []
		header_index = 0

		x = self.base_net(x)
		print(x.shape)
		confidence, location = self.compute_header(header_index, x)
		header_index += 1
		confidences.append(confidence)
		locations.append(location)

		x = self.conv_final(x)
		x, _ = self.lstm_layers[0](x)
		print(x.shape)
		confidence, location = self.compute_header(header_index, x)
		header_index += 1
		confidences.append(confidence)
		locations.append(location)

		

		for i in range(len(self.extras)):
			if (i < len(self.lstm_layers)-1): 
				x = self.extras[i](x)
				print(x.shape)
				x, _ = self.lstm_layers[i+1](x)
				confidence, location = self.compute_header(header_index, x)
				header_index += 1
				confidences.append(confidence)
				locations.append(location)
			else:
				x = self.extras[i](x)
				confidence, location = self.compute_header(header_index, x)
				header_index += 1
				confidences.append(confidence)
				locations.append(location)
				
			
		confidences = torch.cat(confidences, 1)
		locations = torch.cat(locations, 1)

		if self.is_test:
			confidences = F.softmax(confidences, dim=2)
			boxes = box_utils.convert_locations_to_boxes(
				locations, self.priors, self.config.center_variance, self.config.size_variance
			)
			boxes = box_utils.center_form_to_corner_form(boxes)
			return confidences, boxes
		else:
			# print(locations.size())
			# print(confidence.size())
			return confidences, locations

	def compute_header(self, i, x):
		confidence = self.classification_headers[i](x)
		confidence = confidence.permute(0, 2, 3, 1).contiguous()
		confidence = confidence.view(confidence.size(0), -1, self.num_classes)

		location = self.regression_headers[i](x)
		location = location.permute(0, 2, 3, 1).contiguous()
		location = location.view(location.size(0), -1, 4)
		return confidence, location

	def init_from_base_net(self, model):
		self.base_net.load_state_dict(torch.load(
			model, map_location=lambda storage, loc: storage), strict=True)
		self.source_layer_add_ons.apply(_xavier_init_)
		self.extras.apply(_xavier_init_)
		self.classification_headers.apply(_xavier_init_)
		self.regression_headers.apply(_xavier_init_)

	def detach_all(self):
		for layer in self.lstm_layers:
			layer.hidden_state.detach_()
			layer.cell_state.detach_()

	def init_from_pretrained_ssd(self, model):
		state_dict = torch.load(
			model, map_location=lambda storage, loc: storage)
		state_dict = {k: v for k, v in state_dict.items() if not (k.startswith(
			"classification_headers") or k.startswith("regression_headers"))}
		model_dict = self.state_dict()
		model_dict.update(state_dict)
		self.load_state_dict(model_dict)
		self.classification_headers.apply(_xavier_init_)
		self.regression_headers.apply(_xavier_init_)

	def init(self):
		self.base_net.apply(_xavier_init_)
		self.source_layer_add_ons.apply(_xavier_init_)
		self.extras.apply(_xavier_init_)
		self.classification_headers.apply(_xavier_init_)
		self.regression_headers.apply(_xavier_init_)

	def load(self, model):
		self.load_state_dict(torch.load(
			model, map_location=lambda storage, loc: storage))

	def save(self, model_path):
		torch.save(self.state_dict(), model_path)


class MatchPrior(object):
	def __init__(self, center_form_priors, center_variance, size_variance, iou_threshold):
		self.center_form_priors = center_form_priors
		self.corner_form_priors = box_utils.center_form_to_corner_form(
			center_form_priors)
		self.center_variance = center_variance
		self.size_variance = size_variance
		self.iou_threshold = iou_threshold

	def __call__(self, gt_boxes, gt_labels):
		if type(gt_boxes) is np.ndarray:
			gt_boxes = torch.from_numpy(gt_boxes)
		if type(gt_labels) is np.ndarray:
			gt_labels = torch.from_numpy(gt_labels)
		boxes, labels = box_utils.assign_priors(gt_boxes, gt_labels,
												self.corner_form_priors, self.iou_threshold)
		boxes = box_utils.corner_form_to_center_form(boxes)
		locations = box_utils.convert_boxes_to_locations(
			boxes, self.center_form_priors, self.center_variance, self.size_variance)
		return locations, labels


def _xavier_init_(m: nn.Module):
	if isinstance(m, nn.Conv2d):
		nn.init.xavier_uniform_(m.weight)


if __name__ == '__main__':
	model = ResNetLSTM(num_classes=21, config=config)
	i = torch.Tensor(1, 3, 300, 300)
	confidences, locations = model(i)
	print(confidences.shape)
	print(locations.shape)
