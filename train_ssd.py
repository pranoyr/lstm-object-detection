import argparse
import os
import logging
import sys
import itertools


import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from vision.utils.misc import str2bool, Timer, freeze_net_layers, store_labels
from vision.ssd.lstm_mobilenet import MatchPrior
# from vision.ssd.vgg_ssd import create_vgg_ssd
# from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd
# from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite
# from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite
# from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite
# from vision.ssd.resnet50_ssd1 import create_resnet18_ssd
from vision.ssd.lstm_mobilenet import MobileNetLSTM
from vision.ssd.lstm_resnet import ResNetLSTM
from vision.ssd.lstm_resnet1 import ResNetLSTM1
from vision.ssd.lstm_resnet2 import ResNetLSTM2
from vision.ssd.lstm_resnet3 import ResNetLSTM3

from vision.datasets.voc_dataset_video import VOCDataset
from vision.datasets.vid_dataset import VIDDataset
from vision.datasets.open_images import OpenImagesDataset
from vision.nn.multibox_loss import MultiboxLoss
from vision.ssd.config import vgg_ssd_config
from vision.ssd.config import mobilenetv1_ssd_config
# from vision.ssd.config import squeezenet_ssd_config
from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform
import random
import numpy as np
from args import parser

args = parser.parse_args()

DEVICE = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
	seed = 0
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


# def train(loader, net, criterion, optimizer, device, debug_steps=2, epoch=-1):
#     net.train(True)
#     # encoder.train()
#     # decoder.train()

#     running_loss = 0.0
#     running_regression_loss = 0.0
#     running_classification_loss = 0.0
#     for i, data in enumerate(loader):

#         # batch , timesteps, channel, height, width
#         #videos = torch.Tensor(2,2,3,300,300)
#         videos, videos_boxes, videos_labels = data

#         videos = videos.to(device)
#         videos_boxes = videos_boxes.to(device)
#         videos_labels = videos_labels.to(device)

#         # timesteps, batch,  channel, height, width
#         # torch.Size([2, 24, 512, 38, 38])
#         # out_enc_23, out_enc_final = encoder(videos)
#         # out_dec_23, out_dec_final = decoder([out_enc_23, out_enc_final])

#         # permute videos
#         videos = videos.permute(1, 0, 2, 3, 4)

#         # permute boxes and labels to match videos size
#         videos_boxes = videos_boxes.permute(1, 0, 2, 3)
#         videos_labels = videos_labels.permute(1, 0, 2)

#         for j in range(videos.size(0)):
#             video = videos[j, :, :, :, :]  # get image batch for each time step
#             # out_dec_final_batch = out_dec_final[j,:,:,:,:] # get image batch for each time step

#             #images = [out_dec_23_batch , out_dec_final_batch]

#             confidence, locations = net(video)

#             #confidence, locations = net(images)
#             regression_loss, classification_loss = criterion(
#                 confidence, locations, videos_labels[j], videos_boxes[j])  # TODO CHANGE BOXES
#             loss = regression_loss + classification_loss

#             optimizer.zero_grad()
#             loss.backward(retain_graph=True)
#             optimizer.step()

#             # calculating loss for all timesteps
#             running_loss += loss.item()
#             running_regression_loss += regression_loss.item()
#             running_classification_loss += classification_loss.item()

#         # net.zero_grad()
#         # running_loss += total_loss.item()
#         # running_regression_loss += total_regression_loss.item()
#         # running_classification_loss += total_classification_loss.item()
#         if i and i % debug_steps == 0:
#             avg_loss = running_loss / debug_steps
#             avg_reg_loss = running_regression_loss / debug_steps
#             avg_clf_loss = running_classification_loss / debug_steps
#             logging.info(
#                 f"Epoch: {epoch}, Step: {i}, " +
#                 f"Average Loss: {avg_loss:.4f}, " +
#                 f"Average Regression Loss {avg_reg_loss:.4f}, " +
#                 f"Average Classification Loss: {avg_clf_loss:.4f}"
#             )
#             running_loss = 0.0
#             running_regression_loss = 0.0
#             running_classification_loss = 0.0

#         net.detach_all()
#     net.detach_all()


def train(loader, net, criterion, optimizer, device, debug_steps=2, epoch=-1):
	net.train(True)
	# encoder.train()
	# decoder.train()

	running_loss = 0.0
	running_regression_loss = 0.0
	running_classification_loss = 0.0
	for i, data in enumerate(loader):
		# batch , timesteps, channel, height, width
		#videos = torch.Tensor(2,2,3,300,300)
		videos, videos_boxes, videos_labels = data

		videos = videos.to(device)
		videos_boxes = videos_boxes.to(device)
		videos_labels = videos_labels.to(device)

		# timesteps, batch,  channel, height, width
		# torch.Size([2, 24, 512, 38, 38])
		# out_enc_23, out_enc_final = encoder(videos)
		# out_dec_23, out_dec_final = decoder([out_enc_23, out_enc_final])

		# permute videos
		videos = videos.permute(1, 0, 2, 3, 4)

		# permute boxes and labels to match videos size
		videos_boxes = videos_boxes.permute(1, 0, 2, 3)
		videos_labels = videos_labels.permute(1, 0, 2)

		tot_loss = 0
		reg_loss = 0
		cls_loss = 0
		for j in range(videos.size(0)):
			video = videos[j]  # get image batch for each time step
			# out_dec_final_batch = out_dec_final[j,:,:,:,:] # get image batch for each time step

			#images = [out_dec_23_batch , out_dec_final_batch]

			confidence, locations = net(video)

			#confidence, locations = net(images)
			regression_loss, classification_loss = criterion(
				confidence, locations, videos_labels[j], videos_boxes[j])  # TODO CHANGE BOXES
			loss = regression_loss + classification_loss

			# optimizer.zero_grad()
			# loss.backward(retain_graph=True)
			# optimizer.step()

			tot_loss += loss
			reg_loss += regression_loss
			cls_loss += classification_loss

			# calculating loss for all timesteps
			# running_loss += loss.item()
			# running_regression_loss += regression_loss.item()
			# running_classification_loss += classification_loss.item()

		optimizer.zero_grad()
		tot_loss.backward()
		optimizer.step()

		# net.zero_grad()
		running_loss += tot_loss.item()
		running_regression_loss += reg_loss.item()
		running_classification_loss += cls_loss.item()
		if i and i % debug_steps == 0:
			avg_loss = running_loss / debug_steps
			avg_reg_loss = running_regression_loss / debug_steps
			avg_clf_loss = running_classification_loss / debug_steps
			print(
				f"Epoch: {epoch}, Step: {i}, " +
				f"Average Loss: {avg_loss:.4f}, " +
				f"Average Regression Loss {avg_reg_loss:.4f}, " +
				f"Average Classification Loss: {avg_clf_loss:.4f}"
			)
			running_loss = 0.0
			running_regression_loss = 0.0
			running_classification_loss = 0.0

		net.detach_all()
	# net.detach_all()


def test(loader, net, criterion, device):
	net.eval()
	running_loss = 0.0
	running_regression_loss = 0.0
	running_classification_loss = 0.0
	num = 0
	for _, data in enumerate(loader):
		images, boxes, labels = data
		images = images.to(device)
		boxes = boxes.to(device)
		labels = labels.to(device)
		num += 1

		with torch.no_grad():
			confidence, locations = net(images)
			regression_loss, classification_loss = criterion(
				confidence, locations, labels, boxes)
			loss = regression_loss + classification_loss

		running_loss += loss.item()
		running_regression_loss += regression_loss.item()
		running_classification_loss += classification_loss.item()
	return running_loss / num, running_regression_loss / num, running_classification_loss / num


if __name__ == '__main__':
	timer = Timer()

	logging.info(args)
	# if args.net == 'vgg16-ssd':
	#     create_net = create_vgg_ssd
	#     config = vgg_ssd_config
	# elif args.net == 'mb1-ssd':
	#     create_net = create_mobilenetv1_ssd
	#     config = mobilenetv1_ssd_config
	# elif args.net == 'mb1-ssd-lite':
	#     create_net = create_mobilenetv1_ssd_lite
	#     config = mobilenetv1_ssd_config
	# elif args.net == 'sq-ssd-lite':
	#     create_net = create_squeezenet_ssd_lite
	#     config = squeezenet_ssd_config
	# elif args.net == 'mb2-ssd-lite':
	#     def create_net(num): return create_mobilenetv2_ssd_lite(
	#         num, width_mult=args.mb2_width_mult)
	#     config = mobilenetv1_ssd_config
	# elif args.net == 'resnet-18-ssd':
	#     create_net = create_resnet18_ssd
	#     config = vgg_ssd_config
	if args.net == 'lstm-mobilenet':
		create_net = MobileNetLSTM
		config = mobilenetv1_ssd_config
	elif args.net == 'lstm-resnet':
		create_net = ResNetLSTM
		config = vgg_ssd_config
	elif args.net == 'lstm-resnet1':
		create_net = ResNetLSTM1
		config = vgg_ssd_config
	elif args.net == 'lstm-resnet2':
		create_net = ResNetLSTM2
		config = vgg_ssd_config
	elif args.net == 'lstm-resnet3':
		create_net = ResNetLSTM3
		config = vgg_ssd_config
	else:
		logging.fatal("The net type is wrong.")
		parser.print_help(sys.stderr)
		sys.exit(1)

	train_transform = TrainAugmentation(
		config.image_size, config.image_mean, config.image_std)
	target_transform = MatchPrior(config.priors, config.center_variance,
								  config.size_variance, 0.5)

	test_transform = TestTransform(
		config.image_size, config.image_mean, config.image_std)

	logging.info("Prepare training datasets.")
	datasets = []
	for dataset_path in args.datasets:
		if args.dataset_type == 'vid':
			dataset = VIDDataset(dataset_path, transform=train_transform,
								 target_transform=target_transform)
			label_file = os.path.join(
				args.checkpoint_folder, "voc-model-labels.txt")
			store_labels(label_file, dataset.class_names)
			num_classes = len(dataset.class_names)
		elif args.dataset_type == 'open_images':
			dataset = OpenImagesDataset(dataset_path,
										transform=train_transform, target_transform=target_transform,
										dataset_type="train", balance_data=args.balance_data)
			label_file = os.path.join(
				args.checkpoint_folder, "open-images-model-labels.txt")
			store_labels(label_file, dataset.class_names)
			logging.info(dataset)
			num_classes = len(dataset.class_names)

		else:
			raise ValueError(
				f"Dataset tpye {args.dataset_type} is not supported.")
		datasets.append(dataset)
	logging.info(f"Stored labels into file {label_file}.")
	train_dataset = ConcatDataset(datasets)
	logging.info("Train dataset size: {}".format(len(train_dataset)))
	train_loader = DataLoader(train_dataset, args.batch_size,
							  num_workers=args.num_workers,
							  shuffle=False)
	logging.info("Prepare Validation datasets.")
	if args.dataset_type == "voc":
		val_dataset = VOCDataset(args.validation_dataset, transform=test_transform,
								 target_transform=target_transform, is_test=True)
	elif args.dataset_type == 'open_images':
		val_dataset = OpenImagesDataset(dataset_path,
										transform=test_transform, target_transform=target_transform,
										dataset_type="test")
		logging.info(val_dataset)
	#logging.info("validation dataset size: {}".format(len(val_dataset)))

	# val_loader = DataLoader(val_dataset, args.batch_size,
	#                         num_workers=args.num_workers,
	#                         shuffle=False)
	logging.info("Build network.")
	net = create_net(num_classes)
	min_loss = -10000.0
	last_epoch = -1

	base_net_lr = args.base_net_lr if args.base_net_lr is not None else args.lr
	extra_layers_lr = args.extra_layers_lr if args.extra_layers_lr is not None else args.lr
	if args.freeze_base_net:
		logging.info("Freeze base net.")
		freeze_net_layers(net.base_net)
		params = itertools.chain(net.source_layer_add_ons.parameters(), net.extras.parameters(),
								 net.regression_headers.parameters(), net.classification_headers.parameters())
		params = [
			{'params': itertools.chain(
				net.source_layer_add_ons.parameters(),
				net.extras.parameters()
			), 'lr': extra_layers_lr},
			{'params': itertools.chain(
				net.regression_headers.parameters(),
				net.classification_headers.parameters()
			)}
		]
	elif args.freeze_net:
		freeze_net_layers(net.base_net)
		freeze_net_layers(net.source_layer_add_ons)
		freeze_net_layers(net.extras)
		params = itertools.chain(
			net.regression_headers.parameters(), net.classification_headers.parameters())
		logging.info("Freeze all the layers except prediction heads.")
	else:
		# params = [
		# 	{'params': net.base_net.parameters(), 'lr': base_net_lr},
		# 	{'params': net.extras.parameters(), 'lr': extra_layers_lr},
		# 	{'params': itertools.chain(
		# 		net.regression_headers.parameters(),
		# 		net.classification_headers.parameters()
		# 	)},
		# 	{'params': net.lstm_layers.parameters()},
		# 	{'params': net.conv_final.parameters()}

		# ]
		params = net.parameters()

	timer.start("Load Model")
	if args.resume:
		logging.info(f"Resume from the model {args.resume}")
		net.load(args.resume)
	elif args.base_net:
		logging.info(f"Init from base net {args.base_net}")
		net.init_from_base_net(args.base_net)
	elif args.pretrained_ssd:
		logging.info(f"Init from pretrained ssd {args.pretrained_ssd}")
		net.init_from_pretrained_ssd(args.pretrained_ssd)
	logging.info(
		f'Took {timer.end("Load Model"):.2f} seconds to load the model.')

	net.to(DEVICE)
	# print(net.parameters)

	criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
							 center_variance=0.1, size_variance=0.2, device=DEVICE)
	optimizer = torch.optim.RMSprop(
		params, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
	logging.info(f"Learning rate: {args.lr}, Base net learning rate: {base_net_lr}, "
				 + f"Extra Layers learning rate: {extra_layers_lr}.")

	if args.scheduler == 'multi-step':
		logging.info("Uses MultiStepLR scheduler.")
		milestones = [int(v.strip()) for v in args.milestones.split(",")]
		scheduler = MultiStepLR(optimizer, milestones=milestones,
								gamma=0.1, last_epoch=last_epoch)
	elif args.scheduler == 'cosine':
		logging.info("Uses CosineAnnealingLR scheduler.")
		scheduler = CosineAnnealingLR(
			optimizer, args.t_max, last_epoch=last_epoch)
	else:
		logging.fatal(f"Unsupported Scheduler: {args.scheduler}.")
		parser.print_help(sys.stderr)
		sys.exit(1)

	logging.info(f"Start training from epoch {last_epoch + 1}.")
	for epoch in range(last_epoch + 1, args.num_epochs):
		train(train_loader, net, criterion, optimizer,
			  device=DEVICE, debug_steps=args.debug_steps, epoch=epoch)

		# scheduler.step()

		if epoch % 2 == 0:
			# val_loss, val_regression_loss, val_classification_loss = test(val_loader, net, criterion, DEVICE)
			# logging.info(
			#     f"Epoch: {epoch}, " +
			#     f"Validation Loss: {val_loss:.4f}, " +
			#     f"Validation Regression Loss {val_regression_loss:.4f}, " +
			#     f"Validation Classification Loss: {val_classification_loss:.4f}"
			# )
			model_path = os.path.join(
				args.checkpoint_folder, f"{args.net}-Epoch-{epoch}.pth")
			net.save(model_path)
			logging.info(f"Saved model {model_path}")
