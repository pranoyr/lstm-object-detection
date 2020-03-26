import numpy as np
import logging
import pathlib
import xml.etree.ElementTree as ET
import cv2
import os
import torch


def get_video_ids(root):
	""" Make a list of all annotations.
	"""
	video_ids = open(root, 'r').read().split()
	return video_ids


class VIDDataset:

	def __init__(self, root, transform=None, target_transform=None, is_test=False, keep_difficult=False, label_file=None):
		"""Dataset for VOC data.
		Args:
			root: the root of the VOC2007 or VOC2012 dataset, the directory contains the following sub-directories:
				Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
		"""
		self.root = pathlib.Path(root)
		self.transform = transform
		# self.window_size = 5
		self.target_transform = target_transform
		if is_test:
			image_sets_file = os.path.join(self.root, "ImageSets/VID/test.txt")
		else:
			image_sets_file = os.path.join(
				self.root, "ImageSets/VID/train_5.txt")

		#self.video_ids = get_video_ids(image_sets_file)
		self.video_ids = VIDDataset._read_image_ids(image_sets_file)
		self.keep_difficult = keep_difficult

		logging.info("No labels file, using default VOC classes.")
		self.class_names = ['__background__',  # always index 0
						'n02691156', 'n02419796', 'n02131653', 'n02834778',
						'n01503061', 'n02924116', 'n02958343', 'n02402425',
						'n02084071', 'n02121808', 'n02503517', 'n02118333',
						'n02510455', 'n02342885', 'n02374451', 'n02129165',
						'n01674464', 'n02484322', 'n03790512', 'n02324045',
						'n02509815', 'n02411705', 'n01726692', 'n02355227',
						'n02129604', 'n04468005', 'n01662784', 'n04530566',
						'n02062744', 'n02391049']


		self.class_dict = {class_name: i for i,
						   class_name in enumerate(self.class_names)}

	def __getitem__(self, index):
		video = []
		video_boxes = []
		video_labels = []
		video_path = os.path.join(self.root , 'Annotations/VID/train', self.video_ids[index].split()[0])

		# loop over frames in a video sample
		for img_name in os.listdir(video_path):
			img_path = os.path.join(video_path, img_name)
			boxes, labels = self._get_annotation(img_path)
			# if not self.keep_difficult:
			# 	boxes = boxes[is_difficult == 0]
			# 	labels = labels[is_difficult == 0]
			image = self._read_image(img_path.replace('Annotations', 'Data'))
			if self.transform:
				image, boxes, labels = self.transform(image, boxes, labels)
			if self.target_transform:
				boxes, labels = self.target_transform(boxes, labels)

			video.append(image)
			video_boxes.append(boxes)
			video_labels.append(labels)

		video = torch.stack(video)[0:5]
		video_boxes = torch.stack(video_boxes)[0:5]
		video_labels = torch.stack(video_labels)[0:5]

		# print("_______")
		# print(video.size())
		# print(video_boxes.size())
		# print(video_boxes.size())

		return video, video_boxes, video_labels

	# def get_image(self, index):
	# 	image_id = self.ids[index]
	# 	image = self._read_image(image_id)
	# 	if self.transform:
	# 		image, _ = self.transform(image)
	# 	return image

	# def get_annotation(self, index):
	# 	image_id = self.ids[index]
	# 	return image_id, self._get_annotation(image_id)

	def __len__(self):
		return len(self.video_ids)

	@staticmethod
	def _read_image_ids(image_sets_file):
		ids = []
		with open(image_sets_file) as f:
			for line in f:
				ids.append(line.rstrip())
		return ids

	def _get_annotation(self, image_id):
		annotation_file = image_id
		objects = ET.parse(annotation_file).findall("object")
		boxes = []
		labels = []
		is_difficult = []
		for object in objects:
			class_name = object.find('name').text.lower().strip()
			# we're only concerned with clases in our list
			if class_name in self.class_dict:
				bbox = object.find('bndbox')

				# VOC dataset format follows Matlab, in which indexes start from 0
				x1 = float(bbox.find('xmin').text) - 1
				y1 = float(bbox.find('ymin').text) - 1
				x2 = float(bbox.find('xmax').text) - 1
				y2 = float(bbox.find('ymax').text) - 1
				boxes.append([x1, y1, x2, y2])

				labels.append(self.class_dict[class_name])
				# is_difficult_str = object.find('difficult').text
				# is_difficult.append(int(is_difficult_str)
				# 					if is_difficult_str else 0)

		return (np.array(boxes, dtype=np.float32),
				np.array(labels, dtype=np.int64))

	def _read_image(self, image_id):
		image_file = image_id.replace("xml", "JPEG")
		image = cv2.imread(str(image_file))
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		return image

