# from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
# from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
# from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
# from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
# from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
# from vision.ssd.resnet50_ssd1 import create_resnet18_ssd, create_resnet18_ssd_predictor
from vision.ssd.predictor import Predictor
from vision.ssd.lstm_mobilenet import MobileNetLSTM
from vision.utils.misc import Timer
from vision.ssd.config import mobilenetv1_ssd_config as config
import cv2
import sys
import  os
import numpy as np


# if len(sys.argv) < 5:
#     print('Usage: python run_ssd_example.py <net type>  <model path> <label path> <image path>')
#     sys.exit(0)


class_names = [name.strip() for name in open("./models/voc-model-labels.txt").readlines()]


# if net_type == 'vgg16-ssd':
#     net = create_vgg_ssd(len(class_names), is_test=True)
# elif net_type == 'mb1-ssd':
#     net = create_mobilenetv1_ssd(len(class_names), is_test=True)
# elif net_type == 'mb1-ssd-lite':
#     net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
# elif net_type == 'mb2-ssd-lite':
#     net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
# elif net_type == 'sq-ssd-lite':
#     net = create_squeezenet_ssd_lite(len(class_names), is_test=True)
# elif net_type == 'resnet-18':
#     net = create_resnet18_ssd(len(class_names), is_test=True)

net = MobileNetLSTM(num_classes=31, is_test=True, config=config)


model_path = "/Users/pranoyr/Desktop/lstm-mobilenet-Epoch-60.pth"
net.load(model_path)

# if net_type == 'vgg16-ssd':
#     predictor = create_vgg_ssd_predictor(net, candidate_size=200)
# elif net_type == 'mb1-ssd':
#     predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)
# elif net_type == 'mb1-ssd-lite':
#     predictor = create_mobilenetv1_ssd_lite_predictor(net, candidate_size=200)
# elif net_type == 'mb2-ssd-lite':
#     predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)
# elif net_type == 'sq-ssd-lite':
#     predictor = create_squeezenet_ssd_lite_predictor(net, candidate_size=200)
# elif net_type == 'resnet-18':
#     predictor = create_resnet18_ssd_predictor(net, candidate_size=200)
# if net_type == 'lstm-ssd':
predictor = Predictor(net, config.image_size, config.image_mean,
                          config.image_std,
                          nms_method=None,
                          iou_threshold=config.iou_threshold,
                          candidate_size=200,
                          sigma=None)


dir_path = './data/sample/'
imgs = []
for img_name in os.listdir(dir_path):
    img_path = os.path.join(dir_path, img_name)
    print(img_path)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgs.append(img)


video = np.array(imgs)

for image in video:
    boxes, labels, probs = predictor.predict(image, 10, 0.2)

print(boxes.shape)


draw_img  = video[-1]
for i in range(boxes.size(0)):
    box = boxes[i, :]
    cv2.rectangle(draw_img, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
    #label = f"""{voc_dataset.class_names[labels[i]]}: {probs[i]:.2f}"""
    label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
    cv2.putText(draw_img, label,
                (box[0] + 20, box[1] + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,  # font scale
                (255, 0, 255),
                2)  # line type
path = "run_ssd_example_output.jpg"
cv2.imwrite(path, draw_img)
print(f"Found {len(probs)} objects. The output image is {path}")
