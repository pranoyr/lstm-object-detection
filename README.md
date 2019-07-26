# LSTM-SSD Object Detection in Pytorch

## Dependencies
1. Python 3.6+
2. OpenCV
3. Pytorch 1.0 or Pytorch 0.4+

## Train
python train_ssd.py --datasets ./data --validation_dataset ./data/ --net vgg16-ssd --batch_size 2 --num_epochs 200 --scheduler cosine --lr 0.01 --t_max 200

