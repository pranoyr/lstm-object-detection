# LSTM-SSD Object Detection in Pytorch

## Dependencies
1. Python 3.6+
2. OpenCV
3. Pytorch 1.0 or Pytorch 0.4+

## Train
```
python train_ssd.py --datasets /home/neuroplex/data/ILSVRC/ --validation_dataset ./data/ --net lstm-mobilenet --batch_size 1 --num_epochs 200 --gpu 1
```

