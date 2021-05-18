# yolov3

## PyTorch-YOLOv3

A minimal PyTorch implementation of YOLOv3, with support for training, evaluation. Besides, application with different loss such as CIOU, DIOU etc.

## Base Environment

```
>>> numpy.__version__
'1.18.5'
>>> torch.__version__
'1.6.0'
>>> torchvision.__version__
'0.7.0'
>>> tensorflow.__version__
'2.3.0'
>>> terminaltables.__version__
'3.1.0'
>>> tensorboard.__version__
'2.3.0'
```

## Sample data

```
$ cd data/custom/
```

## Train on Custom Dataset

### Custom model

Run the commands below to create a custom model definition, replacing <num-classes> with the number of classes in your dataset.

```
$ cd config/                                # Navigate to config dir
$ bash create_custom_model.sh 2 # Will create custom model 'yolov3-custom.cfg' 
```

### Classes

Add class names to data/custom/face.names. This file should have one row per class name.

```
face
face_mask
```

### Image Folder

Move the images of your dataset to data/custom/images/.

### Annotation Folder

Move your annotations to data/custom/labels/. 
The dataloader expects that the annotation file corresponding to the image data/custom/images/*.jpg has the path data/custom/labels/train.txt. Each row in the annotation file should define one bounding box, using the syntax label_idx x_center y_center width height. The coordinates should be scaled [0, 1], and the label_idx should be zero-indexed and correspond to the row number of the class name in data/custom/face.names.

### Define Train and Validation Sets

In data/custom/train.txt and data/custom/valid.txt, add paths to images that will be used as train and validation data respectively.

### Train with MSE loss

#### To train on the custom dataset run:

```
$ python train.py --model_def config/yolov3-custom.cfg --data_config config/face.data
```

```
$ train.py [-h] [--epochs EPOCHS] [--batch_size BATCH_SIZE]
                [--gradient_accumulations GRADIENT_ACCUMULATIONS]
                [--model_def MODEL_DEF] [--data_config DATA_CONFIG]
                [--pretrained_weights PRETRAINED_WEIGHTS] [--n_cpu N_CPU]
                [--img_size IMG_SIZE]
                [--checkpoint_interval CHECKPOINT_INTERVAL]
                [--evaluation_interval EVALUATION_INTERVAL]
                [--compute_map COMPUTE_MAP]
                [--multiscale_training MULTISCALE_TRAINING]
```

#### Training log

```
---- [Epoch 30/100, Batch 883/896] ----
+------------+--------------+--------------+--------------+
| Metrics    | YOLO Layer 0 | YOLO Layer 1 | YOLO Layer 2 |
+------------+--------------+--------------+--------------+
| grid_size  | 11           | 22           | 44           |
| loss       | 0.766502     | 0.641550     | 0.605858     |
| x          | 0.022278     | 0.037665     | 0.091217     |
| y          | 0.020608     | 0.037068     | 0.066900     |
| w          | 0.019547     | 0.007745     | 0.016767     |
| h          | 0.017360     | 0.010410     | 0.009473     |
| conf       | 0.662465     | 0.513463     | 0.396596     |
| cls        | 0.024243     | 0.035200     | 0.024905     |
| cls_acc    | 100.00%      | 100.00%      | 100.00%      |
| recall50   | 0.888889     | 1.000000     | 1.000000     |
| recall75   | 0.777778     | 0.888889     | 0.777778     |
| precision  | 0.727273     | 0.346154     | 0.132353     |
| conf_obj   | 0.803481     | 0.836010     | 0.885408     |
| conf_noobj | 0.001976     | 0.001750     | 0.001544     |
+------------+--------------+--------------+--------------+
Total loss 2.0139098167419434
```

## Train with DIOU loss

```
$ python traindiou.py --model_def config/yolov3-custom.cfg --data_config data/custom/face.data
```

## Train with CIOU loss

```
$ python trainciou.py --model_def config/yolov3-custom.cfg --data_config data/custom/face.data
```

## Test

Evaluates the model on your own dataset.

```
$ python test.py --weights_path checkpoints/**.pth --data_config data/custom/face.data --class_path data/custom/face.names 
```

