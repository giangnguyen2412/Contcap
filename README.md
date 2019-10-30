# Data preparation
## Dataset
- Using [MSCOCO 2014 dataset](http://cocodataset.org/#download) because this dataset contains both image captioning annotations and objects detection annotations

## Data splitting
- This diagram is to split MSCOCO by class and pick 20 first classes:
![](https://github.com/luulinh90s/pytorch-framework/blob/master/tutorials/03-advanced/image_captioning/png/workflow_dataset.JPG)

- 20 first classes are like below:
```python
class_dict = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}
```

Here is the super class dictionary of MSCOCO 2014

```python
superclass_dict = {'person': 1, 'vehicle': 8, 'outdoor': 5, 'animal': 10, 'accessory': 5,
                   'sports': 10, 'kitchen': 7, 'food': 10, 'furniture': 6, 'electronic': 6,
                   'appliance': 5, 'indoor': 7}
```

The class means, for example, superclass person accounts 1 first class. Superclass vehicle takes next 8 classes. Superclass outdoor takes 5 next classes and so on.

So first 20 classes are about `person, vehicle, outdoor and animal`, then its reasonable to add classes of indoor or food to observe catastrophic forgetting.

*It should be noticed that by adding new classes to make a new task, we need to rebuild the vocabulary and annotation files*

# Experiments
Diagram of of loss over training and validation
![](https://github.com/luulinh90s/pytorch-framework/blob/giangnv_dev/tutorials/03-advanced/image_captioning/png/training_epoch_1.png)

## Fine tuning
The flow of doing finetuning is like below:
![](https://github.com/luulinh90s/pytorch-framework/blob/giangnv_dev/tutorials/03-advanced/image_captioning/png/fine_tuning.JPG)
### Result
Inquiry the [Google Sheet](https://docs.google.com/spreadsheets/d/1xscvow3zym9HhqekD0KLfBdwk74HLEWwhXpV2i4Zo9w/edit#gid=737597256)

# Test the model 

```bash
$ python sample.py --image='png/example.png'
```
![](https://github.com/luulinh90s/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/png/red_car.png)

# Management
## Project schedule 
[Google Sheet](https://docs.google.com/spreadsheets/d/1xscvow3zym9HhqekD0KLfBdwk74HLEWwhXpV2i4Zo9w/edit#gid=737597256)
## Project file structure
![](https://github.com/luulinh90s/pytorch-framework/blob/giangnv_dev/tutorials/03-advanced/image_captioning/png/KakaoTalk_20190813_105358745.jpg)

# Problems I faced
## Problem 1
### Description
When I run validate the model, I was computing the loss of validation and then accumulating this loss value by this:
```python
loss_over_validation += loss
```
Then I face the problem of running out of CUDA memory.
### Solution
The problem is intead of getting the loss value, I get the loss object and accumulated, and this loss value takes a large amount of memory over iterations. This correction will solve the problem.
```python
loss_over_validation += loss.item()
```
Here, `loss.item()` will return the value of loss, not the loss object since no CUDA memory will be taken more.

## Problem 2
### Description
When using **pickle** to load my vocabulary that is stored before, I got this problem.
`<class 'tuple'>: (<class 'AttributeError'>, AttributeError("Can't get attribute 'Vocabulary' on <module '__main__' from '/home/dexter/pytorch-framework/tutorials/03-advanced/image_captioning/infer.py'>",), None)`
### Solution
By importing `Vocabulary` class, I solved this problem because without importing, the interpreter can not recognize what is object Vocabulary. By all of that, adding this import line helps the program working:
```python3
from build_vocab import Vocabulary
```
## Problem 3 - How to freeze some neurons in a layers
### Description
When adding the new neurons to layers, unfortunately Pytorch doest provide any methods to freeze some specific neurons. A workaround solution is to set the `gradient` of the neuron to 0 before `optimizer.step()` like this:
`decoder.embed.weight.grad[:1000, :] = 0`
Here I freeze the first 1000 neurons in the tensor `decoder.embed.weight`.
Details of implementation can be found in `train.py`
