import argparse
import os
import shutil

classes_2to21 = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21]
classes_once = [1, 37, 72, 70, 44]
classes_one = [1]
classes = []

spc_dict = [[1],									# person: 1
		[2, 3, 4, 5, 6, 7, 8, 9],					# vehicle: 8
		[10, 11, 13, 14, 15],						# outdoor: 5
		[16, 17, 18, 19, 20, 21, 22, 23, 24, 25],   # animal: 10
		[27, 28, 31, 32, 33],						# accessory: 5
		[34, 35, 36, 37, 38, 39, 40, 41, 42, 43],   # sports: 10
		[44, 46, 47, 48, 49, 50, 51],				# kitchen: 7
		[52, 53, 54, 55, 56, 57, 58, 59, 60, 61],   # food: 10
		[62, 63, 64, 65, 67, 70],					# furniture: 6
		[72, 73, 74, 75, 76, 77],					# electronic: 6
		[78, 79, 80, 81, 82],						# appliance: 5
		[84, 85, 86, 87, 88, 89, 90]				# indoor: 7
		]	

class_dict = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}

q_train = {'person': 6712, 'dog': 2350, 'cat': 1925, 'sports ball': 1818, 'tv': 1780, 'train': 1627, 'bottle': 1457, 'airplane': 1412, 'toilet': 1389, 'cell phone': 1276, 'dining table': 1247, 'clock': 1172, 'tie': 1158, 'car': 1068, 'bed': 1029, 'bird': 955, 'horse': 932, 'potted plant': 849, 'umbrella': 841, 'bus': 757, 'bicycle': 736, 'couch': 714, 'chair': 707, 'motorcycle': 692, 'stop sign': 689, 'refrigerator': 675, 'giraffe': 569, 'fire hydrant': 516, 'truck': 512, 'sandwich': 466, 'boat': 455, 'pizza': 423, 'bear': 410, 'vase': 387, 'bench': 366, 'cup': 366, 'fork': 362, 'bowl': 358, 'zebra': 353, 'elephant': 347, 'cow': 313, 'sink': 294, 'teddy bear': 289, 'banana': 274, 'surfboard': 269, 'knife': 251, 'skateboard': 250, 'spoon': 247, 'laptop': 229, 'toothbrush': 224, 'baseball bat': 219, 'traffic light': 202, 'scissors': 190, 'cake': 181, 'broccoli': 179, 'oven': 176, 'tennis racket': 149, 'sheep': 143, 'frisbee': 135, 'hot dog': 132, 'kite': 121, 'microwave': 118, 'mouse': 111, 'snowboard': 106, 'suitcase': 99, 'wine glass': 97, 'skis': 86, 'remote': 73, 'parking meter': 71, 'apple': 67, 'donut': 66, 'keyboard': 57, 'baseball glove': 55, 'orange': 54, 'book': 52, 'backpack': 45, 'handbag': 34, 'carrot': 19, 'toaster': 9, 'hair drier': 4}

q_val = {'person': 1692, 'dog': 574, 'cat': 526, 'sports ball': 431, 'train': 424, 'tv': 418, 'toilet': 339, 'clock': 333, 'bottle': 329, 'cell phone': 326, 'tie': 294, 'dining table': 282, 'bed': 274, 'airplane': 268, 'car': 255, 'bird': 246, 'horse': 232, 'umbrella': 219, 'bus': 212, 'potted plant': 204, 'motorcycle': 194, 'stop sign': 173, 'chair': 169, 'refrigerator': 159, 'bicycle': 154, 'couch': 153, 'giraffe': 139, 'boat': 128, 'fire hydrant': 125, 'bench': 118, 'sandwich': 115, 'truck': 113, 'bear': 108, 'pizza': 100, 'cup': 96, 'fork': 88, 'vase': 87, 'zebra': 86, 'bowl': 78, 'cow': 73, 'elephant': 73, 'teddy bear': 64, 'toothbrush': 62, 'banana': 61, 'surfboard': 61, 'sink': 59, 'spoon': 59, 'laptop': 59, 'knife': 58, 'baseball bat': 55, 'tennis racket': 51, 'traffic light': 51, 'cake': 49, 'skateboard': 48, 'oven': 42, 'broccoli': 41, 'hot dog': 37, 'scissors': 37, 'frisbee': 37, 'microwave': 34, 'kite': 31, 'suitcase': 30, 'sheep': 29, 'orange': 25, 'skis': 25, 'snowboard': 24, 'remote': 23, 'mouse': 23, 'wine glass': 23, 'parking meter': 21, 'donut': 20, 'keyboard': 15, 'backpack': 13, 'book': 9, 'handbag': 9, 'apple': 9, 'carrot': 8, 'baseball glove': 8, 'hair drier': 3, 'toaster': 2}

q_test = {'person': 1692, 'dog': 574, 'cat': 526, 'sports ball': 430, 'train': 424, 'tv': 418, 'toilet': 339, 'clock': 332, 'bottle': 328, 'cell phone': 325, 'tie': 294, 'dining table': 281, 'bed': 273, 'airplane': 267, 'car': 254, 'bird': 245, 'horse': 231, 'umbrella': 219, 'bus': 212, 'potted plant': 204, 'motorcycle': 193, 'stop sign': 173, 'chair': 169, 'refrigerator': 158, 'couch': 153, 'bicycle': 153, 'giraffe': 138, 'boat': 128, 'fire hydrant': 124, 'bench': 117, 'sandwich': 115, 'truck': 113, 'bear': 108, 'pizza': 100, 'cup': 95, 'vase': 87, 'fork': 87, 'zebra': 85, 'bowl': 78, 'cow': 73, 'elephant': 73, 'teddy bear': 64, 'toothbrush': 62, 'surfboard': 61, 'banana': 60, 'laptop': 59, 'sink': 58, 'spoon': 58, 'knife': 57, 'baseball bat': 55, 'tennis racket': 51, 'traffic light': 51, 'cake': 49, 'skateboard': 48, 'oven': 42, 'broccoli': 41, 'hot dog': 37, 'frisbee': 37, 'scissors': 36, 'microwave': 34, 'kite': 31, 'suitcase': 30, 'sheep': 29, 'orange': 24, 'skis': 24, 'snowboard': 24, 'remote': 23, 'mouse': 22, 'wine glass': 22, 'parking meter': 20, 'donut': 19, 'keyboard': 15, 'backpack': 12, 'book': 8, 'handbag': 8, 'apple': 8, 'baseball glove': 8, 'carrot': 7, 'hair drier': 3, 'toaster': 2}


def make_dir(path):
	"""
	Make a directory
	:param path:
	:return:
	"""
	if not os.path.exists(path):
		os.makedirs(path)


def pick_image(image_dir, output_dir, t_type, name, seq):
	"""
	Pick image to train|val|test
	:param image_dir:
	:param output_dir:
	:param t_type:
	:param name:
	:param seq:
	:return:
	"""
	global classes

	in_dir = image_dir + '/' + t_type
	
	make_dir(output_dir)
	o_dir = output_dir + '/img'
	make_dir(o_dir)
	o_dir += '/' + name
	make_dir(o_dir)
	o_dir += '/' + t_type
	make_dir(o_dir)

	if name == 'one':
		classes = classes_one
	elif name == 'once':
		classes = classes_once
	elif name == '2to21':
		classes = classes_2to21
	elif seq:
		classes = [int(name)]

	for c in classes:
		src_dir = in_dir + '/' + class_dict[c]
		imgs = [os.path.join(src_dir, x) for x in os.listdir(src_dir)]
		count = 0
		for img in imgs:
			im_name = img.rsplit('/')[-1]
			shutil.copyfile(img, o_dir + '/' + im_name)
			count += 1
		print('Copied %d images from %s to %s' % (count, src_dir, o_dir))


def main(args):
	image_dir = args.image_dir
	output_dir = args.output_dir
	name = args.name
	seq = args.seq

	pick_image(image_dir, output_dir, 'train', name, seq)
	pick_image(image_dir, output_dir, 'val', name, seq)
	pick_image(image_dir, output_dir, 'test', name, seq)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--image_dir', type=str, default='../dataset/processed', help='directory for processed images')
	parser.add_argument('--output_dir', type=str, default='../data', help='directory for data')
	parser.add_argument('--name', type=str, default='base20', help='name of folder')
	parser.add_argument('--seq', type=bool, default=False, help='Seq or not')

	args = parser.parse_args()
	main(args)
