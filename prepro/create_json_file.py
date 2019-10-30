import os
import json
import argparse
from pick_image import make_dir


def create_class_annotations(test_dir, json_file, output_dir, class_name):
	"""
	Create annotation files for each class in test set, set is created from val
	so image name will be like in validation set
	:param test_dir:
	:param json_file:
	:param output_dir:
	:param class_name: category name e.g dog, cat ...
	:return:
	"""

	input_img_path =  test_dir + class_name + '/'
	input_json_file = json_file
	output_dir = input_img_path
	output_path = output_dir + 'captions_test.json'

	image_subdirs = [x[2] for x in os.walk(input_img_path)]

	with open(input_json_file) as json_file:
		data = json.load(json_file)
		class_json = dict()
		class_json['info'] = data['info']
		class_json['licenses'] = data['licenses']
		class_json['images'] = []
		class_json['annotations'] = []
		class_json['type'] = 'captions'  # Add this to adapt with coco-eval

		for image in data['images']:
			if image['file_name'] in image_subdirs[0]:
				class_json['images'].append(image)

		for annotation in data['annotations']:
			image_id = annotation['image_id']
			file_name = ('COCO_%s2014_%012d.jpg' % ('val',  image_id))
			if file_name in image_subdirs[0]:
				class_json['annotations'].append(annotation)

	print("Finishing build JSON object to dump to {}".format(output_path))

	with open(output_path, 'w') as file:
		json.dump(class_json, file)


def create_annotations(image_dir, json_file, output_dir, t_type, name):
	"""
	Create annotation for a task
	:param image_dir:
	:param json_file:
	:param output_dir:
	:param t_type:
	:param name:
	:return:
	"""
	image_subdirs = [x[2] for x in os.walk(image_dir + '/' + name + '/' + t_type + '/')]

	if t_type == 'train':
		input_path = json_file + '/captions_train2014.json'
	else:		
		input_path = json_file + '/captions_val2014.json'

	make_dir(output_dir)
	output_path = output_dir + '/' + name
	make_dir(output_path)	
	output_path += '/captions_%s.json' % (t_type)

	if t_type == 'test':
		t_type = 'val'

	with open(input_path) as json_file:
		data = json.load(json_file)
		split_json = dict()
		split_json['info'] = data['info']
		split_json['licenses'] = data['licenses']
		split_json['images'] = []
		split_json['annotations'] = []
		split_json['type'] = 'captions'  # Add this to adapt with coco-eval

		for image in data['images']:
			if image['file_name'] in image_subdirs[0]:
				split_json['images'].append(image)

		for annotation in data['annotations']:
			image_id = annotation['image_id']
			file_name = ('COCO_%s2014_%012d.jpg' % (t_type,  image_id))
			if file_name in image_subdirs[0]:
				split_json['annotations'].append(annotation)

		print("Finishing build JSON object to dump to {}".format(output_path))

	with open(output_path, 'w') as file:
		json.dump(split_json, file)


def main(args):
	image_dir = args.image_dir
	json_file = args.json_file
	output_dir = args.output_dir
	name = args.name

	create_annotations(image_dir, json_file, output_dir, 'train', name)
	create_annotations(image_dir, json_file, output_dir, 'val', name)
	create_annotations(image_dir, json_file, output_dir, 'test', name)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--image_dir', type=str, default='../data/img', help='directory for images')
	parser.add_argument('--json_file', type=str, default='../dataset/original/annotations_trainval2014',
						help='directory for json files')
	parser.add_argument('--output_dir', type=str, default='../data/annotations', help='directory for output json file')
	parser.add_argument('--name', type=str, default='base20', help='name of folder')
	args = parser.parse_args()
	main(args)


