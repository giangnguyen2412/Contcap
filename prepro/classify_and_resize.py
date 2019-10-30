import argparse
import json
import os
from PIL import Image
import sys
import shutil
from create_json_file import create_class_annotations
from pick_image import class_dict, make_dir


def resize_image(image, size):
    return image.resize(size, Image.ANTIALIAS)


def classify_data(in_dir, ins_file, output_dir, t_type, size):
    if (t_type != 'train') and (t_type != 'val') and (t_type != 'test'):
        print('Type is invalid.')
        return

    make_dir(output_dir)
    make_dir(output_dir + '/' + t_type)

    if t_type == 'test':
        t_type = 'val'

    with open(ins_file) as json_file:
        data = json.load(json_file)
        categories = data['categories']

        image_num = len(data['images'])
        classes = dict()  # Contains 80 classes of MS-COCO, key is class id
        for category in categories:
            classes[category['id']] = category['name']

        for key, value in classes.items():
            make_dir(output_dir + '/%s/%s' % (t_type, value))

        annotations = data['annotations']

        annotation_dict = {}
        image_count = 0

        id2class_dict = {}

        for i, annotation in enumerate(annotations, 1):
            image_id = annotation['image_id']
            file_name = 'COCO_%s2014_%012d.jpg' % (t_type, image_id)
            category_id = annotation['category_id']
            id2class_dict[file_name] = classes[category_id]

            src_file = in_dir + '/%s' % file_name
            dst_file = output_dir + '/%s/%s/%s' % (t_type, classes[category_id], file_name)

            if not os.path.exists(src_file):
                continue

            if image_id in annotation_dict:  # At least two annotations in an image, should delete this image
                if os.path.exists(dst_file):
                    image_count -= 1
                    os.remove(dst_file)
                continue  # Here to only pick images having just one class / or one annotation
            else:
                annotation_dict[image_id] = 1

            with open(src_file, 'r+b') as f:
                with Image.open(f) as img:
                    img = resize_image(img, size)
                    img.save(dst_file, img.format)

            if os.path.exists(dst_file):  # If copy successfully
                image_count += 1
            else:
                print("Copy failed!")

            if i%1000 == 0:  # With 1k annotations parsed, we print the number of images having a single class
                sys.stdout.write('\r')
                sys.stdout.write("{}/{} images have been copied!".format(image_count, image_num))
                sys.stdout.flush()
        sys.stdout.write('\r')
        print("{}/{} images have been copied!".format(image_count, image_num))

        if t_type != 'val':	# Only create mapping from filename to class on validation set
            id2class_path = '../dataset/processed/id2class.json'
            with open(id2class_path, 'w') as file:
                json.dump(id2class_dict, file)
            print("Finishing build JSON object to dump to {}".format(id2class_path))


def pick_testset(output_dir):
    o_val_dir = output_dir + '/val'
    o_test_dir = output_dir + '/test'
    make_dir(o_test_dir)
    subdirs = [os.path.join(o_val_dir, x) for x in os.listdir(o_val_dir)]

    img_count = 0
    for subdir in subdirs:
        if not os.path.isdir(subdir):
            continue
        category = subdir.rsplit('/')[-1]
        dst_dir = o_test_dir + '/' + category
        make_dir(dst_dir)
        imgs = [os.path.join(subdir, x) for x in os.listdir(subdir)]

        n = len(imgs)//2
        m = 0
        for img in imgs:
            if m == n:
                break
            shutil.move(img, dst_dir)
            m += 1
            img_count += 1
    print("{} images have been moved for testset!".format(img_count))


def main(args):
    train_dir = args.train_dir
    val_dir = args.val_dir
    ins_dir = args.ins_dir
    output_dir = args.output_dir
    size = (args.image_size, args.image_size)

    classify_data(train_dir, ins_dir + '/instances_train2014.json', output_dir, 'train', size)
    classify_data(val_dir, ins_dir + '/instances_val2014.json', output_dir, 'val', size)

    pick_testset(output_dir)

    for _, class_name in class_dict.items():
        create_class_annotations(test_dir= '../dataset/processed/test/',
                                 json_file='../dataset/original/annotations_trainval2014/captions_val2014.json',
                                 output_dir='../dataset/processed/test/',
                                 class_name=class_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='../dataset/original/train2014',
                        help='directory for training set')
    parser.add_argument('--val_dir', type=str, default='../dataset/original/val2014',
                        help='directory for validating set')

    parser.add_argument('--ins_dir', type=str, default='../dataset/original/annotations_trainval2014',
                        help='directory for instance files')

    parser.add_argument('--output_dir', type=str, default='../dataset/processed',
                        help='directory for image after classified and resized')

    parser.add_argument('--image_size', type=int, default=256, help='size for image after processing')

    args = parser.parse_args()
    main(args)
