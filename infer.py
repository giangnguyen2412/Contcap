import torch
import json
import torchvision.transforms as transforms
import pickle
import os

from PIL import Image
from prepro.pick_image import class_dict, make_dir
from utils import load_and_print_cfg
from prepro.build_vocab import Vocabulary


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image(image_path, transform=None):
    """
    :param image_path:
    :param transform:
    :return:
    """
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)

    if transform is not None:
        image = transform(image).unsqueeze(0)

    return image


def infer_caption(img_path, json_path, model, vocab_path, prediction_path, id2class_path):
    """
    Compute average metrics
    :param img_path:
    :param json_path:
    :param model:
    :param vocab_path:
    :param prediction_path:
    :param id2class_path: '../dataset/preprocessed/id2class.json'
    :return:
    """

    class_num = len(class_dict)
    if id2class_path is not None:
        with open(id2class_path) as json_file:
            id2class_dict = json.load(json_file)

    # Image pre-processing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # Load vocabulary wrapper
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    print("Reading {} entries to vocab {}".format(vocab.idx, vocab_path))

    # Load model
    checkpoint = torch.load(model)
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()

    annotation_path = json_path
    with open(annotation_path) as json_file:
        data = json.load(json_file)

    images = data['images']

    # Prediction for every class
    prediction = []
    # Prediction splitted by class
    prediction_class = {}

    img_num = len(images)
    img_gray_num = 0
    for i, img in enumerate(images, 1):
        image_id = img['id']
        file_path = img_path + img['file_name']
        # Prepare an image
        image = load_image(file_path, transform)
        image_tensor = image.to(device)

        #image_class = id2class_dict[img['file_name']]  # Get image class from file name
        # Uncomment above to support id2 class
        image_class = 'test'

        # Generate an caption from the image
        if image_tensor.size()[1] == 1:
            img_gray_num += 1
            continue
        feature = encoder(image_tensor)
        sampled_ids = decoder.sample(feature)
        sampled_ids = sampled_ids[0].cpu().numpy()  # (1, max_seq_length) -> (max_seq_length)

        # Convert word_ids to words
        sampled_caption = []
        for word_id in sampled_ids:
            word = vocab.idx2word[word_id]
            sampled_caption.append(word)
            if word == '<end>':
                break
        sentence = ' '.join(sampled_caption)
        sentence = sentence.replace('<start>', '')
        sentence = sentence.replace('<end>', '')
        entry = {}
        entry['image_id'] = image_id
        entry['caption'] = sentence
        prediction.append(entry)

        # A dictionary with key is class name, each entry of dictionary is a list of prediction for that class
        if image_class not in prediction_class:
            prediction_class[image_class] = [entry]
        else:
            prediction_class[image_class].append(entry)

        if i%100 == 0:
            print("Tested on {}/{} images on test set".format(i, img_num))

    ''' Uncomment to support id2 class
    for class_name, prediction in prediction_class.items():
        json_class_path = prediction_path + class_name + '/'
        make_dir(json_class_path)

        with open(json_class_path + 'prediction.json', 'w') as file:
            json.dump(prediction, file)
    '''

    # TODO: Why run on experiment.py normally but here failed?
    print("Can not process {} gray images".format(img_gray_num))
    if prediction_path is not None:
        make_dir(prediction_path)
        with open(prediction_path + 'prediction.json', 'w') as file:
            json.dump(prediction, file)

    return prediction


if __name__ == '__main__':

    cfg = load_and_print_cfg('config.yaml')

    img_path = cfg['infer']['img_path']
    json_path = cfg['infer']['json_path']
    model = cfg['infer']['model']
    vocab_path = cfg['infer']['vocab_path']
    prediction_path = cfg['infer']['prediction_path']
    id2class_path = cfg['infer']['id2class_path']

    infer_caption(img_path=img_path,
                  json_path=json_path,
                  model=model, vocab_path=vocab_path,
                  prediction_path=prediction_path, id2class_path=None)

