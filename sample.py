import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np 
import argparse
import pickle 
import os
from torchvision import transforms 
from prepro.build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from PIL import Image


def visualize(image_path, words, smooth=True):
    """
    Visualizes caption with weights at every word.
    :param image_path: path to image that has been captioned
    :param words: caption
    :param smooth: smooth weights?
    """

    # Uncomment to switch backend and choose your backend is necessary
    # plt.switch_backend("TKAgg")
    image = Image.open(image_path)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    plt.text(0, 1, '%s' % words, color='black', backgroundcolor='white', fontsize=12)
    plt.imshow(image)

    plt.set_cmap(cm.Greys_r)
    plt.axis('image')
    plt.show()


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_image(image_path, transform=None):
    """
    Load an image and return
    :param image_path:
    :param transform:
    :return:
    """
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image


def main(args):
    # Image pre-processing
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build models
    checkpoint = torch.load(args.model)
    encoder = checkpoint['encoder']
    decoder = checkpoint['decoder']
    '''
    encoder = EncoderCNN(args.embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))
    '''

    # Prepare an image
    image = load_image(args.image, transform)
    image_tensor = image.to(device)
    
    # Generate an caption from the image
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
    
    # Print out the image and the generated caption
    print (sentence)
    # image = Image.open(args.image)
    visualize(args.image, sentence)
    # plt.imshow(np.asarray(image))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='input image for generating caption')
    # parser.add_argument('--image', type=str, default = 'png/cat2.jpg', help='input image for generating caption')
    '''
    parser.add_argument('--encoder_path', type=str, default='models/seq/44_distill_seq/best/encoder.ckpt', 
                        help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='models/seq/44_distill_seq/best/decoder.ckpt', 
                        help='path for trained decoder')
    '''
    parser.add_argument('--model', type=str, required=True,
                        help='path for trained model')
    parser.add_argument('--vocab_path', type=str, required=True,
                        help='path for vocabulary wrapper')
    
    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    args = parser.parse_args()
    main(args)
