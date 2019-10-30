import nltk
import pickle
import argparse
import glob
import os
from collections import Counter
from pycocotools.coco import COCO

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def make_dir(path):
    """
    Make a directory
    :param path:
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)


class Vocabulary(object):
    """
    Simple vocabulary wrapper.
    """
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def build_vocab(caption, threshold, name):
    """
    Build a vocabulary from annotations
    :param caption:
    :param threshold:
    :param name:
    :return:
    """
    json_path = caption + '/' + name + '/captions_train.json'
    jsons = glob.glob(json_path, recursive=True)

    print(jsons)

    counter = Counter()
    for json in jsons:
        print("Tokenizing in file {}".format(json))
        coco = COCO(json)

        ids = coco.anns.keys()
        for i, id in enumerate(ids):
            caption = str(coco.anns[id]['caption'])
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            counter.update(tokens)

            if (i+1) % 1000 == 0:
                print("[{}/{}] Tokenized the captions.".format(i+1, len(ids)))

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab, words


def main(args):
    caption = args.caption
    vocab_path = args.vocab_path
    threshold = args.threshold
    name = args.name

    vocab, _ = build_vocab(caption, threshold, name)
    make_dir(vocab_path)
    vocab_path += '/' + name
    make_dir(vocab_path)
    vocab_path += '/vocab.pkl'
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)

    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption', type=str, default='../data/annotations', help='directory for annotation')
    parser.add_argument('--vocab_path', type=str, default='../data/vocab', help='directory for vocabulary')
    parser.add_argument('--threshold', type=int, default=4, help='minimum word count threshold')
    parser.add_argument('--name', type=str, default='base20', help='name of folder')
    args = parser.parse_args()
    main(args)
