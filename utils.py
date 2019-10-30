import torch
import yaml
import pprint
import os
import matplotlib.pyplot as plt
import json
import pickle
import numpy as np


def append_vocab(check_point_vocab, vocab):
    """
    Progressively build vocabulary
    :param check_point_vocab:
    :param vocab:
    :return:
    """
    with open(check_point_vocab, 'rb') as old:
        old_vocab = pickle.load(old)

    old_vocab_size = len(old_vocab)

    for _, word in vocab.idx2word.items():
        old_vocab.add_word(word)

    vocab = old_vocab

    return vocab, old_vocab_size


def append_json(pseudo_labels, train_json):
    """
    Append to train_json
    :param pseudo_labels:
    :param train_json:
    :return:
    """

    # Modify the train annotation file of new task
    with open(train_json) as json_file:
        data = json.load(json_file)

    ids = [x['id'] for x in data['annotations']]

    # Make the id for annotations, increment from the max id to avoid duplication
    max_ids = max(ids)
    for labels in pseudo_labels:
        max_ids += 1
        labels['id'] = max_ids

    # Concatenate pseudo-labels to ground-truth labels
    data['annotations'] += pseudo_labels
    print("Length of annotations is {}".format(len(data['annotations'])))

    return data


def loss_visualize(train_step, train_loss_step, val_step, val_loss_step):
    """
    Visualize loss curve
    :param train_step:
    :param train_loss_step:
    :param val_step:
    :param val_loss_step:
    :return:
    """
    # Plot loss after 1 epoch
    plt.plot(train_step, train_loss_step, color='orange', label='Train loss')
    plt.plot(val_step, val_loss_step, color='blue', label='Validation loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Loss over time in training and validation')
    plt.legend()
    plt.show()


def make_paths_absolute(dir_, cfg):
    """
    Make a dir with abs path
    :param dir_:
    :param cfg:
    :return:
    """
    for key in cfg.keys():
        if type(cfg[key]) is dict:
            cfg[key] = make_paths_absolute(dir_, cfg[key])
    return cfg


def load_and_print_cfg(config_file):
    """
    Load and print configuration yaml file
    :param config_file:
    :return:
    """

    # Read YAML experiment definition file
    with open(config_file, 'r') as stream:
        cfg = yaml.load(stream)
    cfg = make_paths_absolute(os.path.dirname(config_file), cfg)

    # Print the configuration - just to make sure that you loaded what you
    # wanted to load
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(cfg)

    # Here is an example how you load modules of which you put the path in the
    # configuration. Use this for configuring the model you use, for dataset
    # loading, ...
    return cfg


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, cpkt_path, data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                 decoder_optimizer, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(cpkt_path, data_name, epoch, epochs_since_improvement, encoder,
                                 decoder, encoder_optimizer,
                                 decoder_optimizer, val_loss)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(cpkt_path, data_name, epoch, epochs_since_improvement, encoder,
                                 decoder, encoder_optimizer,
                                 decoder_optimizer, val_loss)
            self.counter = 0

    def save_checkpoint(self, cpkt_path, data_name, epoch, epochs_since_improvement, encoder,
                        decoder, encoder_optimizer,
                        decoder_optimizer, val_loss):
        """
        Saves model checkpoint.

        :param cpkt_path:
        :param data_name: base name of processed dataset
        :param epoch: epoch number
        :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
        :param encoder: encoder model
        :param decoder: decoder model
        :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
        :param decoder_optimizer: optimizer to update decoder's weights
        :param val_loss: validation loss
        """

        if self.verbose:
            print('Validation loss decreased ({} --> {}).  Saving model ...'.format(self.val_loss_min, val_loss))
        state = {'epoch': epoch,
                 'epochs_since_improvement': epochs_since_improvement,
                 'val_loss': val_loss,
                 'encoder': encoder,
                 'decoder': decoder,
                 'encoder_optimizer': encoder_optimizer,
                 'decoder_optimizer': decoder_optimizer}
        filename = 'checkpoint_' + data_name + '.pth.tar'
        # torch.save(state, cpkt_path + filename)
        self.val_loss_min = val_loss
        print("Saving the best model ...")
        torch.save(state, cpkt_path + 'BEST_' + filename)

        torch.save(decoder.state_dict(), os.path.join(
            cpkt_path, 'decoder.ckpt'))
        torch.save(encoder.state_dict(), os.path.join(
            cpkt_path, 'encoder.ckpt'))



