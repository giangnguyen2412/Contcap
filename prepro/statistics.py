import os
import matplotlib.pyplot as plt
import numpy as np

DATASET = 'MSCOCO'
PARENT_PATH = '../dataset/processed'
NUM_PER_IMG = 80
FONT_SIZE = 7


def cal_num(path):
    """
    Calculate the number of each class:
    path - the directory where contains the images divided into classes
    return: a dictionary with key is name of class, value is the number of images of each class respectively
    """	
    stat = {}

    subdirs = os.listdir(path)
    for subdir in subdirs:
        p = os.path.join(path, subdir)
        if not os.path.isdir(p):
            continue
        stat[subdir] = len([img for img in os.listdir(p) if img.find('.jpg')])
    
    return stat


def bar_chart(objs, vals, t):
    """
    Draw bar chart to show the statistics of the number of images of each class
    objs - objects to draw
    vals - values corresponding to objects
    t - 'train' for training set and 'val' for validation test
    """

    objs = list(reversed(objs))
    vals = list(reversed(vals))

    y_pos = np.arange(len(objs))

    plt.barh(y_pos, vals, align='center', alpha=0.5)
    plt.yticks(y_pos, objs, fontsize=FONT_SIZE)
    plt.xlabel('Number of images')
    plt.ylabel('%d classes' % NUM_PER_IMG)
    plt.title(DATASET + ": " + t)

    plt.show()
    plt.clf()


def make_stat(t):
    """
    Make the statistics of the number of images of each class
    t - 'train' for training set and 'val' for validation set
    """

    path = PARENT_PATH + '/' + t
    classes = cal_num(path)	
    classes = sorted(classes.items(), key=lambda item: item[1], reverse=True)

    cnt = 0     
    objs = []
    vals = []
    for key, val in classes:
        objs += [key]
        vals += [val]
        cnt += 1

        if cnt == NUM_PER_IMG:
            bar_chart(objs, vals, t)	
            cnt = 0
            objs = []
            vals = []

    if cnt != 0:
        bar_chart(objs, vals, t)


def main():
    make_stat('train')
    make_stat('val')


if __name__ == '__main__':
    main()
