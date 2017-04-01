from __future__ import print_function, division
import os.path
import numpy as np
import numpy.random as npr
from collections import defaultdict

data_folder = os.path.abspath(os.path.realpath(__file__) + '/../../data/')   


def get_mnist(N_dev=5000, shuffle=False, perm=False, preprocessing=None):
    """Load the MNIST dataset

    Must be located in data/mnist/mnist.npz

    Keyword Arguments:
        N_dev {number} -- Number of samples to use as validation data (picked from the training data) (default: {5000})
        shuffle {bool} -- Shuffle the training set (before splitting train/dev) (default: {False})
        perm {bool} -- Permute the pixels (default: {False})
        preprocessing {function} -- preprocessing function (default: {None})

    Returns:
        tuple -- Returns train_x, dev_x, test_x, train_y, dev_y, test_y
    """
    mnist_data = np.load(data_folder+'/mnist/mnist.npz')
    train_x = mnist_data['x_train'].reshape(60000, 784).astype(float)
    train_y = mnist_data['y_train'].astype(int)
    test_x = mnist_data['x_test'].reshape(10000, 784).astype(float)
    test_y = mnist_data['y_test'].astype(int)

    order = range(len(train_x))
    if shuffle:
        if os.path.isfile(data_folder + '/mnist_rand_split.txt'):
            order = np.loadtxt(data_folder + '/mnist_rand_split.txt').astype(int)
        else:
            npr.shuffle(order)
            np.savetxt(data_folder + '/mnist_rand_split.txt', order, fmt='%d')
    train_x, dev_x = train_x[order[:-N_dev]], train_x[order[-N_dev:]]
    train_y, dev_y = train_y[order[:-N_dev]], train_y[order[-N_dev:]]

    if preprocessing is not None:
        train_x = preprocessing(train_x)
        dev_x = preprocessing(dev_x)
        test_x = preprocessing(test_x)

    if perm:
        if os.path.isfile(data_folder + '/mnist_perm.txt'):
            order = np.loadtxt(data_folder + '/mnist_perm.txt').astype(int)
        else:
            order = range(len(train_x[0]))
            npr.shuffle(order)
            print(order)
            np.savetxt(data_folder + '/mnist_perm.txt', order, fmt='%d')
        train_x = train_x[:, order].reshape(-1, 784)
        dev_x = dev_x[:, order].reshape(-1, 784)
        test_x = test_x[:, order].reshape(-1, 784)

    return train_x, dev_x, test_x, train_y, dev_y, test_y

def get_mr(char=False, N_train=1500, N_dev=200,N_test=300):
    """Load the Cornell Movie Review Dataset 
    
    Must be located in data/rt-polarity under two files rt-polarity.pos and rt-polarity.neg
    
    Keyword Arguments:
        char {bool} -- Character level sentences (as opposed to word level) (default: {False})
        N_train {number} -- Number of train samples (default: {1500})
        N_dev {number} -- Number of validation samples (default: {200})
        N_test {number} -- Number of test samples (default: {300})
    
    Returns:
        tuple -- train_x, dev_x, train_y, dev_y, dic, rdic. Here dic maps words to ids and rdic does the inverse operation
    """
    if N_train + N_dev + N_test != 2000:
        raise ValueError('The size of the train, validation and test set must sum to 2000')
    dic = defaultdict(lambda: len(dic))
    _ = dic['<p>']
    sentences, labels = [], []
    with open(data_folder+'/rt-polarity/rt-polarity.pos', 'r') as f:
        for l in f:
            line = l[:-1] if char else l[:-1].split()
            sent = [dic[w] for w in line]
            sentences.append(sent)
            labels.append(1)
    with open(data_folder+'/rt-polarity/rt-polarity.neg', 'r') as f:
        for l in f:
            line = l[:-1] if char else l[:-1].split()
            sent = [dic[w] for w in line]
            sentences.append(sent)
            labels.append(0)
    sentences, labels = np.asarray(sentences), np.asarray(labels)
    order = range(len(sentences))
    npr.shuffle(order)
    train_x, dev_x = sentences[order[:N_train]], sentences[order[N_train:]]
    train_y, dev_y = labels[order[:N_train]], labels[order[N_train:]]
    return train_x, dev_x, train_y, dev_y, dic, reverse_dic(dic)


def get_ptb(char=False):
    """Loads the penn treebank dataset
    
    Must be located in data/ptb under train.txt, dev.txt, test.txt
    
    Keyword Arguments:
        char {bool} -- [description] (default: {False})
    
    Returns:
        [type] -- [description]
    """
    dic = defaultdict(lambda: len(dic))
    _, _, _, _ = dic['<p>'], dic['<unk>'], dic['<s>'], dic['</s>']
    train, dev, test = [], [], []
    with open(data_folder+'/ptb/train.txt', 'r') as f:
        for l in f:
            line = l[:-1] if char else l[:-1].split()
            sent = [dic['<s>']]+[dic[w] for w in line]+[dic['</s>']]
            train.append(sent)
    with open(data_folder+'/data/ptb/dev.txt', 'r') as f:
        for l in f:
            line = l[:-1] if char else l[:-1].split()
            sent = [dic['<s>']]
            sent += [(dic['<unk>'] if w not in dic.keys() else dic[w]) for w in line]
            sent += [dic['</s>']]
            dev.append(sent)
    with open(data_folder+'/data/ptb/test.txt', 'r') as f:
        for l in f:
            line = l[:-1] if char else l[:-1].split()
            sent = [dic['<s>']]
            sent += [(dic['<unk>'] if w not in dic.keys() else dic[w]) for w in line]
            sent += [dic['</s>']]
            test.append(sent)
    train, dev, test = np.asarray(train), np.asarray(dev), np.asarray(test)
    order = range(len(train))
    npr.shuffle(order)
    train = train[order]
    return train, dev, test, dic, reverse_dic(dic)

###### Useful functions for pre/post-processing
def reverse_dic(dic):
    """Reverses a dictionary

    This assumes the dicitonary is injective
    
    Arguments:
        dic {dict} -- Dictionary to reverse
    
    Returns:
        dict-- Reversed dictionary
    """
    rev_dic = dict()
    for k, v in dic.items():
        rev_dic[v] = k
    return rev_dic

def rescale(x):
    """Rescale the data between -1 and 1
    
    returns x / max(|x|)
    
    Arguments:
        x {np.array} -- Input data
    
    Returns:
        np.array -- Rescaled data
    """
    M = np.abs(x).max()
    x = x / M
    return x

def whiten(x):
    """Whiten the data
    
    returns (x - mean(x)) / std(x)
    
    Arguments:
        x {np.array} -- Input data
    
    Returns:
        np.array -- Whitened data
    """
    mu = x.mean()
    std = x.std()
    x = (x-mu)/std
    return x


def to_str(l, rdic, char=False):
    """Given a sequence of indices, returns the corresponding string
    
    Arguments:
        l {list} -- List of indices
        rdic {dict} -- Dictionary mapping indices to strings
    
    Keyword Arguments:
        char {bool} -- Controls whether to add a whitespace between each token (default: {False})
    
    Returns:
        str -- Corresponding string
    """
    if char:
        sent = ''.join([rdic[i] for i in l])
    else:
        sent = ' '.join([rdic[i] for i in l])
    if sent[-4:] == '</s>':
        sent = sent[:-4]
    return sent
