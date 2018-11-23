import numpy as np
from tensorflow.keras.datasets import mnist, cifar10


import argparse
import datetime
import logging
import pathlib
import sys
import os


import models
from trainer import Trainer


def get_exp_path():
    '''Return new experiment path.'''
    return '/tmp/log/exp-{0}'.format(
        datetime.datetime.now().strftime('%m-%d-%H:%M:%S'))


def get_logger(path):
    '''Get logger for experiment.'''
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - '
                                  '%(levelname)s - %(message)s')

    # stderr log
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # file log
    handler = logging.FileHandler(path)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def load_data(dataset):
    if dataset == 'MNIST' or dataset == 'PI_MNIST':
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train.astype(np.float)
        X_test = X_test.astype(np.float)
        X_mean = np.mean(X_train, axis=0)
        X_train -= X_mean
        X_test -= X_mean
        X_train /= 128
        X_test /= 128

        if dataset == 'PI_MNIST':
            X_train = X_train.reshape(-1, 784)
            X_test = X_test.reshape(-1, 784)
        else:
            X_train = np.expand_dims(X_train, axis=3)
            X_test = np.expand_dims(X_test, axis=3)

    elif dataset == 'CIFAR10':
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        X_train = X_train.astype(np.float)
        X_test = X_test.astype(np.float)
        y_train = y_train.astype(np.int32).squeeze()
        y_test = y_test.astype(np.int32).squeeze()
        X_mean = np.mean(X_train, axis=0)
        X_train -= X_mean
        X_test -= X_mean
        X_train /= 128
        X_test /= 128

    else:
        assert False, 'Invalid value for `dataset`: %s' % dataset

    return (X_train, y_train), (X_test, y_test)


def get_model_and_dataset(params):
    if params.model == 'PI_MNIST':
        Model, dataset = models.PI_MNIST_Model, 'PI_MNIST'
    elif params.model == 'MNIST':
        Model, dataset = models.MNIST_Model, 'MNIST'
    elif params.model == 'CIFAR10':
        Model, dataset = models.CIFAR10_Model, 'CIFAR10'
    elif params.model == 'CIFAR10_VGG':
        Model, dataset = models.CIFAR10_VGG_Model, 'CIFAR10'
    elif params.model == 'CIFAR10_Resnet20':
        Model, dataset = models.CIFAR10_Resnet20, 'CIFAR10'
    elif params.model == 'CIFAR10_Resnet32':
        Model, dataset = models.CIFAR10_Resnet32, 'CIFAR10'
    elif params.model == 'CIFAR10_Resnet44':
        Model, dataset = models.CIFAR10_Resnet44, 'CIFAR10'
    elif params.model == 'CIFAR10_Resnet56':
        Model, dataset = models.CIFAR10_Resnet56, 'CIFAR10'
    else:
        assert False, 'Invalid value for `model`: %s' % params.model

    return Model(params.bits, params.dropout, params.weight_decay, params.stochastic), load_data(dataset)


def main():
    parser = argparse.ArgumentParser(description='DFXP')
    # experiment path
    parser.add_argument('--exp_path', type=str, default=None,
                        help='Experiment path')
    # model architecture
    parser.add_argument('--model', type=str, default='CIFAR10_Resnet20', help='Experiment model')
    parser.add_argument('--bits', type=int, default=8, help='DFXP bitwidth')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout keep probability')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay factor')
    # training
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--lr_decay_factor', type=float, default=0.1, help='Learning rate decay factor')
    parser.add_argument('--lr_decay_epoch', type=int, default=50, help='Learning rate decay epoch')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--n_epoch', type=int, default=150, help='Number of training epoch')
    parser.add_argument('--stochastic', action='store_true', help='Use stochastic quantization in backward pass')
    params = parser.parse_args()

    # experiment path
    if params.exp_path is None:
        params.exp_path = get_exp_path()
    pathlib.Path(params.exp_path).mkdir(parents=True, exist_ok=False)

    # logger
    logger = get_logger(params.exp_path + '/experiment.log')
    logger.info('Start of experiment')
    logger.info('============ Initialized logger ============')
    logger.info('\n\t' + '\n\t'.join('%s: %s' % (k, str(v))
        for k, v in sorted(dict(vars(params)).items())))

    # get model and dataset
    model, dataset = get_model_and_dataset(params)

    # build trainer
    trainer = Trainer(
        model=model,
        dataset=dataset,
        logger=logger,
        logdir=params.exp_path,
        lr=params.lr,
        lr_decay_factor=params.lr_decay_factor,
        lr_decay_epoch=params.lr_decay_epoch,
        momentum=params.momentum,
        n_epoch=params.n_epoch,
        batch_size=params.batch_size,
    )

    # training
    trainer.init_model()
    trainer.train()
    trainer.save_model(params.exp_path)

    # end
    logger.info('End of experiment')


if __name__ == '__main__':
    main()
