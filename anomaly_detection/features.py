from models.cycle_gan_model import CycleGANModel
from options.test_options import TestOptions
from options.train_options import TrainOptions
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
import numpy as np
import pickle 


import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html, util

path = "results/pickles" # path de sauvegarde du pickle

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    original_model = create_model(opt)
    original_model.setup(opt)

    model = create_model(opt)
    model.setup(opt)
    model.netD_A.model = nn.Sequential(
        *list(original_model.netD_A.model)[:-1]
    )

    print('original model :')
    print(original_model.netD_A.model)
    print('modified model :')
    print(model.netD_A.model)

    # On récupère tous les feature ayer N-1
    embedding_dic = dict()
    for index, data in enumerate(dataset):
        if index > 5:
            break
        print(f'data element {index}')
        model.set_input(data)
        image_A, image_B = data['A'], data['B']
        embedding_A = model.netD_A(image_A)
        embedding_dic[index] = embedding_A
        
    # On les sauvegarde pour les étudier dans un code ultérieur
    current = os.getcwd()
    os.chdir(path)
    pickle.dump(embedding_dic, open(str("embedding_plot_rec_1"), 'wb'))  # embedding_plot_rec_1 nom à changer pour sauvegarder le pickle
    os.chdir(current)