#!/usr/bin/env python
# coding: utf-8

# # Randomized Image Sampling for Explanations (RISE)

import os
import argparse
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from utils import (read_tensor, tensor_imshow,
                   get_class_name, preprocess,
                   RangeSampler)
from explanations import RISE

cudnn.benchmark = True

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device :', device)


parser = argparse.ArgumentParser()

# Directory with images split into class folders.
# Since we don't use ground truth labels for saliency all images can be
# moved to one class folder.
# e.g 'datasets/tiny-imagenet-200/val'
parser.add_argument('--datadir', required=True,
                    help='Directory with images (needs atleast 1 subfolder)')

parser.add_argument('--model', required=False, type=str, default='resnet18',
                    help='type of architecture. string must match pytorch zoo')

parser.add_argument('--workers', required=False, type=int, default=16,
                    help='Number of workers to load data')

parser.add_argument('--gpu_batch', required=False, type=int, default=250,
                    help='Size of batches for GPU (use maximum that the GPU allows)')


args = parser.parse_args()
print(args)


# Sets the range of images to be explained for dataloader.
args.range = range(95, 105)
# Size of imput images.
args.input_size = (224, 224)


def load_model(model_name='resnet18', ptrained=True):
    known_models = [x for x in dir(models)]
    if model_name not in known_models:
        raise ValueError('specified model doesnt exist in pytorch zoo')

    # This is equivalent to calling models.model_name(pretrained=True)
    # e.g models.alexnet(pretrained=True)
    model = getattr(models, model_name)(pretrained=ptrained)

    model.eval()
    return model


def example(img, top_k=3):
    saliency = explainer(img.to(device)).cpu().numpy()

    p, c = torch.topk(model(img.to(device)), k=top_k)
    p, c = p[0], c[0]

    plt.figure(figsize=(10, 5*top_k))
    for k in range(top_k):
        plt.subplot(top_k, 2, 2*k+1)
        plt.axis('off')
        plt.title('{:.2f}% {}'.format(100*p[k], get_class_name(c[k])))
        tensor_imshow(img[0])

        plt.subplot(top_k, 2, 2*k+2)
        plt.axis('off')
        plt.title(get_class_name(c[k]))
        tensor_imshow(img[0])
        sal = saliency[c[k]]
        plt.imshow(sal, cmap='jet', alpha=0.5)
        plt.colorbar(fraction=0.046, pad=0.04)

    plt.savefig('0-explain.png')
    # plt.show()
    plt.close()


# ## Explaining all images in dataloader
# Explaining the top predicted class for each image.

def explain_all(data_loader, explainer):
    # Get all predicted labels first
    target = np.empty(len(data_loader), np.int)
    for i, (img, _) in enumerate(tqdm(data_loader, total=len(data_loader), desc='Predicting labels')):
        p, c = torch.max(model(img.to(device)), dim=-1)
        target[i] = c[0]

    # Get saliency maps for all images in val loader
    explanations = np.empty((len(data_loader), *args.input_size))
    for i, (img, _) in enumerate(tqdm(data_loader, total=len(data_loader), desc='Explaining images')):
        saliency_maps = explainer(img.to(device))
        explanations[i] = saliency_maps[target[i]].cpu().numpy()
    return explanations


if __name__ == '__main__':
    # ## Prepare data
    dataset = datasets.ImageFolder(args.datadir, preprocess)

    # This example only works with batch size 1. For larger batches see RISEBatch in explanations.py.
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=RangeSampler(args.range))

    print('Found {: >5} images belonging to {} classes.'.format(len(dataset), len(dataset.classes)))
    print('      {: >5} images will be explained.'.format(len(data_loader) * data_loader.batch_size))


    # ## Load a black-box model for explanations from pytorch-zoo
    # ## choose from any of
    '''
    names  = ['alexnet', 'vgg16',
              'resnet18', 'resnet34', 'resnet50',
              'squeezenet1_0', 'densenet161', 'inception_v3',
              'googlenet', 'shufflenet_v2_x1_0', 'mobilenet_v2']
              and more in https://pytorch.org/docs/stable/torchvision/models.html
    '''
    model = load_model(args.model)
    model = nn.Sequential(model, nn.Softmax(dim=1))
    model.to(device)

    for p in model.parameters():
        p.requires_grad = False

    # To use multiple GPUs
    model = nn.DataParallel(model)

    # ## Create explainer instance

    explainer = RISE(model, args.input_size, args.gpu_batch)

    # Generate masks for RISE or use the saved ones.
    maskspath = 'masks.npy'
    generate_new = True

    if generate_new or not os.path.isfile(maskspath):
        explainer.generate_masks(N=6000, s=8, p1=0.1, savepath=maskspath)
    else:
        explainer.load_masks(maskspath)
        print('Masks are loaded.')

    # ## Explaining one instance
    # Producing saliency maps for top $k$ predicted classes.
    example(read_tensor('catdog.png'), 5)

    explanations = explain_all(data_loader, explainer)

    # Save explanations if needed.
    # explanations.tofile('exp_{:05}-{:05}.npy'.format(args.range[0], args.range[-1]))

    for i, (img, _) in enumerate(data_loader):
        p, c = torch.max(model(img.to(device)), dim=-1)
        p, c = p[0].item(), c[0].item()

        prob = torch.softmax(model(img.to(device)), dim=-1)
        pred_prob = prob[0][c]

        plt.figure(figsize=(10, 5))
        plt.suptitle('RISE Explanation for model {}'.format(args.model))

        plt.subplot(121)
        plt.axis('off')
        plt.title('{:.2f}% {}'.format(100*p, get_class_name(c)))
        tensor_imshow(img[0])

        plt.subplot(122)
        plt.axis('off')
        plt.title(get_class_name(c))
        tensor_imshow(img[0])
        sal = explanations[i]
        plt.imshow(sal, cmap='jet', alpha=0.5)
        # plt.colorbar(fraction=0.046, pad=0.04)

        plt.savefig('{}-explain-{}.png'.format(i+1, args.model))
        # plt.show()
        plt.close()
