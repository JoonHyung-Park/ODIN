import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import dataload

import warnings
warnings.filterwarnings("ignore")


def main(args):
    """
    Recommended parameters for each out_distribution dataset

    Tiny-Imagenet -- epsilon: 0.0014, temperature: 1000
    LSUN dataset  -- epsilon: 0.0028, temperature: 1000

    You can change the hyperparameters as you want.
    """
    epsilon = args.epsilon
    T = args.temperature
    out_dataset = args.out_dataset

    # Load Pretrained densenet model (CIFAR100)
    densenet = torch.load("./models/densenet100.pth")

    ################DO NOT MODIFY##################
    #####YOU DON'T NEED TO UNDERSTAND THIS PART####
    for i, (name, module) in enumerate(densenet._modules.items()):
        module = recursion_change_bn(densenet)
    #densenet.eval()
    ################################################


    # Load in-distribution test set && out-distribution test set
    in_testloader, out_testloader = dataload.test_loader(out_dataset,workers=2)

    # Files for calculate metric
    f1 = open("./softmax_scores/confidence_Base_In.txt", 'w')
    f2 = open("./softmax_scores/confidence_Base_Out.txt", 'w')
    g1 = open("./softmax_scores/confidence_Our_In.txt", 'w')
    g2 = open("./softmax_scores/confidence_Our_Out.txt", 'w')

    N = 10000
    t0 = time.time()

    print("Processing in-distribution images")
    ###In-distribution###
    for idx, data in enumerate(in_testloader):


        # TODO 1: calculate the max softmax score of baseline (no perturbation & no T scaling)

        # In order to use metric.py, pass the max_score(float) to below line
        max_score: float  = None
        f1.write("{}, {}, {}\n".format(T, epsilon, max_score))


        # TODO 2: calculate the max softmax score of ODIN
        # Hint: torch.nn.autograd.Variable would be helpful

        max_score: float  = None
        g1.write("{}, {}, {}\n".format(T, epsilon, max_score))

        if idx  % 100 == 99:
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(idx+1, N, time.time()-t0))
            t0 = time.time()

        if idx == N - 1: break


    t0 = time.time()
    print("Processing out-of-distribution images")
    ###Out-of-Distributions###
    for idx, data in enumerate(out_testloader):



        # TODO 3: calculate the max softmax score of baseline (no perturbation & no T scaling)
        max_score: float  = None
        f2.write("{}, {}, {}\n".format(T, epsilon, max_score))




        # TODO 4: calculate the max softmax score of baseline (no perturbation & no T scaling)
        max_score: float  = None
        g2.write("{}, {}, {}\n".format(T,epsilon, max_score))

        if idx % 100 == 99:
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(idx+1, N, time.time()-t0))
            t0 = time.time()

        if idx == N-1: break


#########DO NOT MODIFY THIS FUNTION#########
#########Funtion for torch version sync####
def recursion_change_bn(module):
    if isinstance(module, torch.nn.BatchNorm2d):
        module.track_running_stats = 1
    else:
        for i, (name, module1) in enumerate(module._modules.items()):
            module1 = recursion_change_bn(module1)
    return module
#############################################

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='ODIN implementation')
    parser.add_argument(
        '--epsilon',
        type=float,
        default=0.0014,
        help= 'perturbation magnitude')

    parser.add_argument(
        '--temperature',
        type=int,
        default=1000,
        help = 'temperature scaling')
    parser.add_argument(
        '--out_dataset',
        default="tiny-imagenet",
        type=str,
        help=' tiny-imagenet | lsun ')

    args = parser.parse_args()
    main(args)
