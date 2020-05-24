import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

def test_loader(out_dataset, workers):

    transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((125.3/255, 123.0/255, 113.9/255), \
                             (63.0/255, 62.1/255.0, 66.7/255.0)),
                ])

    in_testset = torchvision.datasets.CIFAR100(root='./data/', train= False,
                                               download=True, transform = transform)
    in_testloader = torch.utils.data.DataLoader(in_testset, batch_size = 1,
                                                num_workers = workers, shuffle = False)

    out_testset = torchvision.datasets.ImageFolder("./data/{}".format(out_dataset), transform=transform)
    out_testloader = torch.utils.data.DataLoader(out_testset, batch_size=1,
                                                num_workers=workers, shuffle = False)
    return in_testloader, out_testloader

