from .getdataloader import *

def datapool(DATANAME, batchsize):
    if DATANAME.lower() == 'cifar10':
        return GetCifar10(batchsize)
    elif DATANAME.lower() == 'cifar100':
        return GetCifar100(batchsize)
    elif DATANAME.lower() == 'imagenet':
        return GetImageNet(batchsize)
    else:
        print("Error:only support cifar10,cifar100,imagenet")
        exit(0)
