from .ResNet import *
from .VGG import *

def modelpool(MODELNAME, DATANAME):
    if 'imagenet' in DATANAME.lower():
        num_classes = 1000
    elif '100' in DATANAME.lower():
        num_classes = 100
    elif '10' in DATANAME.lower():
        num_classes = 10
    if MODELNAME.lower() == 'vgg16':
        return vgg16(num_classes=num_classes)
    elif MODELNAME.lower() == 'resnet18':
        return resnet18(num_classes=num_classes)
    elif MODELNAME.lower() == 'resnet34':
        return resnet34(num_classes=num_classes)
    elif MODELNAME.lower() == 'resnet20':
        return resnet20(num_classes=num_classes)
    else:
        print("Error:only support vgg16,resnet18,resnet20,resnet34,")
        exit(0)
