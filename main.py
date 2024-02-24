import argparse
import torch.multiprocessing as mp
from Models import modelpool
from Preprocess import datapool
from funcs import *
from utils import replace_maxpool2d_with_avgpool2d,replace_qcfs_with_sn,replace_relu_with_qcfs
import torch.nn as nn

parser = argparse.ArgumentParser()

parser.add_argument('action', default='train', type=str, help='Action: train or test.')
parser.add_argument('--gpus', default=1, type=int, help='GPU number to use.')
parser.add_argument('--bs', default=128, type=int, help='Batchsize')
parser.add_argument('--lr', default=0.1, type=float, help='Learning rate') 
parser.add_argument('--lr_min', default=1e-6, type=float, help='Learning rate min in cosine annealing')
parser.add_argument('--wd', default=5e-4, type=float, help='Weight decay')
parser.add_argument('--epochs', default=300, type=int, help='Training epochs')
parser.add_argument('--id', default=None, type=str, help='Model identifier when saving and loading')
parser.add_argument('--device', default='cuda', type=str, help='cuda or cpu')
parser.add_argument('--l', default=16, type=int, help='The parameter L in QCFS. Details are in the QCFS paper.')
parser.add_argument('--t', default=16, type=int, help='T (time-steps)')
parser.add_argument('--mode', type=str, default='ann')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--data', type=str, default='cifar100')
parser.add_argument('--model', type=str, default='pre_act_resnet34',help='Model architecture')
parser.add_argument('--sn_type', type=str, default='gn')
parser.add_argument('--tau', type=int, default=4,help='members of one gn or pgn')
parser.add_argument('--amp', type=bool,default=False, help='use amp on imagenet')
args = parser.parse_args()
seed_all(args.seed)
if __name__ == "__main__":
    
    # preparing data
    train, test = datapool(args.data, args.bs)
    # preparing model
    model = modelpool(args.model, args.data)
    model = replace_maxpool2d_with_avgpool2d(model)
    model = replace_relu_with_qcfs(model, L=args.l)
    criterion = nn.CrossEntropyLoss()
    if args.action == 'train':
        model=model.to(args.device)
        train_ann(train, test, model, args.epochs, args.device, criterion, args.lr, args.lr_min,args.wd, args.id)
    elif args.action == 'test':
        model.load_state_dict(torch.load('./saved_models/' + args.id + '.pth'))
        if args.mode == 'snn':
            model = replace_qcfs_with_sn(model,members=args.tau,sn_type=args.sn_type)
            model.to(args.device)
            acc = eval_snn(test, model,criterion, args.device, args.t)
            print('Accuracy: ', acc)
        elif args.mode == 'ann':
            model.to(args.device)
            acc, _ = eval_ann(test, model, criterion, args.device)
            print('Accuracy: {:.4f}'.format(acc))
        else:
            AssertionError('Unrecognized mode')
