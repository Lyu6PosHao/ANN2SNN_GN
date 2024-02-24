import numpy as np
import torch
from tqdm import tqdm
from utils import *
import torch.distributed as dist
import random
import os
from spikingjelly.activation_based import functional
from torch.utils.tensorboard import SummaryWriter   

def seed_all(seed=42):
    print(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def amp_train_ann(train_dataloader, test_dataloader, model, 
              epochs, device, loss_fn,lr=0.1,lr_min=1e-5,wd=5e-4 , save=None, parallel=False,
                rank=0):
    use_amp=True

    if rank==0:
        with open('./runs/'+save+'_log.txt','a') as log:
            log.write('lr={},epochs={},wd={}\n'.format(lr,epochs,wd))

    model.cuda(device)
    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=lr, weight_decay=wd, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,eta_min=lr_min, T_max=epochs)

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_acc=0.
    for epoch in range(epochs):
        model.train()
        if parallel:
            train_dataloader.sampler.set_epoch(epoch)
        epoch_loss = 0
        length = 0
        model.train()
        for img, label in tqdm(train_dataloader):
            img = img.to(device)
            label = label.to(device)
            
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                out = model(img)
                loss = loss_fn(out, label)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad()

            epoch_loss += loss.item()
            length += len(label)
        tmp_acc, val_loss = eval_ann(test_dataloader, model, loss_fn, device, rank)
        if parallel:
            dist.all_reduce(tmp_acc)
            tmp_acc/=dist.get_world_size()
        if rank == 0 and save != None and tmp_acc >= best_acc:
            checkpoint = {"model": model.state_dict(),
              "optimizer": optimizer.state_dict(),
              "scaler": scaler.state_dict()}
            torch.save(checkpoint, './saved_models/' + save + '.pth')
        if rank == 0:
            info='Epoch:{},Train_loss:{},Val_loss:{},Acc:{}'.format(epoch, epoch_loss/length,val_loss, tmp_acc.item())
            with open('./runs/'+save+'_log.txt','a') as log:
                log.write(info+'\n')
            if epoch % 10 == 0:
                print(model)
        best_acc = max(tmp_acc, best_acc)
        scheduler.step()

    return best_acc, model


def train_ann(train_dataloader, test_dataloader, model, 
              epochs, device, loss_fn,lr=0.1,lr_min=1e-6,wd=5e-4 , save=None, parallel=False,
                rank=0):
    # model.cuda(device)
    # writer = SummaryWriter('./runs/'+save)
    # mt=monitor.InputMonitor(model,SteppedReLU)
    # qcfs_vth={}
    # cnt=1
    # for name in mt.monitored_layers:
    #     qcfs=get_module_by_name(model,name)[1]
    #     #assert isinstance(qcfs,QCFS)
    #     qcfs_vth[str(cnt)+'+'+name]=qcfs.v_threshold
    #     #qcfs_p0[str(cnt)+'+'+name]=qcfs.p0
    #     cnt=cnt+1

    # mt.clear_recorded_data()
    # mt.remove_hooks()
    if parallel:
        wd=1e-4

    if rank==0:
        with open('./runs/'+save+'_log.txt','a') as log:
            log.write('lr={},epochs={},wd={}\n'.format(lr,epochs,wd))

    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=lr, weight_decay=wd, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,eta_min=lr_min, T_max=epochs)


    best_acc=eval_ann(test_dataloader, model, loss_fn, device, rank)[0]
    if parallel:
        dist.all_reduce(best_acc)
        best_acc/=dist.get_world_size()
    if rank==0:
        print(best_acc)
    for epoch in tqdm(range(epochs)):
        model.train()
        if parallel:
            train_dataloader.sampler.set_epoch(epoch)
        epoch_loss = 0
        length = 0
        model.train()
        for img, label in tqdm(train_dataloader):
            img = img.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            out = model(img)
            loss = loss_fn(out, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            length += len(label)

        tmp_acc, val_loss = eval_ann(test_dataloader, model, loss_fn, device, rank)
        if parallel:
            dist.all_reduce(tmp_acc)
            tmp_acc/=dist.get_world_size()
        if rank == 0 and save != None and tmp_acc >= best_acc:
            torch.save(model.state_dict(), './saved_models/' + save + '.pth')
        if rank == 0:
            info='Epoch:{},Train_loss:{},Val_loss:{},Acc:{},lr:{}'.format(epoch, epoch_loss/length,val_loss, tmp_acc.item(),scheduler.get_last_lr()[0])
            with open('./runs/'+save+'_log.txt','a') as log:
                log.write(info+'\n')
        best_acc = max(tmp_acc, best_acc)
        # print('Epoch:{},Train_loss:{},Val_loss:{},Acc:{}'.format(epoch, epoch_loss/length,val_loss, tmp_acc), flush=True)
        # print(f'lr={scheduler.get_last_lr()[0]}')
        # print('best_acc: ', best_acc)

        # writer.add_scalars('Acc',{'val_acc':tmp_acc,'best_acc':best_acc},epoch)
        # writer.add_scalars('Loss',{'train_loss':epoch_loss/length,'val_loss':val_loss},epoch)
        # writer.add_scalar('lr',scheduler.get_last_lr()[0],epoch)
        # writer.add_scalars('vth',qcfs_vth,epoch)
        scheduler.step()
        #print(module)
    # writer.close()
    return best_acc, model

def eval_snn(test_dataloader, model,loss_fn, device, sim_len=8, rank=0):
    tot = torch.zeros(sim_len).cuda()
    length = 0
    model = model.cuda()
    model.eval()

    with torch.no_grad():
        for idx, (img, label) in enumerate(tqdm((test_dataloader))):
            spikes = 0
            length += len(label)
            img = img.cuda()
            label = label.cuda()
            for t in range(sim_len):
                out = model(img)
                spikes += out
                tot[t] += (label==spikes.max(1)[1]).sum()
            spikes/=sim_len
            loss = loss_fn(spikes, label)
            functional.reset_net(model)
    return (tot/length),loss.item()/length

def eval_ann(test_dataloader, model, loss_fn, device, rank=0):
    epoch_loss = 0
    tot = torch.tensor(0.).cuda(device)
    model.eval()
    model.cuda(device)
    length = 0
    with torch.no_grad():
        for img, label in tqdm(test_dataloader):
            img = img.cuda(device)
            label = label.cuda(device)
            out = model(img)
            loss = loss_fn(out, label)
            epoch_loss += loss.item()
            length += len(label)    
            tot += (label==out.max(1)[1]).sum().data
    return (tot/length), epoch_loss/length
