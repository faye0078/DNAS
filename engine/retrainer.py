import os
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import cv2
import torch

from utils.loss import SegmentationLosses
from dataloaders import make_data_loader
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.evaluator import Evaluator
from model.RetrainNet import RetrainNet
from utils.copy_state_dict import copy_state_dict

class Trainer(object):
    def __init__(self, args):
        self.args = args
        # data
        kwargs = {'num_workers': args.num_worker, 'pin_memory': True, 'drop_last':True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

        # model
        torch.cuda.empty_cache()
        if args.model_name == 'DNAS':
            layers = np.ones([args.layers, 4])
            cell_arch = np.load(args.cell_arch)
            connections = np.load(args.model_encode_path)
            model = RetrainNet(layers, 4, connections, cell_arch, self.args.dataset, self.nclass, 'normal')
            
        self.model = model
        if args.cuda:
            self.model = self.model.cuda()
        
        # train
        self.saver = Saver(args)
        self.criterion = SegmentationLosses(weight=None, cuda=args.cuda).build_loss(mode=args.loss_type)
        iters_per_epoch = len(self.train_loader)
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                      args.epochs, iters_per_epoch, min_lr=args.min_lr)
        optimizer = torch.optim.SGD(
                model.parameters(),
                args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay
            )
        
        self.optimizer = optimizer
        
        self.evaluator = Evaluator(self.nclass)
        
        # load
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            self.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'],  strict=False)


            copy_state_dict(self.optimizer.state_dict(), checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print(self.best_pred)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            self.start_epoch = 0


    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        
        tbar = tqdm(self.train_loader, ncols=80)
        for i, sample in enumerate(tbar):
            image = sample["image"]
            target = sample["mask"]
            if self.args.cuda:
                image, target = image.cuda().float(), target.cuda().float()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r', ncols=80)
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['mask']
            if self.args.cuda:
                image, target = image.cuda().float(), target.cuda().float()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU, IoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()

        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}, IoU:{}".format(Acc, Acc_class, mIoU, FWIoU, IoU))
        print('Loss: %.3f' % test_loss)
        new_pred = mIoU
        is_best = False

        state_dict = self.model.state_dict()
        self.saver.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': state_dict,
            'optimizer': self.optimizer.state_dict(),
            'best_pred': self.best_pred,
        }, is_best, 'current_checkpoint.pth.tar')

        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            if torch.cuda.device_count() > 1:
                state_dict = self.model.module.state_dict()
            else:
                state_dict = self.model.state_dict()
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': state_dict,
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best, 'epoch{}_checkpoint.pth.tar'.format(str(epoch + 1)))

            self.test_model(epoch)
        self.saver.save_train_info(test_loss, epoch, Acc, mIoU, FWIoU, IoU, is_best)

    def test_model(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.test_loader, desc='\r', ncols=80)
        test_loss = 0.0

        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['mask']
            if self.args.cuda:
                image, target = image.cuda().float(), target.cuda().float()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU, IoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()

        print('Test:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}, IoU:{}".format(Acc, Acc_class, mIoU, FWIoU, IoU))
        print('Loss: %.3f' % test_loss)

        self.saver.save_test_info(test_loss, epoch, Acc, mIoU, FWIoU, IoU)

    def predict(self):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.test_loader, desc='\r', ncols=80)
        test_loss = 0.0

        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['mask']
            if self.args.cuda:
                image, target = image.cuda().float(), target.cuda().float()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            pred = np.squeeze(pred)
            # pred = padding_label(pred)
            lut = get_GID_lut()
            img = lut[pred]
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            name = sample["name"][0]

            squeeze_target = target.squeeze()
            img[squeeze_target == 255] = (0, 0, 0)

            self.saver.save_img(img, name)
            

        
def get_GID_lut():
    lut = np.zeros((512,3), dtype=np.uint8)
    lut[0] = [255,0,0]
    lut[1] =  [0,255,0]
    lut[2] =  [0,255,255]
    lut[3] =  [255,255,0]
    lut[4] =  [0,0,255]
    return lut

def padding_label(label):
    label[0, :] = label[1, :]
    label[-1, :] = label[-2, :]
    label[:, 0] = label[:, 1]
    label[:, -1] = label[:, -2]
    return label
    
