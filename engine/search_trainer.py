import os
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch
from collections import OrderedDict
from torchstat import stat
import sys 
sys.path.append("./apex")

import sys 
sys.path.append("..")

try:
    from apex import amp
    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False
from search.loss import SegmentationLosses
from dataloaders import make_data_loader

from search.lr_scheduler import LR_Scheduler
from search.saver import Saver
from search.evaluator import Evaluator
from search.search_model import AutoDeeplab
from search.search_model_forward import AutoDeeplab_forward
from model.DCNAS import DCNASNet
from model.SearchNetStage1 import SearchNet1
from model.SearchNetStage2 import SearchNet2
from model.SearchNetStage3 import SearchNet3
from model.early_fusion.SearchNetStage1 import SearchNet1 as Fusion_SearchNet1
from model.early_fusion.SearchNetStage2 import SearchNet2 as Fusion_SearchNet2
from model.early_fusion.SearchNetStage3 import SearchNet3 as Fusion_SearchNet3
from search.copy_state_dict import copy_state_dict
from model.cell import ReLUConvBN, ReLUConv5BN, ConvBNReLU, MixedCell, DCNAS_cell, MixedCellMini, MixedRetrainCell

class Trainer(object):
    def __init__(self, args):
        self.args = args

        # 定义保存
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # 可视化
        # self.summary = TensorboardSummary(self.saver.experiment_dir)
        # self.writer = self.summary.create_summary()
        # 使用amp
        self.use_amp = True if (APEX_AVAILABLE and args.use_amp) else False
        self.opt_level = args.opt_level

        # 定义dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True, 'drop_last':True}
        self.train_loaderA, self.train_loaderB, self.val_loader, self.nclass = make_data_loader(args, **kwargs)


        self.criterion = SegmentationLosses(weight=None, cuda=args.cuda).build_loss(mode=args.loss_type)

        torch.cuda.empty_cache()
        # 定义网络
        if self.args.model_name == 'AutoDeeplab':
            if self.args.forward:
                model = AutoDeeplab_forward(self.nclass, 12, self.criterion, self.args.filter_multiplier,
                                 self.args.block_multiplier, self.args.step, self.args.dataset)
            else:
                model = AutoDeeplab(self.nclass, 12, self.criterion, self.args.filter_multiplier,
                                 self.args.block_multiplier, self.args.step, self.args.dataset)
        if self.args.model_name == 'RSNet':
            if self.args.forward:
                model = AutoDeeplab_forward(self.nclass, 7, self.criterion, self.args.filter_multiplier,
                                 self.args.block_multiplier, self.args.step, self.args.dataset)
            else:
                model = AutoDeeplab(self.nclass, 7, self.criterion, self.args.filter_multiplier,
                                 self.args.block_multiplier, self.args.step, self.args.dataset)
        if self.args.model_name == 'DCNAS':
            layers = np.ones([12, 4])
            connections = np.load(self.args.model_encode_path)
            model = DCNASNet(layers, 4, connections, DCNAS_cell, self.args.dataset, self.nclass)
        elif self.args.model_name == 'FlexiNet':
            if self.args.search_stage == "first":
                layers = np.ones([16, 4])
                connections = np.load(self.args.model_encode_path)
                model = SearchNet1(layers, 4, connections, ReLUConvBN, self.args.dataset, self.nclass)
                # cell_arch = np.load('/media/dell/DATA/wy/Seg_NAS/model/model_encode/uadataset/one_loop_14layers_mixedcell/init_cell_arch.npy')
                # model = Fusion_SearchNet1(layers, 4, connections, cell_arch, MixedRetrainCell, self.args.dataset, self.nclass)

            elif self.args.search_stage == "second":
                layers = np.ones([16, 4])
                connections = np.load(self.args.model_encode_path)
                core_path = np.load('/media/dell/DATA/wy/Seg_NAS/model/model_encode/core_path.npy').tolist()
                model = SearchNet2(layers, 4, connections, ReLUConvBN, self.args.dataset, self.nclass, core_path=core_path)
                # cell_arch = np.load('/media/dell/DATA/wy/Seg_NAS/model/model_encode/uadataset/one_loop_14layers_mixedcell/init_cell_arch.npy')
                # model = Fusion_SearchNet2(layers, 4, connections, cell_arch, MixedRetrainCell, self.args.dataset, self.nclass, core_path=core_path)

            elif self.args.search_stage == "third":
                layers = np.ones([16, 4])
                connections = np.load(self.args.model_encode_path)
                print(connections)
                # connections = 0
                model = SearchNet3(layers, 4, connections, MixedCell, self.args.dataset, self.nclass)
                # model = Fusion_SearchNet3(layers, 4, connections, MixedCell, self.args.dataset, self.nclass)

        # device = torch.device("cpu")
        # model = model.to(device)
        # stat(model, (4, 512, 512))

        optimizer = torch.optim.SGD(
            model.weight_parameters(),
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        self.model, self.optimizer = model, optimizer
        self.architect_optimizer = torch.optim.Adam(self.model.arch_parameters(),
                                                    lr=args.arch_lr, betas=(0.9, 0.999),
                                                    weight_decay=args.arch_weight_decay)
        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                      args.epochs, 1000, min_lr=args.min_lr)

        if args.cuda:
            self.model = self.model.cuda()


        # 使用apex支持混合精度分布式训练
        if self.use_amp and args.cuda:
            keep_batchnorm_fp32 = True if (self.opt_level == 'O2' or self.opt_level == 'O3') else None

            self.model, [self.optimizer, self.architect_optimizer] = amp.initialize(
                self.model, [self.optimizer, self.architect_optimizer], opt_level=self.opt_level,
                keep_batchnorm_fp32=keep_batchnorm_fp32, loss_scale="dynamic")

            print('cuda finished')


        # 加载模型
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            self.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            copy_state_dict(self.optimizer.state_dict(), checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            self.start_epoch = 0

    def training(self, epoch):
        train_loss = 0.0
        
        self.model.train()
        tbar = tqdm(self.train_loaderA, ncols=80)

        for i, sample in enumerate(tbar):
            image = sample["image"]
            target = sample["mask"]
            # image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda().float(), target.cuda().float()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.criterion(output, target)
            if self.use_amp:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            self.optimizer.step()

            if epoch >= self.args.alpha_epoch:
            # if True:
                search = next(iter(self.train_loaderB))
                image_search, target_search = search['image'], search['mask']
                if self.args.cuda:
                    image_search, target_search = image_search.cuda(), target_search.cuda()

                self.architect_optimizer.zero_grad()
                output_search = self.model(image_search)
                arch_loss = self.criterion(output_search, target_search)
                if self.use_amp:
                    with amp.scale_loss(arch_loss, self.architect_optimizer) as arch_scaled_loss:
                        arch_scaled_loss.backward()
                else:
                    arch_loss.backward()
                self.architect_optimizer.step()

            train_loss += loss.item()
            # print(self.model.cells[0][0]['[-1  0]']._ops['sobel_operator'].filter.weight)
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))

        if self.args.model_name == 'FlexiNet':
            if self.args.search_stage == "third":
                alphas = self.model.alphas.cpu().detach().numpy()
                alphas_dir = '/media/dell/DATA/wy/Seg_NAS/' + self.saver.experiment_dir + '/alphas'
                if not os.path.exists(alphas_dir):
                    os.makedirs(alphas_dir)
                alphas_path = alphas_dir + '/alphas_{}.npy'.format(epoch)
                np.save(alphas_path, alphas, allow_pickle=True)
            else:
                betas = self.model.betas.cpu().detach().numpy()
                betas_dir = '/media/dell/DATA/wy/Seg_NAS/' + self.saver.experiment_dir + '/betas'
                if not os.path.exists(betas_dir):
                    os.makedirs(betas_dir)
                betas_path = betas_dir + '/betas_{}.npy'.format(epoch)
                np.save(betas_path, betas, allow_pickle=True)
        elif self.args.model_name == 'AutoDeeplab' or self.args.model_name == 'RSNet':
            alphas = self.model.alphas.cpu().detach().numpy()
            alphas_dir = '/media/dell/DATA/wy/Seg_NAS/' + self.saver.experiment_dir + '/alphas'
            if not os.path.exists(alphas_dir):
                os.makedirs(alphas_dir)
            alphas_path = alphas_dir + '/alphas_{}.npy'.format(epoch)
            np.save(alphas_path, alphas, allow_pickle=True)

            betas = self.model.betas.cpu().detach().numpy()
            betas_dir = '/media/dell/DATA/wy/Seg_NAS/' + self.saver.experiment_dir + '/betas'
            if not os.path.exists(betas_dir):
                os.makedirs(betas_dir)
            betas_path = betas_dir + '/betas_{}.npy'.format(epoch)
            np.save(betas_path, betas, allow_pickle=True)


        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)

        if not self.args.val:
            # save checkpoint every epoch
            is_best = False
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

        state_dict = self.model.state_dict()
        self.saver.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': state_dict,
            'optimizer': self.optimizer.state_dict(),
            'best_pred': self.best_pred,
        }, is_best, 'current_checkpoint.pth.tar'.format(str(epoch + 1)))
        self.saver.save_train_info(test_loss, epoch, Acc, mIoU, FWIoU, IoU, is_best)
