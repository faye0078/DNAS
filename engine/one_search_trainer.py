# 1. path 2. operations 3.connections

import os
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch
from decode.first_decoder import Decoder as Decoder_1
from decode.one_loop_decoder import Decoder_2, get_connections_2
from decode.second_decoder import Decoder as Decoder_3
from model.create_model_encode import second_connect, third_connect
import sys
sys.path.append("./apex")
sys.path.append("..")
from search.loss import SegmentationLosses
from dataloaders import make_data_loader
from search.lr_scheduler import LR_Scheduler
from search.saver import Saver
from search.evaluator import Evaluator
from model.model_loop.SearchNetStage1 import SearchNet1
from model.model_loop.SearchNetStage2 import SearchNet2
from model.model_loop.SearchNetStage3 import SearchNet3
from search.copy_state_dict import copy_state_dict
from model.cell import MixedCell, MixedRetrainCell
try:
    from apex import amp

    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False


class Trainer(object):
    def __init__(self, args):
        self.args = args

        # 定义保存
        self.saver = Saver(args)
        self.saver.save_experiment_config()

        # 使用amp
        self.use_amp = True if (APEX_AVAILABLE and args.use_amp) else False
        self.opt_level = args.opt_level

        # 定义dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True, 'drop_last': True}
        self.train_loaderA, self.train_loaderB, self.val_loader, self.nclass = make_data_loader(args, **kwargs)
        self.evaluator = Evaluator(self.nclass)

        self.connections_1 = np.load(self.args.model_encode_path)
        self.connections_2 = None
        self.connections_3 = None
        self.core_path = None
        self.cell_arch_1 = np.load(self.args.model_cell_arch)
        self.tem_cell_arch_1 = None

        self.loops = 0

        self.criterion = SegmentationLosses(weight=None, cuda=args.cuda).build_loss(mode=args.loss_type)

        torch.cuda.empty_cache()

    def training_stage1(self, epochs):
        self.loops += 1
        print(self.cell_arch_1.sum(axis=-1))

        arch_path_dir = '/media/dell/DATA/wy/Seg_NAS/' + self.saver.experiment_dir + '/cell_arch'
        if not os.path.exists(arch_path_dir):
            os.makedirs(arch_path_dir)
        arch_path = arch_path_dir + '/{}_cell_arch_sum.npy'.format(str(self.loops))
        np.save(arch_path, self.cell_arch_1.sum(axis=-1))

        layers = np.ones([14, 4])
        model = SearchNet1(layers, 4, self.connections_1, self.cell_arch_1, MixedRetrainCell, self.args.dataset, self.nclass)

        optimizer = torch.optim.SGD(
            model.weight_parameters(),
            self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay
        )

        architect_optimizer = torch.optim.Adam(model.arch_parameters(),
                                                    lr=self.args.arch_lr, betas=(0.9, 0.999),
                                                    weight_decay=self.args.arch_weight_decay)
        # Define Evaluator

        # Define lr scheduler
        scheduler = LR_Scheduler(self.args.lr_scheduler, self.args.lr,
                                      self.args.epochs, 1000, min_lr=self.args.min_lr)

        if self.args.cuda:
            model = model.cuda()

        # 使用apex支持混合精度分布式训练
        if self.use_amp and self.args.cuda:
            keep_batchnorm_fp32 = True if (self.opt_level == 'O2' or self.opt_level == 'O3') else None

            model, [optimizer, architect_optimizer] = amp.initialize(
                model, [optimizer, architect_optimizer], opt_level=self.opt_level,
                keep_batchnorm_fp32=keep_batchnorm_fp32, loss_scale="dynamic")

            print('cuda finished')

        # 加载模型
        self.best_pred = 0.0
        if self.args.resume is not None:
            if not os.path.isfile(self.args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(self.args.resume))
            checkpoint = torch.load(self.args.resume)
            self.start_epoch = 0
            model.load_state_dict(checkpoint['state_dict'])
            # copy_state_dict(optimizer.state_dict(), checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(self.args.resume, checkpoint['epoch']))
        else:
            self.start_epoch = 0

        for epoch in range(self.start_epoch, epochs):
            train_loss = 0.0
            model.train()
            tbar = tqdm(self.train_loaderA, ncols=80)

            for i, sample in enumerate(tbar):
                image = sample["image"]
                target = sample["mask"]
                # image, target = sample['image'], sample['label']
                print(target.min())
                if self.args.cuda:
                    image, target = image.cuda().float(), target.cuda().float()
                scheduler(optimizer, i, epoch+1, self.best_pred)
                optimizer.zero_grad()
                output = model(image)
                loss = self.criterion(output, target)
                if self.use_amp:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                optimizer.step()

                if epoch+1 >= self.args.alpha_epoch:
                    # if True:
                    search = next(iter(self.train_loaderB))
                    image_search, target_search = search['image'], search['mask']
                    if self.args.cuda:
                        image_search, target_search = image_search.cuda().float(), target_search.cuda().float()

                    architect_optimizer.zero_grad()
                    output_search = model(image_search)
                    arch_loss = self.criterion(output_search, target_search)
                    if self.use_amp:
                        with amp.scale_loss(arch_loss, architect_optimizer) as arch_scaled_loss:
                            arch_scaled_loss.backward()
                    else:
                        arch_loss.backward()
                    architect_optimizer.step()

                # for name, module in model.named_modules():
                #     if 'sobel_operator.filter' in name:
                #         tem = module.weight.abs().squeeze()
                #         g1 = (tem[0][0] + tem[2][2]) / 2
                #         g2 = (tem[0][1] + tem[2][1]) / 2
                #         g3 = (tem[0][2] + tem[2][0]) / 2
                #         g4 = (tem[1][2] + tem[1][0]) / 2
                #         G = torch.tensor([[g1, g2, -g3], [g4, 0.0, -g4], [g3, -g2, -g1]])
                #         G = G.unsqueeze(0).unsqueeze(0)
                #         module.weight.data = nn.Parameter(G).cuda()

                train_loss += loss.item()
                # print(self.model.cells[0][0]['[-1  0]']._ops['sobel_operator'].filter.weight)
                tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))

            print('[Epoch: %d, numImages: %5d]' % (epoch+1, i * self.args.batch_size + image.data.shape[0]))
            print('Loss: %.3f' % train_loss)

            self.validation(epoch, model, 'stage1')

        self.connections_2 = get_connections_2(self.core_path)

    def training_stage2(self, epochs):
        layers = np.ones([14, 4])
        self.connections_2 = get_connections_2(self.core_path)
        model = SearchNet2(layers, 4, self.connections_2, self.cell_arch_1, MixedCell, self.args.dataset, self.nclass)

        for name, module in model.named_modules():
            if 'filter' in name:
                print(module)
        optimizer = torch.optim.SGD(
            model.weight_parameters(),
            self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay
        )

        architect_optimizer = torch.optim.Adam(model.arch_parameters(),
                                                    lr=self.args.arch_lr, betas=(0.9, 0.999),
                                                    weight_decay=self.args.arch_weight_decay)
        # Define Evaluator

        # Define lr scheduler
        scheduler = LR_Scheduler(self.args.lr_scheduler, self.args.lr,
                                      self.args.epochs, 1000, min_lr=self.args.min_lr)

        if self.args.cuda:
            model = model.cuda()

        # 使用apex支持混合精度分布式训练
        if self.use_amp and self.args.cuda:
            keep_batchnorm_fp32 = True if (self.opt_level == 'O2' or self.opt_level == 'O3') else None

            model, [optimizer, architect_optimizer] = amp.initialize(
                model, [optimizer, architect_optimizer], opt_level=self.opt_level,
                keep_batchnorm_fp32=keep_batchnorm_fp32, loss_scale="dynamic")

            print('cuda finished')

        # 加载模型
        self.best_pred = 0.0
        if self.args.resume is not None:
            if not os.path.isfile(self.args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(self.args.resume))
            checkpoint = torch.load(self.args.resume)
            self.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            copy_state_dict(optimizer.state_dict(), checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(self.args.resume, checkpoint['epoch']))
        else:
            self.start_epoch = 0

        for epoch in range(epochs):
            train_loss = 0.0
            model.train()
            tbar = tqdm(self.train_loaderA, ncols=80)

            for i, sample in enumerate(tbar):
                image = sample["image"]
                target = sample["mask"]
                # image, target = sample['image'], sample['label']
                if self.args.cuda:
                    image, target = image.cuda().float(), target.cuda().float()
                scheduler(optimizer, i, epoch+1, self.best_pred)
                optimizer.zero_grad()
                output = model(image)
                loss = self.criterion(output, target)
                if self.use_amp:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                optimizer.step()

                if epoch+1 >= self.args.alpha_epoch:
                    # if True:
                    search = next(iter(self.train_loaderB))
                    image_search, target_search = search['image'], search['mask']
                    if self.args.cuda:
                        image_search, target_search = image_search.cuda().float(), target_search.cuda().float()

                    architect_optimizer.zero_grad()
                    output_search = model(image_search)
                    arch_loss = self.criterion(output_search, target_search)
                    if self.use_amp:
                        with amp.scale_loss(arch_loss, architect_optimizer) as arch_scaled_loss:
                            arch_scaled_loss.backward()
                    else:
                        arch_loss.backward()
                    architect_optimizer.step()

                # for name, module in model.named_modules():
                #     if 'sobel_operator.filter' in name:
                #         tem = module.weight.abs().squeeze()
                #         g1 = (tem[0][0] + tem[2][2]) / 2
                #         g2 = (tem[0][1] + tem[2][1]) / 2
                #         g3 = (tem[0][2] + tem[2][0]) / 2
                #         g4 = (tem[1][2] + tem[1][0]) / 2
                #         G = torch.tensor([[g1, g2, -g3], [g4, 0.0, -g4], [g3, -g2, -g1]])
                #         G = G.unsqueeze(0).unsqueeze(0)
                #         module.weight.data = nn.Parameter(G).cuda()

                train_loss += loss.item()
                # print(self.model.cells[0][0]['[-1  0]']._ops['sobel_operator'].filter.weight)
                tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))

            print('[Epoch: %d, numImages: %5d]' % (epoch+1, i * self.args.batch_size + image.data.shape[0]))
            print('Loss: %.3f' % train_loss)

            self.validation(epoch, model, 'stage2')

        self.cell_arch_1 = self.tem_cell_arch_1.copy()

    def training_stage3(self, epochs):
        self.connections_3 = second_connect(14, 4, self.core_path)
        layers = np.ones([14, 4])
        model = SearchNet3(layers, 4, self.connections_3, self.cell_arch_1, MixedRetrainCell, self.args.dataset, self.nclass, core_path=self.core_path.tolist())

        optimizer = torch.optim.SGD(
            model.weight_parameters(),
            self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay
        )

        architect_optimizer = torch.optim.Adam(model.arch_parameters(),
                                                    lr=self.args.arch_lr, betas=(0.9, 0.999),
                                                    weight_decay=self.args.arch_weight_decay)
        # Define Evaluator

        # Define lr scheduler
        scheduler = LR_Scheduler(self.args.lr_scheduler, self.args.lr,
                                      self.args.epochs, 1000, min_lr=self.args.min_lr)

        if self.args.cuda:
            model = model.cuda()

        # 使用apex支持混合精度分布式训练
        if self.use_amp and self.args.cuda:
            keep_batchnorm_fp32 = True if (self.opt_level == 'O2' or self.opt_level == 'O3') else None

            model, [optimizer, architect_optimizer] = amp.initialize(
                model, [optimizer, architect_optimizer], opt_level=self.opt_level,
                keep_batchnorm_fp32=keep_batchnorm_fp32, loss_scale="dynamic")

            print('cuda finished')

        # 加载模型
        self.best_pred = 0.0
        if self.args.resume is not None:
            if not os.path.isfile(self.args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(self.args.resume))
            checkpoint = torch.load(self.args.resume)
            self.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            copy_state_dict(optimizer.state_dict(), checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(self.args.resume, checkpoint['epoch']))
        else:
            self.start_epoch = 0

        for epoch in range(epochs):
            train_loss = 0.0
            model.train()
            tbar = tqdm(self.train_loaderA, ncols=80)

            for i, sample in enumerate(tbar):
                image = sample["image"]
                target = sample["mask"]
                # image, target = sample['image'], sample['label']
                if self.args.cuda:
                    image, target = image.cuda().float(), target.cuda().float()
                scheduler(optimizer, i, epoch+1, self.best_pred)
                optimizer.zero_grad()
                output = model(image)
                loss = self.criterion(output, target)
                if self.use_amp:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                optimizer.step()

                if epoch+1 >= self.args.alpha_epoch:
                    # if True:
                    search = next(iter(self.train_loaderB))
                    image_search, target_search = search['image'], search['mask']
                    if self.args.cuda:
                        image_search, target_search = image_search.cuda().float(), target_search.cuda().float()

                    architect_optimizer.zero_grad()
                    output_search = model(image_search)
                    arch_loss = self.criterion(output_search, target_search)
                    if self.use_amp:
                        with amp.scale_loss(arch_loss, architect_optimizer) as arch_scaled_loss:
                            arch_scaled_loss.backward()
                    else:
                        arch_loss.backward()
                    architect_optimizer.step()

                # for name, module in model.named_modules():
                #     if 'sobel_operator.filter' in name:
                #         tem = module.weight.abs().squeeze()
                #         g1 = (tem[0][0] + tem[2][2]) / 2
                #         g2 = (tem[0][1] + tem[2][1]) / 2
                #         g3 = (tem[0][2] + tem[2][0]) / 2
                #         g4 = (tem[1][2] + tem[1][0]) / 2
                #         G = torch.tensor([[g1, g2, -g3], [g4, 0.0, -g4], [g3, -g2, -g1]])
                #         G = G.unsqueeze(0).unsqueeze(0)
                #         module.weight.data = nn.Parameter(G).cuda()

                train_loss += loss.item()
                # print(self.model.cells[0][0]['[-1  0]']._ops['sobel_operator'].filter.weight)
                tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))

            print('[Epoch: %d, numImages: %5d]' % (epoch+1, i * self.args.batch_size + image.data.shape[0]))
            print('Loss: %.3f' % train_loss)

            self.validation(epoch, model, 'stage3')

    def validation(self, epoch, model, stage):
        model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r', ncols=80)
        test_loss = 0.0

        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['mask']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = model(image)

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
        print('[Epoch: %d, numImages: %5d]' % (epoch+1, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}, IoU:{}".format(Acc, Acc_class, mIoU, FWIoU, IoU))
        print('Loss: %.3f' % test_loss)
        new_pred = mIoU
        is_best = False

        if new_pred > self.best_pred or epoch + 1 % 10 == 0:
            is_best = True
            self.best_pred = new_pred
            if torch.cuda.device_count() > 1:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': state_dict,
                'best_pred': self.best_pred,
            }, is_best, '{}_{}_epoch{}_checkpoint.pth.tar'.format(str(self.loops), stage, str(epoch + 1)))

            if stage == 'stage1':
                betas = model.betas.cpu().detach().numpy()
                # get second connections
                decoder = Decoder_1(betas)
                self.core_path = decoder.viterbi_decode()
                core_path_dir = '/media/dell/DATA/wy/Seg_NAS/' + self.saver.experiment_dir + '/path'
                if not os.path.exists(core_path_dir):
                    os.makedirs(core_path_dir)
                core_path_path = core_path_dir + '/{}_core_path_epoch{}.npy'.format(str(self.loops), epoch+1)
                np.save(core_path_path, self.core_path, allow_pickle=True)
            elif stage == 'stage2':
                alphas = model.alphas.cpu().detach().numpy()
                # get the new first cell arch
                decoder = Decoder_2(alphas, self.core_path, self.cell_arch_1)
                self.tem_cell_arch_1 = decoder.get_n_arch()
                arch_path_dir = '/media/dell/DATA/wy/Seg_NAS/' + self.saver.experiment_dir + '/cell_arch'
                if not os.path.exists(arch_path_dir):
                    os.makedirs(arch_path_dir)
                arch_path = arch_path_dir + '/{}_cell_arch_epoch{}.npy'.format(str(self.loops), epoch+1)
                np.save(arch_path, self.tem_cell_arch_1, allow_pickle=True)
            elif stage == 'stage3':
                betas = model.betas.cpu().detach().numpy()
                core_path_num = np.zeros(len(self.core_path))
                for i in range(len(self.core_path)):
                    if i == 0:
                        continue
                    core_path_num[i] = core_path_num[i - 1] + self.core_path[i - 1] + 1
                decoder = Decoder_3(betas, self.core_path, core_path_num)
                used_betas = decoder.used_betas
                connections = third_connect(used_betas)
                connections_path_dir = '/media/dell/DATA/wy/Seg_NAS/' + self.saver.experiment_dir + '/connections'
                if not os.path.exists(connections_path_dir):
                    os.makedirs(connections_path_dir)
                connections_path = connections_path_dir + '/connections_epoch{}.npy'.format(epoch+1)
                np.save(connections_path, connections)

        file_name = '{}_{}_train_info.txt'.format(str(self.loops), stage)
        self.saver.save_loop_train_info(test_loss, epoch, Acc, mIoU, FWIoU, IoU, is_best, file_name)
