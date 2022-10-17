import os
import numpy as np
import torch
from tqdm import tqdm
# model
from model.SearchNetStage1 import SearchNet1
from model.SearchNetStage2 import SearchNet2
from model.SearchNetStage3 import SearchNet3
from model.cell import ReLUConvBN, MixedCell
from utils.copy_state_dict import copy_state_dict
# data
from dataloaders import make_data_loader
# train
from utils.lr_scheduler import LR_Scheduler
from utils.loss import SegmentationLosses
from utils.saver import Saver
from utils.evaluator import Evaluator

class Trainer(object):
    def __init__(self, args):
        self.args = args
        
        # data
        kwargs = {'num_workers': args.num_workers, 'pin_memory': True, 'drop_last':True}
        self.train_loaderA, self.train_loaderB, self.val_loader, self.nclass = make_data_loader(args, **kwargs)
        
        # model
        torch.cuda.empty_cache()
        if self.args.model_name == 'DNAS':
            if self.args.search_stage == "first":
                layers = np.ones([args.layers, 4])
                connections = np.load(self.args.model_encode_path)
                model = SearchNet1(layers, 4, connections, ReLUConvBN, self.args.dataset, self.nclass)
            elif self.args.search_stage == "second":
                layers = np.ones([args.layers, 4])
                connections = np.load(self.args.model_encode_path)
                core_path = np.load(self.args.model_core_path).tolist()
                model = SearchNet2(layers, 4, connections, ReLUConvBN, self.args.dataset, self.nclass, core_path=core_path)
            elif self.args.search_stage == "third":
                layers = np.ones([args.layers, 4])
                connections = np.load(self.args.model_encode_path)
                print(connections)
                model = SearchNet3(layers, 4, connections, MixedCell, self.args.dataset, self.nclass)
        self.model = model
        if args.cuda:
            self.model = self.model.cuda()

        # train
        self.saver = Saver(args)
        self.criterion = SegmentationLosses(weight=None, cuda=args.cuda).build_loss(mode=args.loss_type)
        iters_per_epoch = len(self.train_loaderA)
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                      args.epochs, iters_per_epoch, min_lr=args.min_lr)
        self.optimizer = torch.optim.SGD(
            model.weight_parameters(),
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        self.architect_optimizer = torch.optim.Adam(self.model.arch_parameters(),
                                                    lr=args.arch_lr, betas=(0.9, 0.999),
                                                    weight_decay=args.arch_weight_decay)
        self.evaluator = Evaluator(self.nclass)

        # load
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
            if self.args.cuda:
                image, target = image.cuda().float(), target.cuda().float()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            if epoch >= self.args.alpha_epoch:
                search = next(iter(self.train_loaderB))
                image_search, target_search = search['image'], search['mask']
                if self.args.cuda:
                    image_search, target_search = image_search.cuda(), target_search.cuda()
                self.architect_optimizer.zero_grad()
                output_search = self.model(image_search)
                arch_loss = self.criterion(output_search, target_search)
                arch_loss.backward()
                self.architect_optimizer.step()

            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))

        if self.args.model_name == 'DNAS':
            if self.args.search_stage == "third":
                alphas = self.model.alphas.cpu().detach().numpy()
                alphas_dir = self.saver.experiment_dir + '/alphas' # TODO: root path
                if not os.path.exists(alphas_dir):
                    os.makedirs(alphas_dir)
                alphas_path = alphas_dir + '/alphas_{}.npy'.format(epoch)
                np.save(alphas_path, alphas, allow_pickle=True)
            else:
                betas = self.model.betas.cpu().detach().numpy()
                betas_dir = self.saver.experiment_dir + '/betas'
                if not os.path.exists(betas_dir):
                    os.makedirs(betas_dir)
                betas_path = betas_dir + '/betas_{}.npy'.format(epoch)
                np.save(betas_path, betas, allow_pickle=True)

        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)

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
