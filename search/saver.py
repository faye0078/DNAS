import os
import shutil
import torch
from collections import OrderedDict
import glob
import torch.distributed as dist


class Saver(object):

    def __init__(self, args, use_dist=False):
        self.args = args
        self.use_dist = use_dist
        self.directory = os.path.join('run', args.dataset, args.checkname)
        self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')))
        run_id = max([int(x.split('_')[-1]) for x in self.runs]) + 1 if self.runs else 0

        self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        """Saves checkpoint to disk"""
        filename = os.path.join(self.experiment_dir, filename)
        if (self.use_dist and dist.get_rank() == 0) or not self.use_dist:
            torch.save(state, filename)
            if is_best:
                best_pred = state['best_pred']
                with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'w') as f:
                    f.write(str(best_pred))
                if self.runs:
                    previous_miou = [0.0]
                    for run in self.runs:
                        run_id = run.split('_')[-1]
                        path = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)), 'best_pred.txt')
                        if os.path.exists(path):
                            with open(path, 'r') as f:
                                miou = float(f.readline())
                                previous_miou.append(miou)
                        else:
                            continue
                    max_miou = max(previous_miou)
                    if best_pred > max_miou:
                        shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))
                else:
                    shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))

    def save_experiment_config(self):
        if (self.use_dist and dist.get_rank() == 0) or not self.use_dist:
            logfile = os.path.join(self.experiment_dir, 'parameters.txt')
            log_file = open(logfile, 'w')
            p = OrderedDict()
            p['datset'] = self.args.dataset
            p['batchsize'] = self.args.batch_size
            p['out_stride'] = self.args.out_stride
            p['lr'] = self.args.lr
            p['lr_scheduler'] = self.args.lr_scheduler
            p['loss_type'] = self.args.loss_type
            p['epoch'] = self.args.epochs
            p['resize'] = self.args.resize
            p['crop_size'] = self.args.crop_size

            for key, val in p.items():
                log_file.write(key + ':' + str(val) + '\n')
            log_file.close()

    def save_train_info(self, loss, epoch, acc, miou, fwiou, iou, is_best):
        train_info = 'loss:{}, epoch:{}, acc:{}, miou:{}, fwiou:{}, iou:{}'.format(str(loss), str(epoch+1), str(acc), str(miou), str(fwiou), str(iou))
        if is_best:
            train_info = train_info + 'new best'

        info_file = os.path.join(self.experiment_dir, 'train_info.txt')
        file = open(info_file, 'a')

        file.write(train_info + '\n')
        file.close()

    def save_loop_train_info(self, loss, epoch, acc, miou, fwiou, iou, is_best, file_name):
        train_info = 'loss:{}, epoch:{}, acc:{}, miou:{}, fwiou:{}, iou:{}'.format(str(loss), str(epoch+1), str(acc), str(miou), str(fwiou), str(iou))
        if is_best:
            train_info = train_info + 'new best'

        info_file = os.path.join(self.experiment_dir, file_name)
        file = open(info_file, 'a')

        file.write(train_info + '\n')
        file.close()
