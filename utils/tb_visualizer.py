import numpy as np
import os
import time
from tensorboardX import SummaryWriter
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from PIL import Image
import cv2

class TBVisualizer:
    def __init__(self, opt):
        self._opt = opt
        self._save_path = os.path.join(opt.checkpoints_dir, opt.name)

        self._log_path = os.path.join(self._save_path, 'loss_log.txt')
        self._tb_path = os.path.join(self._save_path, 'summary.json')
        self._writer = SummaryWriter(self._save_path)

        with open(self._log_path, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def __del__(self):
        self._writer.close()

    def display_current_results(self, visuals, i_epoch, is_train, save_visuals=False):
        for label, mesh in visuals.items():
            if 'name' in label:
                continue
            sum_name = '{}/{}'.format('Train' if is_train else 'Test', label)
            if save_visuals:
                sum_name = sum_name.split('/')
                sum_name[-1] = 'trimesh'
                sum_name = '/'.join(sum_name)
                save_path = os.path.join(self._opt.checkpoints_dir, self._opt.name, 'event_imgs', sum_name)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                
                if 'trimesh' in label:
                    mesh.export(os.path.join(save_path, '%s-%05d-%s.obj' % (label, i_epoch, visuals[label+'_name'])), include_color=True)
                elif 'pattern' in label:
                    cv2.imwrite(os.path.join(save_path, '%s-%05d.png' % (label, i_epoch)), mesh)
                elif 'img' in label:
                    cv2.imwrite(os.path.join(save_path, '%s-%05d.png' % (label, i_epoch)), mesh)

        self._writer.export_scalars_to_json(self._tb_path)


    def plot_losses(self, losses, iters_per_epoch, is_train):
        for label, loss in losses.items():
            if len(loss)==0:
                continue

            x = [(i+1)/iters_per_epoch for i in range(len(loss))]
            y = loss
            plt.plot(x, y, 'b')
            plt.axis([0, x[-1], 0, max(loss)])
            plt.ylabel(label + ' Loss')
            plt.xlabel('Iter')
            plt.title(label)
            sum_name = '{}/Loss'.format('Train' if is_train else 'Test')
            save_path = os.path.join(self._opt.checkpoints_dir, self._opt.name, 'event_imgs', sum_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(os.path.join(save_path, '%s.png' % label))
            plt.clf()

    def print_current_train_errors(self, epoch, i, iters_per_epoch, errors, t, visuals_were_stored):
        log_time = time.strftime("[%d/%m/%Y %H:%M:%S]")
        visuals_info = "v" if visuals_were_stored else ""
        message = '%s (T%s, epoch: %d, it: %d/%d, t/smpl: %.5fs) ' % (log_time, visuals_info, epoch, i, iters_per_epoch, t)
        for k, v in errors.items():
            if len(v)==0:
                continue
            message += '%s:%.5f ' % (k, v[-1])

        print(message)
        with open(self._log_path, "a") as log_file:
            log_file.write('%s\n' % message)

    def mkdir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
