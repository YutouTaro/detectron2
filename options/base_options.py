import argparse
import os
import torch
from datetime import datetime
import pytz

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):

        # currently, some of the arguements are in train_options, they will be moved to base_options when the test_options is completed
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='the directory where all the projects are saved')
        self.parser.add_argument('--name', type=str, default='try', help='name of the project. The folder containing all the weights of the current project')
        self.parser.add_argument('--gpu_ids', nargs='+', type=int, default=0, help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain

        # str_ids = self.opt.gpu_ids.split(',')
        # self.opt.gpu_ids = []
        # for str_id in str_ids:
        #     id = int(str_id)
        #     if id >= 0:
        #         self.opt.gpu_ids.append(id)

        assert isinstance(self.opt.gpu_ids, list), print( 'type of "self.opt.gpu_ids" is {}'.format(type(self.opt.gpu_ids)))
        for id in self.opt.gpu_ids:
            if id < 0:
                self.opt.gpu_ids.remove(id)

        # set gpu ids
        if torch.cuda.is_available() and len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0]) # TODO: Usage of this function is discouraged in favor of device. In most cases it’s better to use CUDA_VISIBLE_DEVICES environmental variable.
            print('Using GPU {}'.format(self.opt.gpu_ids[0])) # TODO confirm whether one gpu is using
        else:
            print('Using CPU')
        args = vars(self.opt)

        #save the log to checkpoints_dir
        dir_log = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        if not os.path.exists(dir_log):
            os.makedirs(dir_log)
            print('folder {} created.'.format(dir_log))
        timenow = datetime.now(pytz.timezone('Asia/Singapore'))
        logfile_name = 'opt_' + self.opt.phase + timenow.strftime('_%y%m%d-%H%M%S') + '.txt'
        path_logfile = os.path.join(dir_log, logfile_name)
        with open(path_logfile, 'wt') as opt_file:
            str0 = 'Options'.center(30, '-')
            print(str0)
            opt_file.write(str0 + '\n')

            for k, v in sorted(args.items()):
                line = '{}: {}'.format(str(k), str(v))
                print(line)
                opt_file.write(line + '\n')

            str1 = 'End'.center(30, '-')
            print(str1)
            opt_file.write(str1 + '\n')

        return self.opt
