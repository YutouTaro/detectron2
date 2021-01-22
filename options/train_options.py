from base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')

        self.isTrain = True



if __name__ == '__main__':
    opt = TrainOptions().parse()