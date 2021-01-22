from base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')

        self.isTrain = False



if __name__ == '__main__':
    opt = TestOptions().parse()