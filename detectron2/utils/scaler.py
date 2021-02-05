import numpy as np

class scaler():
    def __init__(self, data=None, scale_range=(-1,1), val_range=None):

        assert data is not None or val_range is not None, 'Error initializating util.scaler(): data and val_range are both none'

        # check scale_range is valid
        assert len(scale_range) == 2, 'scale_range should have 2 values, {} value(s) found'.format(len(scale_range))
        self.scale_range = scale_range
        self.scale_min, self.scale_max = np.min(scale_range), np.max(scale_range)
        assert not self.scale_max == self.scale_min, 'scale_range upper & lower boundary having equal value'

        if val_range is None: # calculate val_range from data
            self.val_min, self.val_max = np.min(data), np.max(data)
            self.val_range = (self.val_min, self.val_max)
        else:
            assert len(val_range) == 2, 'val_range should have 2 values, {} value(s) found'.format(len(val_range))
            self.val_min, self.val_max = np.min(val_range), np.max(val_range)
            assert not self.val_max == self.val_min, 'val_range upper & lower boundary having equal value {}'.format(self.val_min)
            self.val_range = val_range

    def scale(self, data):
        data_scaled = (data - self.val_min) / (self.val_max - self.val_min) * (self.scale_max - self.scale_min) + self.scale_min
        return data_scaled

    def inverse_scale(self, data):
        data_inverse_scaled = (data - self.scale_min) / (self.scale_max - self.scale_min) * (self.val_max - self.val_min) + self.val_min
        return data_inverse_scaled

        # if opt.isTrain and not opt.continue_train:
        #    # get data and calculate the position_scale
        # else:
        #     # read the position_scale from opt_train.txt


