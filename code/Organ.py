from interface import Interface, implements

class Organ(Interface):
    def prepare_train(self, in_dir, pixdim=(1.0, 1.0, 1.0), spatial_size=[128, 128, 64], cache=True):
        pass

    def prepare_test(self, in_dir, pixdim=(1.0, 1.0, 1.0), spatial_size=[128, 128, 64]):
        pass
