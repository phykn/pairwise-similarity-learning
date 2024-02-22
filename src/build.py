from torch.utils.data import DataLoader
from .dataset import TrainDataset, TestDataset
from .model import Encoder, SimPLE
from .trainer import Trainer
from .misc import cycle


def build_dl(args):
    train_dataset = TrainDataset(
        root = args['data']['root'],
        unknown_label = args['data']['unknown_label'],
        img_size = args['data']['img_size']
    )
    test_dataset = TestDataset(
        root = args['data']['root'],
        unknown_label = args['data']['unknown_label'],
        img_size = args['data']['img_size']
    )

    train_dl = DataLoader(
        dataset = train_dataset,
        batch_size = args['dl']['batch_size'],
        shuffle = True,
        num_workers = args['dl']['num_workers'],
        pin_memory = args['dl']['pin_memory'],
        persistent_workers = args['dl']['persistent_workers'],
        drop_last = True
    )
    test_dl = DataLoader(
        dataset = test_dataset,
        batch_size = args['dl']['batch_size'],
        shuffle = False,
        num_workers = args['dl']['num_workers'],
        pin_memory = args['dl']['pin_memory'],
        persistent_workers = args['dl']['persistent_workers'],
        drop_last = False
    )
    
    return cycle(train_dl), test_dl


def build_model(args):
    encoder = Encoder(
        in_channels = args['model']['in_channels'],
        embed_dim = args['model']['embed_dim']
    )

    simple = SimPLE(
        b_theta = args['simple']['b_theta'],
        alpha = args['simple']['alpha'],
        r = args['simple']['r'],
        m = args['simple']['m'],
        lw = args['simple']['lw'],
        init_bias = args['simple']['init_bias']
    )

    return encoder, simple


def build_trainer(args, encoder, simple, dl):
    return Trainer(args, encoder, simple, dl)