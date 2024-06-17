from .dataset import RINDataset
from torch.utils.data import DataLoader


def get_traindataset(args):
    return RINDataset(args, mode='train')


def get_trainloader(dataset, args):
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle)


def get_testdataset(args):
    return RINDataset(args, mode='test')


def get_testloader(dataset, args):
    return DataLoader(dataset, batch_size=1, shuffle=False)


def get_dataset(args, mode):
    if mode == 'train':
        return get_traindataset(args)
    elif mode == 'test':
        return get_testdataset(args)
    else:
        raise ValueError("Unknown mode: {}".format(mode))


def get_loader(dataset, args, mode):
    if mode == 'train':
        return get_trainloader(dataset, args)
    elif mode == 'test':
        return get_testloader(dataset, args)
    else:
        raise ValueError("Unknown mode: {}".format(mode))
