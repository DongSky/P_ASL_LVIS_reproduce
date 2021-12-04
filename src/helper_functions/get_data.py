import os
import torch
import torchvision.transforms as transforms
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from src.helper_functions.coco_simulation import simulate_coco
from randaugment import RandAugment
from src.helper_functions.helper_functions import CocoDetection, CutoutPIL, LVISDetection


def get_data(args):
    if args.debug_mode == 'on_farm':
        instances_path_val = os.path.join(args.data, 'annotations/instances_val2014.json')
        instances_path_train = os.path.join(args.data, 'annotations/instances_train2014.json')
    else:
        instances_path_val = os.path.join(args.metadata, 'val.json')
        instances_path_train = os.path.join(args.metadata, 'train.json')

    if args.debug_mode == 'on_farm':
        data_path_val   = f'{args.data}/val2014'    # args.data
        data_path_train = f'{args.data}/train2014'
    else:
        data_path_val   = f'{args.data}/'    # args.data
        data_path_train = f'{args.data}/'  # args.data

    val_dataset = CocoDetection(data_path_val,
                                instances_path_val,
                                transforms.Compose([
                                    transforms.Resize((args.image_size, args.image_size)),
                                    transforms.ToTensor(),
                                    # normalize, # no need, toTensor does normalization
                                ]))
    train_dataset = CocoDetection(data_path_train,
                                  instances_path_train,
                                  transforms.Compose([
                                      transforms.Resize((args.image_size, args.image_size)),
                                      CutoutPIL(cutout_factor=0.5),
                                      RandAugment(),
                                      transforms.ToTensor(),
                                      # normalize,
                                  ]))
    print("len(val_dataset)): ", len(val_dataset))
    print("len(train_dataset)): ", len(train_dataset))

    # Obtain class
    classes = {i: c["name"] for i, c in enumerate(train_dataset.coco.cats.values())}
    train_dataset.classes, val_dataset.classes = classes, classes

    # Simulate partial annotation
    if args.simulate_partial_type is not None:
        train_dataset = simulate_coco(args, train_dataset)

    # Pytorch Data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    return train_loader, val_loader

def get_data_lvis(args):
    if args.debug_mode == 'on_farm':
        instances_path_val = os.path.join(args.data, 'annotations/lvis_v1_val.json')
        instances_path_train = os.path.join(args.data, 'annotations/lvis_v1_train.json')
    else:
        instances_path_val = os.path.join(args.metadata, 'val.json')
        instances_path_train = os.path.join(args.metadata, 'train.json')

    if args.debug_mode == 'on_farm':
        data_path_val   = f'{args.data}'    # args.data
        data_path_train = f'{args.data}'
    else:
        data_path_val   = f'{args.data}/'    # args.data
        data_path_train = f'{args.data}/'  # args.data

    val_dataset = LVISDetection(data_path_val,
                                instances_path_val,
                                transforms.Compose([
                                    transforms.Resize((args.image_size, args.image_size)),
                                    transforms.ToTensor(),
                                    # normalize, # no need, toTensor does normalization
                                ]))
    train_dataset = LVISDetection(data_path_train,
                                  instances_path_train,
                                  transforms.Compose([
                                      transforms.Resize((args.image_size, args.image_size)),
                                      CutoutPIL(cutout_factor=0.5),
                                      RandAugment(),
                                      transforms.ToTensor(),
                                      # normalize,
                                  ]))
    print("len(val_dataset)): ", len(val_dataset))
    print("len(train_dataset)): ", len(train_dataset))

    # Obtain class
    classes = {i: c["name"] for i, c in enumerate(train_dataset.coco.cats.values())}
    train_dataset.classes, val_dataset.classes = classes, classes

    # Simulate partial annotation
    #if args.simulate_partial_type is not None:
    #    train_dataset = simulate_coco(args, train_dataset)

    # Pytorch Data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    return train_loader, val_loader

def get_data_dist(args):
    if args.debug_mode == 'on_farm':
        instances_path_val = os.path.join(args.data, 'annotations/instances_val2014.json')
        instances_path_train = os.path.join(args.data, 'annotations/instances_train2014.json')
    else:
        instances_path_val = os.path.join(args.metadata, 'val.json')
        instances_path_train = os.path.join(args.metadata, 'train.json')

    if args.debug_mode == 'on_farm':
        data_path_val   = f'{args.data}/val2014'    # args.data
        data_path_train = f'{args.data}/train2014'
    else:
        data_path_val   = f'{args.data}/'    # args.data
        data_path_train = f'{args.data}/'  # args.data

    val_dataset = CocoDetection(data_path_val,
                                instances_path_val,
                                transforms.Compose([
                                    transforms.Resize((args.image_size, args.image_size)),
                                    transforms.ToTensor(),
                                    # normalize, # no need, toTensor does normalization
                                ]))
    train_dataset = CocoDetection(data_path_train,
                                  instances_path_train,
                                  transforms.Compose([
                                      transforms.Resize((args.image_size, args.image_size)),
                                      CutoutPIL(cutout_factor=0.5),
                                      RandAugment(),
                                      transforms.ToTensor(),
                                      # normalize,
                                  ]))
    print("len(val_dataset)): ", len(val_dataset))
    print("len(train_dataset)): ", len(train_dataset))

    # Obtain class
    classes = {i: c["name"] for i, c in enumerate(train_dataset.coco.cats.values())}
    train_dataset.classes, val_dataset.classes = classes, classes

    # Simulate partial annotation
    if args.simulate_partial_type is not None:
        train_dataset = simulate_coco(args, train_dataset)
    # PyTorch Distributed Sampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, drop_last=False)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    # Pytorch Data loader
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=args.batch_size, shuffle=True,
    #     num_workers=args.workers, pin_memory=True)

    # val_loader = torch.utils.data.DataLoader(
    #     val_dataset, batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.workers, pin_memory=False)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size // dist.get_world_size(), shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size // dist.get_world_size(), shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    return train_loader, val_loader

def get_data_dist_lvis(args):
    if args.debug_mode == 'on_farm':
        instances_path_val = os.path.join(args.data, 'annotations/lvis_v1_val.json')
        instances_path_train = os.path.join(args.data, 'annotations/lvis_v1_train.json')
    else:
        instances_path_val = os.path.join(args.metadata, 'val.json')
        instances_path_train = os.path.join(args.metadata, 'train.json')

    if args.debug_mode == 'on_farm':
        data_path_val   = f'{args.data}'    # args.data
        data_path_train = f'{args.data}'
    else:
        data_path_val   = f'{args.data}/'    # args.data
        data_path_train = f'{args.data}/'  # args.data

    val_dataset = LVISDetection(data_path_val,
                                instances_path_val,
                                transforms.Compose([
                                    transforms.Resize((args.image_size, args.image_size)),
                                    transforms.ToTensor(),
                                    # normalize, # no need, toTensor does normalization
                                ]))
    train_dataset = LVISDetection(data_path_train,
                                  instances_path_train,
                                  transforms.Compose([
                                      transforms.Resize((args.image_size, args.image_size)),
                                      CutoutPIL(cutout_factor=0.5),
                                      RandAugment(),
                                      transforms.ToTensor(),
                                      # normalize,
                                  ]))
    print("len(val_dataset)): ", len(val_dataset))
    print("len(train_dataset)): ", len(train_dataset))

    # Obtain class
    classes = {i: c["name"] for i, c in enumerate(train_dataset.coco.cats.values())}
    train_dataset.classes, val_dataset.classes = classes, classes

    # Simulate partial annotation
    # if args.simulate_partial_type is not None:
    #     train_dataset = simulate_coco(args, train_dataset)
    # PyTorch Distributed Sampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, drop_last=False)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    # Pytorch Data loader
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=args.batch_size, shuffle=True,
    #     num_workers=args.workers, pin_memory=True)

    # val_loader = torch.utils.data.DataLoader(
    #     val_dataset, batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.workers, pin_memory=False)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size // dist.get_world_size(), shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size // dist.get_world_size(), shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    return train_loader, val_loader
