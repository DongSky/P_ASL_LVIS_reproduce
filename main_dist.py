import os
import argparse
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from src.helper_functions.helper_functions import mAP, CocoDetection, CutoutPIL, ModelEma, add_weight_decay
from src.models import create_model
# from src.loss_functions.losses import AsymmetricLoss
from src.loss_functions.partial_asymmetric_loss import PartialSelectiveLoss, ComputePrior

from randaugment import RandAugment
from torch.cuda.amp import GradScaler, autocast
from src.helper_functions.coco_simulation import simulate_coco
from src.helper_functions.get_data import get_data, get_data_dist
import numpy as np
import torch.backends.cudnn as cudnn
import time

parser = argparse.ArgumentParser(description='PyTorch MS_COCO Training')
parser.add_argument('--data', metavar='DIR', help='path to dataset', default='./data')
parser.add_argument('--metadata', type=str, default='./data/COCO_2014')
parser.add_argument('--lr', default=2e-4, type=float)
parser.add_argument('--epochs', default=30, type=int)
parser.add_argument('--stop_epoch', default=None, type=int)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--model-name', default='resnet101')
parser.add_argument('--model-path', default=None, type=str)
parser.add_argument("--optimizer", type=str, default='adamw')
parser.add_argument('--num-classes', default=80)
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--image-size', default=448, type=int,
                    metavar='N', help='input image size (default: 448)')
parser.add_argument('--simulate_partial_type', type=str, default=None, help="options are fpc or rps")
parser.add_argument('--simulate_partial_param', type=float, default=1000)
parser.add_argument('--partial_loss_mode', type=str, default="negative")
parser.add_argument('--clip', type=float, default=0)
parser.add_argument('--gamma_pos', type=float, default=0)
parser.add_argument('--gamma_neg', type=float, default=1)
parser.add_argument('--gamma_unann', type=float, default=2)
parser.add_argument('--alpha_pos', type=float, default=1)
parser.add_argument('--alpha_neg', type=float, default=1)
parser.add_argument('--alpha_unann', type=float, default=1)
parser.add_argument('--likelihood_topk', type=int, default=5)
parser.add_argument('--prior_path', type=str, default=None)
parser.add_argument('--prior_threshold', type=float, default=0.05)
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--batch_size_eval', default=64, type=int, help="mini-batch for evaluation, smaller is better")
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 64)')
parser.add_argument('--path_dest', type=str, default="./outputs")
parser.add_argument('--debug_mode', type=str, default="on_farm")
parser.add_argument('--amp', help="using auto mix precision or not", action='store_true')
parser.add_argument('--ema_decay', type=float, default=0.9997, help='ema decay rate')

# distribution training
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='env://', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--seed', default=12345, type=int,
                    help='seed for initializing training. ')
parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')

def fix_random_seeds(seed=12345):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def main():

    # ---------------------------------------------------------------------------------
    # Preliminaries
    args = parser.parse_args()
    args.do_bottleneck_head = False
    if not os.path.exists(args.path_dest):
        os.makedirs(args.path_dest)
    # reset final lr
    # 2e-4 : 64 batch_size
    args.lr = args.lr * (args.batch_size / 64)
    # ---------------------------------------------------------------------------------
    # Distributed Training process
    if 'WORLD_SIZE' in os.environ:
        assert args.world_size > 0, 'please set --world-size and --rank in the command line'
        # launch by torch.distributed.launch
        # Single node
        #   python -m torch.distributed.launch --nproc_per_node=8 main.py --world-size 1 --rank 0 ...
        # Multi nodes
        #   python -m torch.distributed.launch --nproc_per_node=8 main.py --world-size 2 --rank 0 --dist-url 'tcp://IP_OF_NODE0:FREEPORT' ...
        #   python -m torch.distributed.launch --nproc_per_node=8 main.py --world-size 2 --rank 1 --dist-url 'tcp://IP_OF_NODE0:FREEPORT' ...
        local_world_size = int(os.environ['WORLD_SIZE'])
        args.world_size = args.world_size * local_world_size
        args.rank = args.rank * local_world_size + args.local_rank
        print('world size: {}, world rank: {}, local rank: {}'.format(args.world_size, args.rank, args.local_rank))
        print('os.environ:', os.environ)
    else:
        # single process, useful for debugging
        #   python main.py ...
        args.world_size = 1
        args.rank = 0
        args.local_rank = 0
    #--------------------------------------------------------------------------
    # random seeds configuration
    if args.seed is not None:
        fix_random_seeds(args.seed)
    # init dist
    torch.cuda.set_device(args.local_rank)
    print('| distributed init (local_rank {}): {}'.format(
        args.local_rank, args.dist_url), flush=True)
    #torch.distributed.init_process_group(backend='nccl', init_method=args.dist_url, 
    #                            world_size=args.world_size, rank=args.rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    cudnn.benchmark = True
    #---------------------------------------------------------------------------
    # Setup model
    #if dist.get_world_size() > 1:
    #    dist.barrier()
    print('creating model...')
    model = create_model(args).cuda()
    ema = ModelEma(model, args.ema_decay)  # 0.9997^641=0.82
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=False)
    if args.model_path:  # make sure to load pretrained ImageNet model
        state = torch.load(args.model_path, map_location='cpu')
        filtered_dict = {k: v for k, v in state['state_dict'].items() if
                         (k in model.state_dict() and 'head.fc' not in k)}
        model.load_state_dict(filtered_dict, strict=False)
    print('done\n')

    # ---------------------------------------------------------------------------------
    train_loader, val_loader = get_data_dist(args)

    # Actuall Training
    train_multi_label_coco(model, ema, train_loader, val_loader, args)


def train_multi_label_coco(model, ema, train_loader, val_loader, args):

    print("Used parameters:")
    print("Image_size:", args.image_size)
    print("Learning_rate:", args.lr)
    print("Epochs:", args.epochs)

    # ema = ModelEma(model, 0.9997)  # 0.9997^641=0.82
    prior = ComputePrior(train_loader.dataset.classes)

    # set optimizer
    Epochs = args.epochs
    if args.stop_epoch is not None:
        Stop_epoch = args.stop_epoch
    else:
        Stop_epoch = args.epochs
    weight_decay = args.weight_decay
    lr = args.lr

    criterion = PartialSelectiveLoss(args)
    # criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=False)
    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr, weight_decay=args.weight_decay, )
    elif args.optimizer == 'adam':
        parameters = add_weight_decay(model, weight_decay)
        optimizer = torch.optim.Adam(params=parameters, lr=lr, weight_decay=0)  # true wd, filter_bias_and_bn
    steps_per_epoch = len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=Epochs,
                                        pct_start=0.2)
    highest_mAP = 0
    trainInfoList = []
    scaler = GradScaler()
    # check validation
    #if args.rank == 0: print("check eval time")
    #eval_start_t = time.time_ns()
    #model.eval()
    #validate_multi(val_loader, model, ema, args)
    #model.train()
    #if args.rank == 0: print("eval time: ", (time.time_ns() - eval_start_t) / (10 ** 9), " seconds")
    for epoch in range(Epochs):
        epoch_start_t = time.time_ns()
        if epoch > Stop_epoch:
            break
        sampler_start_t = time.time_ns()
        train_loader.sampler.set_epoch(epoch)
        if args.rank == 0: print("sampler random time: ", (time.time_ns() - sampler_start_t) / (10 ** 9), " seconds")
        data_start_t = time.time_ns()
        for i, (inputData, target) in enumerate(train_loader):
            inputData = inputData.to(args.rank, non_blocking=True)
            # target = target.max(dim=1)[0]
            target = target.to(args.rank, non_blocking=True)
            data_ready_t = time.time_ns()
            if args.rank == 0 and i % args.print_freq == 0: print("data time: ", (data_ready_t - data_start_t) / (10 ** 9), " seconds")
            amp_forward_t = time.time_ns()
            with autocast(enabled=args.amp):  # mixed precision
                output = model(inputData).float()
            if args.rank == 0 and i % args.print_freq == 0: print("forward time: ", (time.time_ns() - amp_forward_t) / (10 ** 9), " seconds")
            loss = criterion(output, target)
            model.zero_grad()
            backward_start_t = time.time_ns()
            scaler.scale(loss).backward()
            # loss.backward()
            if args.rank == 0 and i % args.print_freq == 0: print("backward time: ", (time.time_ns() - backward_start_t) / (10 ** 9), " seconds")
            optimizer_start_t = time.time_ns()
            scaler.step(optimizer)
            scaler.update()
            # optimizer.step()

            scheduler.step()
            if args.rank == 0 and i % args.print_freq == 0: print("optimizer time: ", (time.time_ns() - optimizer_start_t) / (10 ** 9), " seconds")
            ema_start_t = time.time_ns()
            ema.update(model)
            if args.rank == 0 and i % args.print_freq == 0: print("ema update time: ", (time.time_ns() - ema_start_t) / (10 ** 9), " seconds")
            sync_start_t = time.time_ns()
            output_sum = [torch.zeros_like(output) for i in range(args.world_size)]
            dist.all_gather(output_sum, output)
            dist.barrier()
            if args.rank == 0 and i % args.print_freq == 0: print("sync time: ", (time.time_ns() - sync_start_t) / (10 ** 9), " seconds")
            prior.update(torch.cat(output_sum, dim=0))
            #prior.update(output)

            # store information
            if i % args.print_freq == 0:
                trainInfoList.append([epoch, i, loss.item()])
                if args.rank == 0:
                    print('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.1f}'
                      .format(epoch, Epochs, str(i).zfill(3), str(steps_per_epoch).zfill(3),
                              scheduler.get_last_lr()[0], \
                              loss.item()))
            if args.rank == 0 and i % args.print_freq == 0: print("iter time: ", (time.time_ns() - data_start_t) / (10 ** 9), " seconds")
            data_start_t = time.time_ns()

        # Report prior
        prior.save_prior()
        prior.get_top_freq_classes()

        # Save ckpt
        try:
            if args.rank == 0:
                ckpt = {
                    'state_dict': model.module.state_dict(),
                    'ema_state_dict': ema.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'arch': args.model_name,
                    'epoch': epoch
                }
                torch.save(ckpt, os.path.join(args.path_dest, 'model-{}-{}.ckpt'.format(epoch + 1, i + 1)))
                print("Model checkpoint saved successfully.")
            # torch.save(ema.state_dict(), os.path.join(args.path_dest, 'ema-{}-{}.ckpt'.format(epoch + 1, i + 1)))
            # print("Model EMA checkpoint saved successfully.")
        except:
            print("Saving model failed.")
        if args.rank == 0: print(f"epoch {epoch} training time: ", (time.time_ns() - epoch_start_t) / (10 ** 9), " seconds")
        eval_start_t = time.time_ns()
        model.eval()
        mAP_score = validate_multi(val_loader, model, ema, args)
        model.train()
        if args.rank == 0: print("eval time: ", (time.time_ns() - eval_start_t) / (10 ** 9), " seconds")
        if mAP_score > highest_mAP:
            highest_mAP = mAP_score
            try:
                if args.rank == 0:
                    ckpt = {
                        'state_dict': model.module.state_dict(),
                        'ema_state_dict': ema.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'arch': args.model_name,
                        'mAP_eval_online': mAP_score
                    }
                    torch.save(ckpt, os.path.join(
                        'models/', 'model-highest.ckpt'))
            except:
                pass
        print('current_mAP = {:.2f}, highest_mAP = {:.2f}\n'.format(mAP_score, highest_mAP))

@torch.no_grad()
def validate_multi(val_loader, model, ema_model, args):
    print("starting validation")
    Sig = torch.nn.Sigmoid()
    preds_regular = []
    preds_ema = []
    targets = []
    for i, (input, target) in enumerate(val_loader):
        target = target.to(args.rank, non_blocking=True)
        input = input.to(args.rank, non_blocking=True)
        # target = target.max(dim=1)[0]
        # compute output
        with torch.no_grad():
            with autocast(enabled=args.amp):
                output_regular = Sig(model(input))#.cpu()
                output_ema = Sig(ema_model.module(input))#.cpu()

        # for mAP calculation
        preds_regular.append(output_regular.detach())
        preds_ema.append(output_ema.detach())
        targets.append(target.detach())
    # if dist.get_world_size() > 1:
    #     dist.barrier()
    # if dist.get_rank() == 0:
    #     print("Calculating mAP:")
    #     filenamelist = ['saved_data_tmp.{}.txt'.format(ii) for ii in range(dist.get_world_size())]
    #     metric_func = mAP                
    #     mAP, aps = metric_func([os.path.join(args.output, _filename) for _filename in filenamelist], args.num_class, return_each=True)
        
    #     # logger.info("  mAP: {}".format(mAP))
    #     # logger.info("   aps: {}".format(np.array2string(aps, precision=5)))
    # else:
    #     mAP = 0
    targets = torch.cat(targets)
    preds_regular = torch.cat(preds_regular)
    preds_ema = torch.cat(preds_ema)
    targets_sum = [torch.zeros_like(targets) for i in range(args.world_size)]
    preds_regular_sum = [torch.zeros_like(preds_regular) for i in range(args.world_size)]
    preds_ema_sum = [torch.zeros_like(preds_ema) for i in range(args.world_size)]
    dist.all_gather(targets_sum, targets)
    dist.all_gather(preds_regular_sum, preds_regular)
    dist.all_gather(preds_ema_sum, preds_ema)
    if dist.get_world_size() > 1:
        dist.barrier()
    # if args.rank == 0:
    mAP_score_regular = mAP(torch.cat(targets_sum).cpu().numpy(), torch.cat(preds_regular_sum).cpu().numpy())
    mAP_score_ema = mAP(torch.cat(targets_sum).cpu().numpy(), torch.cat(preds_ema_sum).cpu().numpy())
    if args.rank == 0:
        print("mAP score regular {:.2f}, mAP score EMA {:.2f}".format(mAP_score_regular, mAP_score_ema))
    return max(mAP_score_regular, mAP_score_ema)


if __name__ == '__main__':
    main()
