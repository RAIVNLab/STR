import os
import pathlib
import random
import shutil
import time
import json

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter

from utils.logging import AverageMeter, ProgressMeter
from utils.net_utils import save_checkpoint, get_lr, LabelSmoothing
from utils.schedulers import get_policy
from utils.conv_type import STRConv
from utils.conv_type import sparseFunction

from args import args
from trainer import train, validate

import data
import models


def main():
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Simply call main_worker function
    main_worker(args)


def main_worker(args):
    args.gpu = None

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model and optimizer
    model = get_model(args)
    model = set_gpu(args, model)

    # Set up directories
    run_base_dir, ckpt_base_dir, log_base_dir = get_directories(args)

    # Loading pretrained model
    if args.pretrained:
        pretrained(args, model)

        # Saving a DenseConv (nn.Conv2d) compatible model 
        if args.dense_conv_model:    
            print(f"==> DenseConv compatible model, saving at {ckpt_base_dir / 'model_best.pth'}")
            save_checkpoint(
                {
                    "epoch": 0,
                    "arch": args.arch,
                    "state_dict": model.state_dict(),
                },
                True,
                filename=ckpt_base_dir / f"epoch_pretrained.state",
                save=True,
            )
            return

    optimizer = get_optimizer(args, model)
    data = get_dataset(args)
    lr_policy = get_policy(args.lr_policy)(optimizer, args)

    if args.label_smoothing is None:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = LabelSmoothing(smoothing=args.label_smoothing)

    # optionally resume from a checkpoint
    best_acc1 = 0.0
    best_acc5 = 0.0
    best_train_acc1 = 0.0
    best_train_acc5 = 0.0

    if args.resume:
        best_acc1 = resume(args, model, optimizer)

    # Evaulation of a model
    if args.evaluate:
        acc1, acc5 = validate(
            data.val_loader, model, criterion, args, writer=None, epoch=args.start_epoch
        )
        return

    writer = SummaryWriter(log_dir=log_base_dir)
    epoch_time = AverageMeter("epoch_time", ":.4f", write_avg=False)
    validation_time = AverageMeter("validation_time", ":.4f", write_avg=False)
    train_time = AverageMeter("train_time", ":.4f", write_avg=False)
    progress_overall = ProgressMeter(
        1, [epoch_time, validation_time, train_time], prefix="Overall Timing"
    )

    end_epoch = time.time()
    args.start_epoch = args.start_epoch or 0
    acc1 = None

    # Save the initial state
    save_checkpoint(
        {
            "epoch": 0,
            "arch": args.arch,
            "state_dict": model.state_dict(),
            "best_acc1": best_acc1,
            "best_acc5": best_acc5,
            "best_train_acc1": best_train_acc1,
            "best_train_acc5": best_train_acc5,
            "optimizer": optimizer.state_dict(),
            "curr_acc1": acc1 if acc1 else "Not evaluated",
        },
        False,
        filename=ckpt_base_dir / f"initial.state",
        save=False,
    )

    # Start training
    for epoch in range(args.start_epoch, args.epochs):
        lr_policy(epoch, iteration=None)
        cur_lr = get_lr(optimizer)

        # Gradual pruning in GMP experiments
        if args.conv_type == "GMPConv" and epoch >= args.init_prune_epoch and epoch <= args.final_prune_epoch:
            total_prune_epochs = args.final_prune_epoch - args.init_prune_epoch + 1
            for n, m in model.named_modules():
                if hasattr(m, 'set_curr_prune_rate'):
                    prune_decay = (1 - ((epoch - args.init_prune_epoch)/total_prune_epochs))**3
                    curr_prune_rate = m.prune_rate - (m.prune_rate*prune_decay)
                    m.set_curr_prune_rate(curr_prune_rate)

        # train for one epoch
        start_train = time.time()
        train_acc1, train_acc5 = train(
            data.train_loader, model, criterion, optimizer, epoch, args, writer=writer
        )
        train_time.update((time.time() - start_train) / 60)

        # evaluate on validation set
        start_validation = time.time()
        acc1, acc5 = validate(data.val_loader, model, criterion, args, writer, epoch)
        validation_time.update((time.time() - start_validation) / 60)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        best_acc5 = max(acc5, best_acc5)
        best_train_acc1 = max(train_acc1, best_train_acc1)
        best_train_acc5 = max(train_acc5, best_train_acc5)

        save = ((epoch % args.save_every) == 0) and args.save_every > 0
        if is_best or save or epoch == args.epochs - 1:
            if is_best:
                print(f"==> New best, saving at {ckpt_base_dir / 'model_best.pth'}")

            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "state_dict": model.state_dict(),
                    "best_acc1": best_acc1,
                    "best_acc5": best_acc5,
                    "best_train_acc1": best_train_acc1,
                    "best_train_acc5": best_train_acc5,
                    "optimizer": optimizer.state_dict(),
                    "curr_acc1": acc1,
                    "curr_acc5": acc5,
                },
                is_best,
                filename=ckpt_base_dir / f"epoch_{epoch}.state",
                save=save,
            )

        epoch_time.update((time.time() - end_epoch) / 60)
        progress_overall.display(epoch)
        progress_overall.write_to_tensorboard(
            writer, prefix="diagnostics", global_step=epoch
        )

        writer.add_scalar("test/lr", cur_lr, epoch)
        end_epoch = time.time()

        # Storing sparsity and threshold statistics for STRConv models
        if args.conv_type == "STRConv":
            count = 0
            sum_sparse = 0.0
            for n, m in model.named_modules():
                if isinstance(m, STRConv):
                    sparsity, total_params, thresh = m.getSparsity()
                    writer.add_scalar("sparsity/{}".format(n), sparsity, epoch)
                    writer.add_scalar("thresh/{}".format(n), thresh, epoch)
                    sum_sparse += int(((100 - sparsity) / 100) * total_params)
                    count += total_params
            total_sparsity = 100 - (100 * sum_sparse / count)
            writer.add_scalar("sparsity/total", total_sparsity, epoch)
        writer.add_scalar("test/lr", cur_lr, epoch)
        end_epoch = time.time()

    write_result_to_csv(
        best_acc1=best_acc1,
        best_acc5=best_acc5,
        best_train_acc1=best_train_acc1,
        best_train_acc5=best_train_acc5,
        prune_rate=args.prune_rate,
        curr_acc1=acc1,
        curr_acc5=acc5,
        base_config=args.config,
        name=args.name,
    )
    if args.conv_type == "STRConv":
        json_data = {}
        json_thres = {}
        for n, m in model.named_modules():
            if isinstance(m, STRConv):
                sparsity = m.getSparsity()
                json_data[n] = sparsity[0]
                sum_sparse += int(((100 - sparsity[0]) / 100) * sparsity[1])
                count += sparsity[1]
                json_thres[n] = sparsity[2]
        json_data["total"] = 100 - (100 * sum_sparse / count)
        if not os.path.exists("runs/layerwise_sparsity"):
            os.mkdir("runs/layerwise_sparsity")
        if not os.path.exists("runs/layerwise_threshold"):
            os.mkdir("runs/layerwise_threshold")
        with open("runs/layerwise_sparsity/{}.json".format(args.name), "w") as f:
            json.dump(json_data, f)
        with open("runs/layerwise_threshold/{}.json".format(args.name), "w") as f:
            json.dump(json_thres, f)


def set_gpu(args, model):
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        print(f"=> Parallelizing on {args.multigpu} gpus")
        torch.cuda.set_device(args.multigpu[0])
        args.gpu = args.multigpu[0]
        model = torch.nn.DataParallel(model, device_ids=args.multigpu).cuda(
            args.multigpu[0]
        )

    cudnn.benchmark = True

    return model


def resume(args, model, optimizer):
    if os.path.isfile(args.resume):
        print(f"=> Loading checkpoint '{args.resume}'")

        checkpoint = torch.load(args.resume)
        if args.start_epoch is None:
            print(f"=> Setting new start epoch at {checkpoint['epoch']}")
            args.start_epoch = checkpoint["epoch"]

        best_acc1 = checkpoint["best_acc1"]

        model.load_state_dict(checkpoint["state_dict"])

        optimizer.load_state_dict(checkpoint["optimizer"])

        print(f"=> Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")

        return best_acc1
    else:
        print(f"=> No checkpoint found at '{args.resume}'")


def pretrained(args, model):
    if os.path.isfile(args.pretrained):
        print("=> loading pretrained weights from '{}'".format(args.pretrained))
        pretrained = torch.load(
            args.pretrained,
            map_location=torch.device("cuda:{}".format(args.multigpu[0])),
        )["state_dict"]

        model_state_dict = model.state_dict()

        if not args.ignore_pretrained_weights:

            pretrained_final = {
                k: v
                for k, v in pretrained.items()
                if (k in model_state_dict and v.size() == model_state_dict[k].size())
            }

            if args.conv_type != "STRConv":
                for k, v in pretrained.items():
                    if 'sparseThreshold' in k:
                        wkey = k.split('sparse')[0] + 'weight'
                        weight = pretrained[wkey]
                        pretrained_final[wkey] = sparseFunction(weight, v)

            model_state_dict.update(pretrained_final)
            model.load_state_dict(model_state_dict)

        # Using the budgets of STR models for other models like DNW and GMP
        if args.use_budget:
            budget = {}
            for k, v in pretrained.items():
                if 'sparseThreshold' in k:
                    wkey = k.split('sparse')[0] + 'weight'
                    weight = pretrained[wkey]
                    sparse_weight = sparseFunction(weight, v)
                    budget[wkey] = (sparse_weight.abs() > 0).float().mean().item()

            for n, m in model.named_modules():
                if hasattr(m, 'set_prune_rate'):
                    pr = 1 - budget[n + '.weight']
                    m.set_prune_rate(pr)
                    print('set prune rate', n, pr)


    else:
        print("=> no pretrained weights found at '{}'".format(args.pretrained))


def get_dataset(args):
    print(f"=> Getting {args.set} dataset")
    dataset = getattr(data, args.set)(args)

    return dataset


def get_model(args):
    if args.first_layer_dense:
        args.first_layer_type = "DenseConv"

    print("=> Creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()

    print(f"=> Num model params {sum(p.numel() for p in model.parameters())}")

    # applying sparsity to the network
    if args.conv_type != "DenseConv":

        print(f"==> Setting prune rate of network to {args.prune_rate}")

        def _sparsity(m):
            if hasattr(m, "set_prune_rate"):
                m.set_prune_rate(args.prune_rate)

        model.apply(_sparsity)

    # freezing the weights if we are only doing mask training
    if args.freeze_weights:
        print(f"=> Freezing model weights")

        def _freeze(m):
            if hasattr(m, "mask"):
                m.weight.requires_grad = False
                if hasattr(m, "bias") and m.bias is not None:
                    m.bias.requires_grad = False

        model.apply(_freeze)

    return model


def get_optimizer(args, model):
    for n, v in model.named_parameters():
        if v.requires_grad:
            pass #print("<DEBUG> gradient to", n)

        if not v.requires_grad:
            pass #print("<DEBUG> no gradient to", n)

    if args.optimizer == "sgd":
        parameters = list(model.named_parameters())
        sparse_thresh = [v for n, v in parameters if ("sparseThreshold" in n) and v.requires_grad]
        bn_params = [v for n, v in parameters if ("bn" in n) and v.requires_grad]
        # rest_params = [v for n, v in parameters if ("bn" not in n) and ('sparseThreshold' not in n) and v.requires_grad]
        rest_params = [v for n, v in parameters if ("bn" not in n) and ("sparseThreshold" not in n) and v.requires_grad]
        optimizer = torch.optim.SGD(
            [
                {
                    "params": bn_params,
                    "weight_decay": 0 if args.no_bn_decay else args.weight_decay,
                },
                {
                    "params": sparse_thresh,
                    "weight_decay": args.st_decay if args.st_decay is not None else args.weight_decay,
                },
                {"params": rest_params, "weight_decay": args.weight_decay},
            ],
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
        )

    return optimizer


def _run_dir_exists(run_base_dir):
    log_base_dir = run_base_dir / "logs"
    ckpt_base_dir = run_base_dir / "checkpoints"

    return log_base_dir.exists() or ckpt_base_dir.exists()


def get_directories(args):
    if args.config is None or args.name is None:
        raise ValueError("Must have name and config")

    config = pathlib.Path(args.config).stem
    if args.log_dir is None:
        run_base_dir = pathlib.Path(
            f"runs/{config}/{args.name}/prune_rate={args.prune_rate}"
        )
    else:
        run_base_dir = pathlib.Path(
            f"{args.log_dir}/{config}/{args.name}/prune_rate={args.prune_rate}"
        )
    if args.width_mult != 1.0:
        run_base_dir = run_base_dir / "width_mult={}".format(str(args.width_mult))

    if _run_dir_exists(run_base_dir):
        rep_count = 0
        while _run_dir_exists(run_base_dir / str(rep_count)):
            rep_count += 1

        run_base_dir = run_base_dir / str(rep_count)

    log_base_dir = run_base_dir / "logs"
    ckpt_base_dir = run_base_dir / "checkpoints"

    if not run_base_dir.exists():
        os.makedirs(run_base_dir)

    (run_base_dir / "settings.txt").write_text(str(args))

    return run_base_dir, ckpt_base_dir, log_base_dir


def write_result_to_csv(**kwargs):
    results = pathlib.Path("runs") / "results.csv"

    if not results.exists():
        results.write_text(
            "Date Finished, "
            "Base Config, "
            "Name, "
            "Prune Rate, "
            "Current Val Top 1, "
            "Current Val Top 5, "
            "Best Val Top 1, "
            "Best Val Top 5, "
            "Best Train Top 1, "
            "Best Train Top 5\n"
        )

    now = time.strftime("%m-%d-%y_%H:%M:%S")

    with open(results, "a+") as f:
        f.write(
            (
                "{now}, "
                "{base_config}, "
                "{name}, "
                "{prune_rate}, "
                "{curr_acc1:.02f}, "
                "{curr_acc5:.02f}, "
                "{best_acc1:.02f}, "
                "{best_acc5:.02f}, "
                "{best_train_acc1:.02f}, "
                "{best_train_acc5:.02f}\n"
            ).format(now=now, **kwargs)
        )


if __name__ == "__main__":
    main()
