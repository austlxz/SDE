import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

sys.path.append('..')

# os.environ["OMP_NUM_THREADS"] = str(1)
# os.environ["CUDA_VISIBLE_DEVICES"]='0'

# cudnn.benchmark = True
from mdistiller.models import cifar_model_dict, imagenet_model_dict
from mdistiller.distillers import distiller_dict
from mdistiller.dataset import get_dataset
from mdistiller.engine.utils import load_checkpoint, log_msg
from mdistiller.engine.cfg import CFG as cfg
from mdistiller.engine.cfg import show_cfg
from mdistiller.engine import trainer_dict

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision.models as tm


def init_distributed_mode(args):
    '''initialize DDP
    '''
    print("innnnnnnnnnnnnnnnn")
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        print(111111111111)
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        print(2222222222222)
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    elif hasattr(args, "rank"):
        print(3333333333333333)
        pass
    else:
        print("Not using distributed mode")
        args.distributed = False
        args.gpu = 0  # [修复点1] 单卡兜底：赋予默认 GPU ID 0，防止找不到属性
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(
        f"| distributed init (rank {args.rank}): {args.dist_url}, local rank:{args.gpu}, world size:{args.world_size}",
        flush=True)
    dist.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )


def main(cfg, resume, opts, distribution_arsg):
    experiment_name = cfg.EXPERIMENT.NAME
    if experiment_name == "":
        experiment_name = cfg.EXPERIMENT.TAG
    tags = cfg.EXPERIMENT.TAG.split(",")
    if opts:
        addtional_tags = ["{}:{}".format(k, v) for k, v in zip(opts[::2], opts[1::2])]
        tags += addtional_tags
        experiment_name += ",".join(addtional_tags)
    experiment_name = os.path.join(cfg.EXPERIMENT.PROJECT, experiment_name)

    if cfg.LOG.WANDB:
        try:
            import wandb
            wandb.init(project=cfg.EXPERIMENT.PROJECT, name=experiment_name, tags=tags)
        except:
            print(log_msg("Failed to use WANDB", "INFO"))
            cfg.LOG.WANDB = False

    # cfg & loggers
    show_cfg(cfg)

    init_distributed_mode(distribution_arsg)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # init dataloader & models
    train_loader, val_loader, num_data, num_classes = get_dataset(cfg)

    # vanilla
    if cfg.DISTILLER.TYPE == "NONE":
        if cfg.DATASET.TYPE == "imagenet":
            model_student = imagenet_model_dict[cfg.DISTILLER.STUDENT](pretrained=False)
        else:
            model_student = cifar_model_dict[cfg.DISTILLER.STUDENT][0](
                num_classes=num_classes
            )
        distiller = distiller_dict[cfg.DISTILLER.TYPE](model_student)
    # distillation
    else:
        print(log_msg("Loading teacher model", "INFO"))
        if cfg.DATASET.TYPE == "imagenet":
            model_teacher = imagenet_model_dict[cfg.DISTILLER.TEACHER](pretrained=True, M=cfg.M)
            model_student = imagenet_model_dict[cfg.DISTILLER.STUDENT](pretrained=False, M=cfg.M)
        else:
            net, pretrain_model_path = cifar_model_dict[cfg.DISTILLER.TEACHER]
            assert (
                    pretrain_model_path is not None
            ), "no pretrain model for teacher {}".format(cfg.DISTILLER.TEACHER)
            model_teacher = net(num_classes=num_classes, M=cfg.M)
            model_teacher.load_state_dict(load_checkpoint(pretrain_model_path)["model"])
            model_student = cifar_model_dict[cfg.DISTILLER.STUDENT][0](
                num_classes=num_classes, M=cfg.M
            )
        if cfg.DISTILLER.TYPE == "CRD":
            distiller = distiller_dict[cfg.DISTILLER.TYPE](
                model_student, model_teacher, cfg, num_data
            )
        else:
            distiller = distiller_dict[cfg.DISTILLER.TYPE](
                model_student, model_teacher, cfg
            )

    distiller = torch.nn.SyncBatchNorm.convert_sync_batchnorm(distiller)
    distiller = distiller.to(device)

    # [修复点2] 按需 DDP：判断如果处于 distributed 模式，才包裹 DDP
    if getattr(distribution_arsg, "distributed", False):
        distiller = DDP(distiller, device_ids=[distribution_arsg.gpu], find_unused_parameters=True)
    else:
        pass  # 单卡模式，模型已经 to(device) 了，不需要额外操作

    if cfg.DISTILLER.TYPE != "NONE":
        # [修复点3] 安全获取额外参数：兼容单卡没有 .module 的情况
        try:
            extra_params = distiller.module.get_extra_parameters()
        except AttributeError:
            extra_params = distiller.get_extra_parameters()

        print(
            log_msg(
                "Extra parameters of {}: {}\033[0m".format(
                    cfg.DISTILLER.TYPE, extra_params
                ),
                "INFO",
            )
        )

    # train
    trainer = trainer_dict[cfg.SOLVER.TRAINER](
        experiment_name, distiller, train_loader, val_loader, cfg
    )

    # 开始训练
    trainer.train(resume=resume)

    # =========================================================================
    # 新增逻辑：训练结束后，单独剥离并保存“纯净版”学生模型权重 (为 t-SNE 准备)
    # =========================================================================
    # 确保只在主进程 (Rank 0) 或者单卡模式下保存，防止多卡冲突
    is_main_process = (not getattr(distribution_arsg, "distributed", False)) or (distribution_arsg.rank == 0)

    if is_main_process:
        print(log_msg("Training finished. Extracting pure student model weights...", "INFO"))

        # 1. 兼容 DDP 获取底层的 distiller 模型
        base_distiller = distiller.module if hasattr(distiller, "module") else distiller

        # 2. 提取出纯净的学生模型
        if hasattr(base_distiller, "student"):
            pure_student = base_distiller.student
        else:
            pure_student = base_distiller  # Vanilla 模式可能自己就是 student

        # 3. 确定保存路径
        save_dir = cfg.LOG.PREFIX if cfg.LOG.PREFIX else "./output"
        os.makedirs(save_dir, exist_ok=True)
        student_save_path = os.path.join(save_dir, "pure_student_final.pth")

        # 4. 只保存 model 的 state_dict（格式对齐标准权重）
        torch.save({"model": pure_student.state_dict()}, student_save_path)

        print(log_msg(f"✅ Pure student model weights successfully saved to: {student_save_path}", "INFO"))
    # =========================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser("training for knowledge distillation.")
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--local_rank', type=int, help='local rank, will passed by ddp')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument("--M", default='[1,2,4]')

    parser.add_argument("--output_dir", default="", type=str, help="自定义的输出文件夹路径")

    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.local_rank = args.local_rank
    cfg.distributation = True
    cfg.M = args.M
    cfg.warmup = 1.0

    if args.output_dir:
        cfg.LOG.PREFIX = args.output_dir

    cfg.freeze()
    main(cfg, args.resume, args.opts, args)