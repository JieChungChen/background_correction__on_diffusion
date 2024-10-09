import argparse
import os
from tqdm import tqdm
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast
from data_preprocess import NanoCT_Dataset
from ddpm.model import Diffusion_UNet
from ddpm.diffusion_sr3 import GaussianDiffusionTrainer
from utils import check_distributed, model_eval, model_eval_for_val



def get_args_parser():
    parser = argparse.ArgumentParser('diffusion for background correction', add_help=False)
    parser.add_argument('--train', default=False, type=bool)
    parser.add_argument('--data_dir', default='./training_data_n', type=str)
    parser.add_argument('--model_save_dir', default='./checkpoints', type=str)
    parser.add_argument('--load_weight', default=False, type=bool)
    parser.add_argument('--use_mix_precision', default=False, type=bool)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--accumulate_step', default=1, type=int)
    parser.add_argument('--epoch', default=500, type=int)

    parser.add_argument('--model_name', default='DeRef_DDPM', type=str) 
    parser.add_argument('--checkpoint', default='ckpt_340.pt', type=str)                  

    parser.add_argument('--T', default=2000, type=float)
    parser.add_argument('--beta_sche', default='linear', type=str)
    parser.add_argument('--beta_1', default=1e-4, type=float)
    parser.add_argument('--beta_T', default=0.02, type=float)
    parser.add_argument('--img_size', default=128, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--grad_clip', default=1., type=float)
    return parser


def main(args):
    # multi-GPU settings
    rank, local_rank, world_size, is_distributed = check_distributed()
    print(check_distributed())
    if is_distributed:
        torch.cuda.set_device(local_rank)  # set current device
        device = torch.device("cuda", local_rank)
        dist.init_process_group("nccl")  # initialize process group and set the communication backend betweend GPUs
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # data settings
    os.makedirs(args.model_save_dir, exist_ok=True)
    model = Diffusion_UNet().to(device)
    dataset = NanoCT_Dataset(data_dir='./training_data_n', img_size=args.img_size)
    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
        train_sampler = DistributedSampler(dataset, shuffle=True)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, drop_last=True, pin_memory=True, sampler=train_sampler)
    else:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, drop_last=True, pin_memory=True)

    # model settings
    if args.load_weight:
        model.load_state_dict(torch.load(args.model_save_dir+'/'+args.checkpoint, map_location=device), strict=False)
        print("Model weight load down.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    trainer = GaussianDiffusionTrainer(model, args.beta_1, args.beta_T, args.T).to(device)

    for e in range(args.epoch):
        if is_distributed:
            dataloader.sampler.set_epoch(e)
        model.train()
        step = 0
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for obj_ref, ref in tqdmDataLoader:
                b = ref.shape[0]
                optimizer.zero_grad()
                condit, x_0 = obj_ref.to(device), ref.to(device) 
                loss = None   
                loss = trainer(condit, x_0).sum() / b ** 2.
                loss.backward()   
                step+=1
                if step % args.accumulate_step == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
                torch.cuda.empty_cache()         
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "img shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        if (e+1)%10==0:
            # model_eval(args,。 model, e+1)
            torch.save(model.state_dict(), '%s/ckpt_%d.pt'%(args.model_save_dir, e+1))


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    if args.train:
        main(args)
    else:
        model_eval(args)
