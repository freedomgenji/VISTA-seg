#coding=utf-8
import argparse
import csv
import os
import time
import logging
import random
import numpy as np

import torch
import torch.optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast

import models
from data.transforms import *
from data.datasets_nii import Brats_loadall_nii, Brats_loadall_test_nii
from data.datasets_wmh import WMH_loadall_nii, WMH_loadall_test_nii
from data.data_utils import init_fn
from utils import Parser,criterions
from utils.parser import setup 
from utils.lr_scheduler import LR_Scheduler, record_loss, MultiEpochsDataLoader 
from predict import AverageMeter, test_softmax

parser = argparse.ArgumentParser()

parser.add_argument('-batch_size', '--batch_size', default=2, type=int, help='Batch size')
parser.add_argument('--datapath', default='/root/shared-nvme/brats2023_npy', type=str)
parser.add_argument('--dataname', default='BRATS2023', type=str)
parser.add_argument('--savepath', default='Brats2023', type=str)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--resume_train', default=None, type=str,
                    help='Resume training from a checkpoint. Unlike --resume, this continues training.')
parser.add_argument('--skip_resume_eval', action='store_true', default=False,
                    help='Skip the validation pass normally run before --resume_train continues.')
parser.add_argument('--pretrain', default=None, type=str)
parser.add_argument('--lr', default=2e-4, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--num_epochs', default=700, type=int)
parser.add_argument('--iter_per_epoch', default=150, type=int)
parser.add_argument('--region_fusion_start_epoch', default=20, type=int)
parser.add_argument('--seed', default=1024, type=int)
parser.add_argument('--train_file', default=None, type=str)
parser.add_argument('--test_file', default=None, type=str)
parser.add_argument('--case_name', default=None, type=str,
                    help='Evaluate a single case name, e.g. BraTS-GLI-00043-000. Overrides --test_file for BraTS.')
parser.add_argument('--slice_index', default=None, type=int,
                    help='Optional bookkeeping for qualitative visualization; inference still saves the full 3D NPY.')
parser.add_argument('--pred_savepath', default=None, type=str,
                    help='Root directory for saving predicted uint8 masks as {mask_name}/{case_name}.npy.')
parser.add_argument('--save_cropped_pred', action='store_true', default=False,
                    help='Save predictions in the processed cropped NPY coordinate system.')
parser.add_argument('--metrics_csv', '--metric_csv', dest='metrics_csv', default=None, type=str,
                    help='Write per-mask average Dice/IoU/HD95 metrics to this CSV file.')
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--no_pin_memory', action='store_true', default=False)
parser.add_argument('--amp', action='store_true', default=False)
parser.add_argument('--grad_checkpoint', action='store_true', default=False)
parser.add_argument('--skip_eval', action='store_true', default=False)
parser.add_argument('--early_stop_patience', default=30, type=int,
                    help='Enable early stopping when > 0; number of evaluation checks without improvement.')
parser.add_argument('--early_stop_min_delta', default=0.0, type=float)
parser.add_argument('--early_stop_start_epoch', default=300, type=int)
parser.add_argument('--early_stop_eval_interval', default=100, type=int,
                    help='Run early-stop evaluation every N epochs after early_stop_start_epoch.')
parser.add_argument('--fusion_type', default='RFM', type=str)
parser.add_argument('--shared_rep_method', default='vista_seg', choices=['vista_seg'], type=str)
parser.add_argument('--meta_inner_steps', default=1, type=int)
parser.add_argument('--meta_inner_lr', default=1e-3, type=float)
parser.add_argument('--meta_support_ratio', default=0.5, type=float)
parser.add_argument('--meta_align_weight', default=0.01, type=float,
                    help='Fixed loss weight for full-view teacher alignment in VISTA-Seg mode.')
parser.add_argument('--meta_feature_weight', default=0.2, type=float,
                    help='Fixed loss weight for feature-level reconstruction in VISTA-Seg mode.')
parser.add_argument('--meta_first_order', action='store_true', default=True)
parser.add_argument('--meta_second_order', action='store_false', dest='meta_first_order')
parser.add_argument('--use_reg_loss', action='store_true', default=False)
parser.add_argument('--crop_size', default=80, type=int)
parser.add_argument('--normalize_on_load', action='store_true', default=False,
                    help='Apply foreground z-score when loading raw vol.npy files. Leave off for pre-normalized NPY data.')
parser.add_argument('--use_tumor_aware_crop', action='store_true', default=False,
                    help='Use ET/TC/WT-centered crop sampling for BraTS training patches.')
parser.add_argument('--tumor_crop_prob', default=0.9, type=float)
parser.add_argument('--tumor_crop_et_weight', default=4.0, type=float)
parser.add_argument('--tumor_crop_tc_weight', default=3.0, type=float)
parser.add_argument('--tumor_crop_wt_weight', default=2.0, type=float)
parser.add_argument('--use_region_tversky_loss', action='store_true', default=False,
                    help='Add WT/TC/ET region-level Tversky loss on fused prediction.')
parser.add_argument('--region_tversky_weight', default=0.5, type=float)
parser.add_argument('--region_tversky_wt_weight', default=0.5, type=float)
parser.add_argument('--region_tversky_tc_weight', default=1.5, type=float)
parser.add_argument('--region_tversky_et_weight', default=2.0, type=float)
parser.add_argument('--tversky_alpha', default=0.3, type=float)
parser.add_argument('--tversky_beta', default=0.7, type=float)
path = os.path.dirname(__file__)

## parse arguments
args = parser.parse_args()
setup(args, 'training')
logging.info('VISTA-Seg fixed loss weights: meta_feature_weight=%s, meta_align_weight=%s',
             args.meta_feature_weight, args.meta_align_weight)
if args.dataname == 'wmh':
    args.train_transforms = 'Compose([RandCrop3D((128,128,128)), RandomIntensityChange((0.1,0.1)), RandomFlip(0), NumpyType((np.float32, np.int64)),])'
else:
    if args.use_tumor_aware_crop:
        args.train_transforms = 'Compose([RandomRotion(10), RandomIntensityChange((0.1,0.1)), RandomFlip(0), NumpyType((np.float32, np.int64)),])'
    else:
        args.train_transforms = f'Compose([RandCrop3D(({args.crop_size},{args.crop_size},{args.crop_size})), RandomRotion(10), RandomIntensityChange((0.1,0.1)), RandomFlip(0), NumpyType((np.float32, np.int64)),])'
args.test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'

ckpts = args.savepath
os.makedirs(ckpts, exist_ok=True)

###tensorboard writer
writer = SummaryWriter(os.path.join(args.savepath, 'summary'))

###modality missing mask
if args.dataname == 'wmh':
    masks = [[False, True], [True, False], [True, True]]
    masks_torch = torch.from_numpy(np.array(masks))
    mask_name = ['t1', 'flair', 't1flair']
else:
    masks = [[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
            [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True], [True, False, False, True], [True, True, False, False],
            [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
            [True, True, True, True]]
    masks_torch = torch.from_numpy(np.array(masks))
    mask_name = ['t2', 't1c', 't1', 'flair', 
                't1cet2', 't1cet1', 'flairt1', 't1t2', 'flairt2', 'flairt1ce',
                'flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2',
                'flairt1cet1t2']
print (masks_torch.int())

def shared_alignment_loss(meta_outputs):
    if meta_outputs is None:
        return None
    if "meta_align_loss" in meta_outputs:
        return meta_outputs["meta_align_loss"]
    z_shared = meta_outputs["z_shared"]
    z_teacher = meta_outputs["z_teacher"].detach()
    z_shared = F.normalize(z_shared.flatten(1), dim=1, eps=1e-6)
    z_teacher = F.normalize(z_teacher.flatten(1), dim=1, eps=1e-6)
    return (1.0 - (z_shared * z_teacher).sum(dim=1)).mean()

def meta_feature_reconstruction_loss(meta_outputs):
    if meta_outputs is None or "meta_feature_loss" not in meta_outputs:
        return None
    return meta_outputs["meta_feature_loss"]

def masked_prediction_loss(pred, target, active, num_cls, dataname):
    if not active.any():
        return pred.new_tensor(0.0), pred.new_tensor(0.0)
    if dataname != 'wmh':
        cross = criterions.softmax_weighted_loss(pred[active], target[active], num_cls=num_cls)
        dice = criterions.dice_loss(pred[active], target[active], num_cls=num_cls)
    else:
        cross = pred.new_tensor(0.0)
        dice = criterions.dice_loss(pred[active], target[active], num_cls=num_cls)
    return cross, dice

def _region_sum(tensor, indices):
    return tensor[:, indices, ...].sum(dim=1).clamp(0.0, 1.0)

def region_tversky_loss(pred, target, num_cls, alpha=0.3, beta=0.7,
                        wt_weight=0.5, tc_weight=1.5, et_weight=2.0, eps=1e-6):
    if num_cls < 4:
        return pred.new_tensor(0.0)

    et_index = 3 if num_cls == 4 else num_cls - 1
    wt_indices = list(range(1, num_cls))
    tc_indices = [1, et_index]
    regions = [
        (_region_sum(pred, wt_indices), _region_sum(target, wt_indices), wt_weight),
        (_region_sum(pred, tc_indices), _region_sum(target, tc_indices), tc_weight),
        (_region_sum(pred, [et_index]), _region_sum(target, [et_index]), et_weight),
    ]

    total = pred.new_tensor(0.0)
    weight_sum = 0.0
    dims = tuple(range(1, pred.dim() - 1))
    for pred_region, target_region, weight in regions:
        if weight <= 0:
            continue
        true_pos = (pred_region * target_region).sum(dim=dims)
        false_pos = (pred_region * (1.0 - target_region)).sum(dim=dims)
        false_neg = ((1.0 - pred_region) * target_region).sum(dim=dims)
        score = (true_pos + eps) / (true_pos + alpha * false_pos + beta * false_neg + eps)
        total = total + float(weight) * (1.0 - score.mean())
        weight_sum += float(weight)

    if weight_sum <= 0.0:
        return pred.new_tensor(0.0)
    return total / weight_sum

def score_to_float(score):
    score = np.asarray(score, dtype=np.float32)
    return float(np.nanmean(score))

def move_optimizer_state_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)

def write_metrics_csv(metrics_csv, rows):
    if metrics_csv is None:
        return
    if not rows:
        logging.info('No metric rows to write.')
        return

    numeric_fields = [
        field for field in rows[0].keys()
        if field not in ('case_name', 'mask_name')
    ]
    output_rows = []
    mask_names = sorted(set(row['mask_name'] for row in rows))
    for current_mask in mask_names:
        mask_rows = [row for row in rows if row['mask_name'] == current_mask]
        summary = {
            'mask_name': current_mask,
            'num_cases': len(mask_rows),
        }
        for field in numeric_fields:
            values = np.asarray([row[field] for row in mask_rows], dtype=np.float64)
            summary[field] = float(np.nanmean(values)) if not np.all(np.isnan(values)) else float('nan')
        output_rows.append(summary)

    metrics_dir = os.path.dirname(metrics_csv)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)

    fieldnames = list(output_rows[0].keys())
    with open(metrics_csv, 'w', newline='', encoding='utf-8') as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(output_rows)
    logging.info('Saved metrics CSV: {}'.format(metrics_csv))

def evaluate_all_masks(test_loader, model, dataname, crop_size, log_prefix='',
                       pred_savepath=None, save_cropped_pred=False, metrics_csv=None):
    test_score = AverageMeter()
    metrics_rows = [] if metrics_csv is not None else None
    with torch.no_grad():
        logging.info('{}###########test set wi/wo postprocess###########'.format(log_prefix))
        for i, eval_mask in enumerate(masks):
            logging.info('{}{}'.format(log_prefix, mask_name[i]))
            dice_score = test_softmax(
                            test_loader,
                            model,
                            dataname=dataname,
                            feature_mask=eval_mask,
                            mask_name=mask_name[i],
                            crop_size=crop_size,
                            pred_savepath=pred_savepath,
                            save_cropped_pred=save_cropped_pred,
                            metrics_rows=metrics_rows)
            test_score.update(dice_score)
        logging.info('{}Avg scores: {}'.format(log_prefix, test_score.avg))
    write_metrics_csv(metrics_csv, metrics_rows)
    return test_score.avg, score_to_float(test_score.avg)

def main():
    ##########setting seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    ##########setting models
    if args.dataname in ['BRATS2020', 'BRATS2023', 'BRATS2018']:
        num_cls = 4
    elif args.dataname == 'BRATS2015':
        num_cls = 5
    elif args.dataname == 'wmh':
        num_cls = 1
    else:
        print ('dataset is error')
        exit(0)
    if args.dataname == 'wmh':
        model = models.VISTASegWMH(
            num_cls=num_cls, fusion_type=args.fusion_type, activation='sigmoid',
            shared_rep_method=args.shared_rep_method, meta_support_ratio=args.meta_support_ratio,
            meta_inner_steps=args.meta_inner_steps, meta_inner_lr=args.meta_inner_lr,
            meta_first_order=args.meta_first_order)
    else:
        model = models.VISTASeg(
            num_cls=num_cls, fusion_type=args.fusion_type,
            shared_rep_method=args.shared_rep_method, meta_support_ratio=args.meta_support_ratio,
            meta_inner_steps=args.meta_inner_steps, meta_inner_lr=args.meta_inner_lr,
            meta_first_order=args.meta_first_order)
    # print (model)
    model = torch.nn.DataParallel(model).cuda()
    model.module.use_checkpoint = args.grad_checkpoint

    ##########Setting learning schedule and optimizer
    lr_schedule = LR_Scheduler(args.lr, args.num_epochs)
    train_params = [{'params': model.parameters(), 'lr': args.lr, 'weight_decay':args.weight_decay}]
    optimizer = torch.optim.Adam(train_params,  betas=(0.9, 0.999), eps=1e-08, amsgrad=True)
    scaler = GradScaler('cuda', enabled=args.amp)
    if args.resume is not None and args.resume_train is not None:
        raise ValueError('Use only one of --resume or --resume_train.')
    start_epoch = 0
    resume_train_checkpoint = None

    ##########Setting data
    if args.dataname == 'wmh':
        default_train_file = os.path.join('split_person2', 'train.txt')
        default_test_file = os.path.join('split_person2', 'test.txt')
    elif args.dataname in ['BRATS2020', 'BRATS2023', 'BRATS2015']:
        default_train_file = 'train.txt'
        default_test_file = 'test.txt'
    elif args.dataname == 'BRATS2018':
        ####BRATS2018 contains three splits (1,2,3)
        default_train_file = 'train3.txt'
        default_test_file = 'test3.txt'
    else:
        raise ValueError(f'Unsupported dataname: {args.dataname}')

    train_file = args.train_file or default_train_file
    test_file = args.test_file or default_test_file
    eval_only = args.resume is not None
    if args.case_name is not None and args.dataname == 'wmh':
        raise ValueError('--case_name is currently supported for BraTS NPY datasets only.')

    logging.info(str(args))
    if args.dataname == 'wmh':
        train_set = None
        if not eval_only:
            train_set = WMH_loadall_nii(transforms=args.train_transforms, root=args.datapath, train_file=train_file)
        test_set = WMH_loadall_test_nii(transforms=args.test_transforms, root=args.datapath, test_file=test_file)
    else:
        train_set = None
        if not eval_only:
            train_set = Brats_loadall_nii(
                transforms=args.train_transforms,
                root=args.datapath,
                num_cls=num_cls,
                train_file=train_file,
                tumor_aware_crop=args.use_tumor_aware_crop,
                crop_size=(args.crop_size, args.crop_size, args.crop_size),
                tumor_crop_prob=args.tumor_crop_prob,
                tumor_crop_et_weight=args.tumor_crop_et_weight,
                tumor_crop_tc_weight=args.tumor_crop_tc_weight,
                tumor_crop_wt_weight=args.tumor_crop_wt_weight,
                normalize_on_load=args.normalize_on_load)
        test_kwargs = {
            'transforms': args.test_transforms,
            'root': args.datapath,
            'test_file': test_file,
            'normalize_on_load': args.normalize_on_load,
        }
        if args.case_name is not None:
            test_kwargs['case_name'] = args.case_name
        test_set = Brats_loadall_test_nii(**test_kwargs)
    train_loader = None
    if not eval_only:
        train_loader = MultiEpochsDataLoader(
            dataset=train_set,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=not args.no_pin_memory,
            shuffle=True,
            worker_init_fn=init_fn)
    test_loader = MultiEpochsDataLoader(
        dataset=test_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=not args.no_pin_memory)
    
    ##########Evaluate
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location='cpu')
        logging.info('best epoch: {}'.format(checkpoint['epoch']))
        missing, unexpected = model.load_state_dict(checkpoint['state_dict'], strict=False)
        if missing or unexpected:
            logging.info('checkpoint load with strict=False, missing={}, unexpected={}'.format(missing, unexpected))
        evaluate_all_masks(
            test_loader, model, args.dataname, args.crop_size,
            pred_savepath=args.pred_savepath,
            save_cropped_pred=args.save_cropped_pred,
            metrics_csv=args.metrics_csv)
        exit(0)
        return None

    if args.resume_train is not None:
        resume_train_checkpoint = torch.load(args.resume_train, map_location='cpu')
        checkpoint_epoch = int(resume_train_checkpoint.get('epoch', -1))
        logging.info('resume training from checkpoint: {}'.format(args.resume_train))
        logging.info('checkpoint epoch: {}, next training epoch: {}'.format(
            checkpoint_epoch + 1, checkpoint_epoch + 2))
        missing, unexpected = model.load_state_dict(resume_train_checkpoint['state_dict'], strict=False)
        if missing or unexpected:
            logging.info('checkpoint load with strict=False, missing={}, unexpected={}'.format(missing, unexpected))
        if 'optim_dict' not in resume_train_checkpoint:
            raise KeyError('Checkpoint does not contain optim_dict; cannot exactly resume training.')
        optimizer.load_state_dict(resume_train_checkpoint['optim_dict'])
        move_optimizer_state_to_device(optimizer, next(model.parameters()).device)
        if args.amp and 'scaler_dict' in resume_train_checkpoint:
            scaler.load_state_dict(resume_train_checkpoint['scaler_dict'])
        start_epoch = checkpoint_epoch + 1

    ##########Training
    start = time.time()
    torch.set_grad_enabled(True)
    logging.info('#############training############')
    logging.info('AMP enabled: {}'.format(args.amp))
    logging.info('Gradient checkpoint enabled: {}'.format(args.grad_checkpoint))
    logging.info('Visible CUDA devices: {}'.format(os.environ.get('CUDA_VISIBLE_DEVICES', 'all')))
    logging.info('torch.cuda.device_count(): {}'.format(torch.cuda.device_count()))
    early_stop_enabled = args.early_stop_patience > 0 and not args.skip_eval
    best_early_stop_score = -float('inf')
    best_early_stop_epoch = -1
    evals_without_improvement = 0
    last_eval_epoch_num = None
    early_stop_eval_interval = max(1, args.early_stop_eval_interval)
    if resume_train_checkpoint is not None:
        if 'early_stop_best_score' in resume_train_checkpoint:
            best_early_stop_score = float(resume_train_checkpoint['early_stop_best_score'])
            best_early_stop_epoch = int(resume_train_checkpoint.get('early_stop_best_epoch', -1))
        elif 'early_stop_score' in resume_train_checkpoint:
            best_early_stop_score = float(resume_train_checkpoint['early_stop_score'])
            best_early_stop_epoch = int(resume_train_checkpoint.get('epoch', -1))
        evals_without_improvement = int(
            resume_train_checkpoint.get('early_stop_evals_without_improvement', 0))
    logging.info('Early stopping enabled: {}, patience: {}, min_delta: {}, start_epoch: {}, eval_interval: {}'.format(
        early_stop_enabled, args.early_stop_patience, args.early_stop_min_delta,
        args.early_stop_start_epoch, early_stop_eval_interval))
    logging.info('Tumor-aware crop enabled: {}, prob: {}, ET/TC/WT weights: {}/{}/{}'.format(
        args.use_tumor_aware_crop, args.tumor_crop_prob,
        args.tumor_crop_et_weight, args.tumor_crop_tc_weight, args.tumor_crop_wt_weight))
    logging.info('Region Tversky enabled: {}, loss weight: {}, region weights WT/TC/ET: {}/{}/{}, alpha: {}, beta: {}'.format(
        args.use_region_tversky_loss, args.region_tversky_weight,
        args.region_tversky_wt_weight, args.region_tversky_tc_weight, args.region_tversky_et_weight,
        args.tversky_alpha, args.tversky_beta))
    # iter_per_epoch = args.iter_per_epoch
    iter_per_epoch = len(train_loader) if args.iter_per_epoch == -1 else args.iter_per_epoch

    def save_checkpoint(file_name, epoch, extra=None):
        payload = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'early_stop_best_score': best_early_stop_score,
            'early_stop_best_epoch': best_early_stop_epoch,
            'early_stop_evals_without_improvement': evals_without_improvement,
        }
        if args.amp:
            payload['scaler_dict'] = scaler.state_dict()
        if extra is not None:
            payload.update(extra)
        torch.save(payload, file_name)

    def run_early_stop_evaluation(epoch, log_prefix='[early-stop] '):
        nonlocal best_early_stop_score, best_early_stop_epoch, evals_without_improvement, last_eval_epoch_num
        epoch_num = epoch + 1
        eval_avg, eval_score = evaluate_all_masks(
            test_loader, model, args.dataname, args.crop_size,
            log_prefix=log_prefix)
        last_eval_epoch_num = epoch_num
        writer.add_scalar('early_stop_score', eval_score, global_step=epoch_num)
        if eval_score > best_early_stop_score + args.early_stop_min_delta:
            best_early_stop_score = eval_score
            best_early_stop_epoch = epoch
            evals_without_improvement = 0
            file_name = os.path.join(ckpts, 'model_best_early_stop.pth')
            save_checkpoint(file_name, epoch, {
                'early_stop_score': eval_score,
                'early_stop_avg': eval_avg,
            })
            logging.info('[early-stop] improved to {:.6f} at epoch {}, saved {}'.format(
                eval_score, epoch_num, file_name))
            return False

        evals_without_improvement += 1
        logging.info('[early-stop] no improvement for {}/{} evals; best {:.6f} at epoch {}'.format(
            evals_without_improvement,
            args.early_stop_patience,
            best_early_stop_score,
            best_early_stop_epoch + 1))
        if evals_without_improvement >= args.early_stop_patience:
            logging.info('[early-stop] stopping at epoch {}'.format(epoch_num))
            return True
        return False

    if args.resume_train is not None and not args.skip_resume_eval and not args.skip_eval:
        resume_eval_epoch = start_epoch - 1
        resume_eval_epoch_num = resume_eval_epoch + 1
        if resume_eval_epoch >= 0:
            should_count_as_early_stop_eval = (
                early_stop_enabled
                and resume_eval_epoch_num >= max(1, args.early_stop_start_epoch)
                and (resume_eval_epoch_num - max(1, args.early_stop_start_epoch)) % early_stop_eval_interval == 0
            )
            if should_count_as_early_stop_eval:
                if run_early_stop_evaluation(resume_eval_epoch, log_prefix='[early-stop] '):
                    return None
            else:
                eval_avg, eval_score = evaluate_all_masks(
                    test_loader, model, args.dataname, args.crop_size,
                    log_prefix='[resume] ')
                last_eval_epoch_num = resume_eval_epoch_num
                writer.add_scalar('resume_eval_score', eval_score, global_step=resume_eval_epoch_num)

    train_iter = iter(train_loader)
    for epoch in range(start_epoch, args.num_epochs):
        step_lr = lr_schedule(optimizer, epoch)
        writer.add_scalar('lr', step_lr, global_step=(epoch+1))
        b = time.time()
        for i in range(iter_per_epoch):
            step = (i+1) + epoch*iter_per_epoch
            ###Data load
            try:
                data = next(train_iter)
            except:
                train_iter = iter(train_loader)
                data = next(train_iter)
            x, target, mask = data[:3]
            x = x.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)

            model.module.is_training = True
            with autocast('cuda', enabled=args.amp):
                fuse_pred, sep_preds, prm_preds, meta_outputs = model(x, mask)

                ###Loss compute
                fuse_cross_loss = criterions.softmax_weighted_loss(fuse_pred, target, num_cls=num_cls)
                fuse_dice_loss = criterions.dice_loss(fuse_pred, target, num_cls=num_cls)
                fuse_loss = fuse_cross_loss + fuse_dice_loss if args.dataname != 'wmh' else fuse_dice_loss

                sep_cross_loss = torch.zeros(1, device=x.device).float()
                sep_dice_loss = torch.zeros(1, device=x.device).float()
                for modal_index, sep_pred in enumerate(sep_preds):
                    active = mask[:, modal_index].bool()
                    modal_cross, modal_dice = masked_prediction_loss(sep_pred, target, active, num_cls, args.dataname)
                    sep_cross_loss += modal_cross
                    sep_dice_loss += modal_dice
                sep_loss = sep_cross_loss + sep_dice_loss if args.dataname != 'wmh' else sep_dice_loss

                prm_cross_loss = torch.zeros(1, device=x.device).float()
                prm_dice_loss = torch.zeros(1, device=x.device).float()
                for prm_pred in prm_preds:
                    prm_cross_loss += criterions.softmax_weighted_loss(prm_pred, target, num_cls=num_cls)
                    prm_dice_loss += criterions.dice_loss(prm_pred, target, num_cls=num_cls)
                prm_loss = prm_cross_loss + prm_dice_loss if args.dataname != 'wmh' else prm_dice_loss
                region_tversky = fuse_pred.new_tensor(0.0)
                if args.use_region_tversky_loss and args.dataname != 'wmh':
                    region_tversky = region_tversky_loss(
                        fuse_pred,
                        target,
                        num_cls=num_cls,
                        alpha=args.tversky_alpha,
                        beta=args.tversky_beta,
                        wt_weight=args.region_tversky_wt_weight,
                        tc_weight=args.region_tversky_tc_weight,
                        et_weight=args.region_tversky_et_weight)

                use_reg_loss = 1 if args.use_reg_loss else 0
                if epoch < args.region_fusion_start_epoch:
                    loss = fuse_loss * 0.0 + sep_loss*use_reg_loss + prm_loss
                else:
                    loss = fuse_loss + sep_loss*use_reg_loss + prm_loss
                    if args.use_region_tversky_loss and args.dataname != 'wmh':
                        loss += args.region_tversky_weight * region_tversky

                meta_align_loss = shared_alignment_loss(meta_outputs)
                if meta_align_loss is None:
                    meta_align_loss = loss.detach().new_tensor(0.0)
                meta_feature_loss = meta_feature_reconstruction_loss(meta_outputs)
                if meta_feature_loss is None:
                    meta_feature_loss = loss.detach().new_tensor(0.0)

                loss += args.meta_feature_weight * meta_feature_loss
                loss += args.meta_align_weight * meta_align_loss

            optimizer.zero_grad(set_to_none=True)
            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            ###log
            writer.add_scalar('loss', loss.item(), global_step=step)
            writer.add_scalar('fuse_cross_loss', fuse_cross_loss.item(), global_step=step)
            writer.add_scalar('fuse_dice_loss', fuse_dice_loss.item(), global_step=step)
            writer.add_scalar('sep_cross_loss', sep_cross_loss.item(), global_step=step)
            writer.add_scalar('sep_dice_loss', sep_dice_loss.item(), global_step=step)
            writer.add_scalar('prm_cross_loss', prm_cross_loss.item(), global_step=step)
            writer.add_scalar('prm_dice_loss', prm_dice_loss.item(), global_step=step)
            if args.use_region_tversky_loss:
                writer.add_scalar('region_tversky_loss', region_tversky.item(), global_step=step)
                writer.add_scalar('region_tversky_weighted_loss',
                                  (args.region_tversky_weight * region_tversky).item(),
                                  global_step=step)
            writer.add_scalar('meta_align_loss', meta_align_loss.item(), global_step=step)
            writer.add_scalar('meta_feature_loss', meta_feature_loss.item(), global_step=step)
            writer.add_scalar('meta_align_weighted_loss',
                              (args.meta_align_weight * meta_align_loss).item(),
                              global_step=step)
            writer.add_scalar('meta_feature_weighted_loss',
                              (args.meta_feature_weight * meta_feature_loss).item(),
                              global_step=step)
            if meta_outputs is not None and "support_feature_loss" in meta_outputs:
                writer.add_scalar('meta_support_feature_loss', meta_outputs["support_feature_loss"].item(), global_step=step)
                writer.add_scalar('meta_query_feature_loss', meta_outputs["query_feature_loss"].item(), global_step=step)

            msg = 'Epoch {}/{}, Iter {}/{}, Loss {:.4f}, '.format((epoch+1), args.num_epochs, (i+1), iter_per_epoch, loss.item())
            msg += 'fusecross:{:.4f}, fusedice:{:.4f},'.format(fuse_cross_loss.item(), fuse_dice_loss.item())
            msg += 'sepcross:{:.4f}, sepdice:{:.4f},'.format(sep_cross_loss.item(), sep_dice_loss.item())
            msg += 'prmcross:{:.4f}, prmdice:{:.4f},'.format(prm_cross_loss.item(), prm_dice_loss.item())
            if args.use_region_tversky_loss:
                msg += 'region_tversky:{:.4f}, '.format(region_tversky.item())
            msg += 'meta_align_loss:{:.4f}, '.format(meta_align_loss.item())
            msg += 'meta_feature_loss:{:.4f}, '.format(meta_feature_loss.item())
            logging.info(msg)
        logging.info('train time per epoch: {}'.format(time.time() - b))

        ##########model save
        file_name = os.path.join(ckpts, 'model_last.pth')
        save_checkpoint(file_name, epoch)
        
        if (epoch+1) % 50 == 0 or (epoch>=(args.num_epochs-10)):
            file_name = os.path.join(ckpts, 'model_{}.pth'.format(epoch+1))
            save_checkpoint(file_name, epoch)

        epoch_num = epoch + 1
        early_stop_start_epoch = max(1, args.early_stop_start_epoch)
        should_run_early_stop_eval = (
            early_stop_enabled
            and epoch_num >= early_stop_start_epoch
            and (epoch_num - early_stop_start_epoch) % early_stop_eval_interval == 0
        )
        if should_run_early_stop_eval:
            if run_early_stop_evaluation(epoch, log_prefix='[early-stop] '):
                break

    msg = 'total time: {:.4f} hours'.format((time.time() - start)/3600)
    logging.info(msg)

    ##########Evaluate the last epoch model
    if args.skip_eval:
        logging.info('Skipping final evaluation because --skip_eval was set.')
        return None
    if last_eval_epoch_num == args.num_epochs:
        logging.info('Skipping final evaluation because epoch {} was already evaluated.'.format(
            args.num_epochs))
        return None

    evaluate_all_masks(test_loader, model, args.dataname, args.crop_size)

if __name__ == '__main__':
    main()
