import os
import time
import logging
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

cudnn.benchmark = True

path = os.path.dirname(__file__)
from utils.generate import generate_snapshot

BRATS_4CLASS_DATASETS = ('BRATS2020', 'BRATS2023', 'BRATS2018')


def softmax_output_dice_class4(output, target):
    eps = 1e-8
    target = target.clone()
    target[target == 4] = 3
    #######label1########
    o1 = (output == 1).float()
    t1 = (target == 1).float()
    intersect1 = torch.sum(2 * (o1 * t1), dim=(1,2,3)) + eps
    denominator1 = torch.sum(o1, dim=(1,2,3)) + torch.sum(t1, dim=(1,2,3)) + eps
    ncr_net_dice = intersect1 / denominator1

    o2 = (output == 2).float()
    t2 = (target == 2).float()
    intersect2 = torch.sum(2 * (o2 * t2), dim=(1,2,3)) + eps
    denominator2 = torch.sum(o2, dim=(1,2,3)) + torch.sum(t2, dim=(1,2,3)) + eps
    edema_dice = intersect2 / denominator2

    o3 = (output == 3).float()
    t3 = (target == 3).float()
    intersect3 = torch.sum(2 * (o3 * t3), dim=(1,2,3)) + eps
    denominator3 = torch.sum(o3, dim=(1,2,3)) + torch.sum(t3, dim=(1,2,3)) + eps
    enhancing_dice = intersect3 / denominator3

    ####post processing:
    if torch.sum(o3) < 500:
       o4 = o3 * 0.0
    else:
       o4 = o3
    t4 = t3
    intersect4 = torch.sum(2 * (o4 * t4), dim=(1,2,3)) + eps
    denominator4 = torch.sum(o4, dim=(1,2,3)) + torch.sum(t4, dim=(1,2,3)) + eps
    enhancing_dice_postpro = intersect4 / denominator4

    o_whole = o1 + o2 + o3 
    t_whole = t1 + t2 + t3 
    intersect_whole = torch.sum(2 * (o_whole * t_whole), dim=(1,2,3)) + eps
    denominator_whole = torch.sum(o_whole, dim=(1,2,3)) + torch.sum(t_whole, dim=(1,2,3)) + eps
    dice_whole = intersect_whole / denominator_whole

    o_core = o1 + o3
    t_core = t1 + t3
    intersect_core = torch.sum(2 * (o_core * t_core), dim=(1,2,3)) + eps
    denominator_core = torch.sum(o_core, dim=(1,2,3)) + torch.sum(t_core, dim=(1,2,3)) + eps
    dice_core = intersect_core / denominator_core

    dice_separate = torch.cat((torch.unsqueeze(ncr_net_dice, 1), torch.unsqueeze(edema_dice, 1), torch.unsqueeze(enhancing_dice, 1)), dim=1)
    dice_evaluate = torch.cat((torch.unsqueeze(dice_whole, 1), torch.unsqueeze(dice_core, 1), torch.unsqueeze(enhancing_dice, 1), torch.unsqueeze(enhancing_dice_postpro, 1)), dim=1)

    return dice_separate.cpu().numpy(), dice_evaluate.cpu().numpy()

def softmax_output_dice_class5(output, target):
    eps = 1e-8
    #######label1########
    o1 = (output == 1).float()
    t1 = (target == 1).float()
    intersect1 = torch.sum(2 * (o1 * t1), dim=(1,2,3)) + eps
    denominator1 = torch.sum(o1, dim=(1,2,3)) + torch.sum(t1, dim=(1,2,3)) + eps
    necrosis_dice = intersect1 / denominator1

    o2 = (output == 2).float()
    t2 = (target == 2).float()
    intersect2 = torch.sum(2 * (o2 * t2), dim=(1,2,3)) + eps
    denominator2 = torch.sum(o2, dim=(1,2,3)) + torch.sum(t2, dim=(1,2,3)) + eps
    edema_dice = intersect2 / denominator2

    o3 = (output == 3).float()
    t3 = (target == 3).float()
    intersect3 = torch.sum(2 * (o3 * t3), dim=(1,2,3)) + eps
    denominator3 = torch.sum(o3, dim=(1,2,3)) + torch.sum(t3, dim=(1,2,3)) + eps
    non_enhancing_dice = intersect3 / denominator3

    o4 = (output == 4).float()
    t4 = (target == 4).float()
    intersect4 = torch.sum(2 * (o4 * t4), dim=(1,2,3)) + eps
    denominator4 = torch.sum(o4, dim=(1,2,3)) + torch.sum(t4, dim=(1,2,3)) + eps
    enhancing_dice = intersect4 / denominator4

    ####post processing:
    if torch.sum(o4) < 500:
        o5 = o4 * 0
    else:
        o5 = o4
    t5 = t4
    intersect5 = torch.sum(2 * (o5 * t5), dim=(1,2,3)) + eps
    denominator5 = torch.sum(o5, dim=(1,2,3)) + torch.sum(t5, dim=(1,2,3)) + eps
    enhancing_dice_postpro = intersect5 / denominator5

    o_whole = o1 + o2 + o3 + o4
    t_whole = t1 + t2 + t3 + t4
    intersect_whole = torch.sum(2 * (o_whole * t_whole), dim=(1,2,3)) + eps
    denominator_whole = torch.sum(o_whole, dim=(1,2,3)) + torch.sum(t_whole, dim=(1,2,3)) + eps
    dice_whole = intersect_whole / denominator_whole

    o_core = o1 + o3 + o4
    t_core = t1 + t3 + t4
    intersect_core = torch.sum(2 * (o_core * t_core), dim=(1,2,3)) + eps
    denominator_core = torch.sum(o_core, dim=(1,2,3)) + torch.sum(t_core, dim=(1,2,3)) + eps
    dice_core = intersect_core / denominator_core

    dice_separate = torch.cat((torch.unsqueeze(necrosis_dice, 1), torch.unsqueeze(edema_dice, 1), torch.unsqueeze(non_enhancing_dice, 1), torch.unsqueeze(enhancing_dice, 1)), dim=1)
    dice_evaluate = torch.cat((torch.unsqueeze(dice_whole, 1), torch.unsqueeze(dice_core, 1), torch.unsqueeze(enhancing_dice, 1), torch.unsqueeze(enhancing_dice_postpro, 1)), dim=1)

    return dice_separate.cpu().numpy(), dice_evaluate.cpu().numpy()

def softmax_output_dice_class_wmh(output, target):
    eps = 1e-8

    o2 = (output == 1).float()
    t2 = (target == 1).float()
    intersect2 = torch.sum(2 * (o2 * t2), dim=(1,2,3)) + eps
    denominator2 = torch.sum(o2, dim=(1,2,3)) + torch.sum(t2, dim=(1,2,3)) + eps
    wmh_dice = intersect2 / denominator2

    dice_separate = [wmh_dice.cpu().numpy()]
    dice_evaluate = [wmh_dice.cpu().numpy()]
    return dice_separate, dice_evaluate

def binary_iou(pred_mask, target_mask):
    pred_mask = np.asarray(pred_mask, dtype=bool)
    target_mask = np.asarray(target_mask, dtype=bool)
    union = np.logical_or(pred_mask, target_mask).sum()
    if union == 0:
        return 1.0
    intersection = np.logical_and(pred_mask, target_mask).sum()
    return float(intersection / union)

def binary_hd95(pred_mask, target_mask):
    pred_mask = np.asarray(pred_mask, dtype=bool)
    target_mask = np.asarray(target_mask, dtype=bool)
    if not pred_mask.any() and not target_mask.any():
        return 0.0
    if not pred_mask.any() or not target_mask.any():
        return float('nan')

    structure = np.ones((3, 3, 3), dtype=bool)
    pred_surface = np.logical_xor(
        pred_mask,
        binary_erosion(pred_mask, structure=structure, border_value=0))
    target_surface = np.logical_xor(
        target_mask,
        binary_erosion(target_mask, structure=structure, border_value=0))

    target_distance = distance_transform_edt(~target_surface)
    pred_distance = distance_transform_edt(~pred_surface)
    distances = np.concatenate((
        target_distance[pred_surface],
        pred_distance[target_surface]))
    if distances.size == 0:
        return 0.0
    return float(np.percentile(distances, 95))

def brats_region_mask(label_np, num_cls, region_name, postprocess_et=False):
    label_np = np.asarray(label_np)
    if num_cls == 4:
        label_np = label_np.copy()
        label_np[label_np == 4] = 3
        et_label = 3
        if region_name == 'WT':
            region = label_np > 0
        elif region_name == 'TC':
            region = np.logical_or(label_np == 1, label_np == et_label)
        elif region_name in ('ET', 'ET_postpro'):
            region = label_np == et_label
        else:
            raise ValueError('Unsupported BraTS region: {}'.format(region_name))
    else:
        et_label = num_cls - 1
        if region_name == 'WT':
            region = label_np > 0
        elif region_name == 'TC':
            region = np.logical_or.reduce((
                label_np == 1, label_np == 3, label_np == et_label))
        elif region_name in ('ET', 'ET_postpro'):
            region = label_np == et_label
        else:
            raise ValueError('Unsupported BraTS region: {}'.format(region_name))

    if postprocess_et and region_name == 'ET_postpro' and region.sum() < 500:
        region = np.zeros_like(region, dtype=bool)
    return region

def compute_iou_hd95_metrics(output, target, dataname, class_evaluation):
    output_np = output.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    rows = []

    if dataname in BRATS_4CLASS_DATASETS:
        num_cls = 4
    elif dataname == 'BRATS2015':
        num_cls = 5
    elif dataname == 'wmh':
        num_cls = 1
    else:
        raise ValueError(f'Unsupported dataname for metric export: {dataname}')

    for sample_idx in range(output_np.shape[0]):
        pred_sample = np.squeeze(output_np[sample_idx])
        target_sample = np.squeeze(target_np[sample_idx])
        metric_values = {}
        if dataname == 'wmh':
            pred_region = pred_sample == 1
            target_region = target_sample == 1
            metric_values['wmh'] = {
                'iou': binary_iou(pred_region, target_region),
                'hd95': binary_hd95(pred_region, target_region),
            }
        else:
            for region_name in class_evaluation:
                pred_region = brats_region_mask(
                    pred_sample, num_cls, region_name,
                    postprocess_et=(region_name == 'ET_postpro'))
                target_region = brats_region_mask(
                    target_sample, num_cls, region_name,
                    postprocess_et=False)
                metric_values[region_name] = {
                    'iou': binary_iou(pred_region, target_region),
                    'hd95': binary_hd95(pred_region, target_region),
                }
        rows.append(metric_values)
    return rows

def create_weighted_tensor(crop_size, min_val, max_val):
    k = (crop_size - 1) / 2
    coords = torch.linspace(-k, k, steps=crop_size)
    z, y, x = torch.meshgrid(coords, coords, coords, indexing='ij')
    
    distance = torch.sqrt(x**2 + y**2 + z**2)
    max_distance = torch.sqrt(torch.tensor(3)) * k
    normalized_distance = torch.clamp(distance / max_distance, 0, 1)
    
    interpolated = max_val - (max_val - min_val) * normalized_distance
    
    return interpolated.unsqueeze(0)

def save_pred_masks(pred, names, pred_savepath, mask_name, expected_shape=None):
    if pred_savepath is None:
        return
    if mask_name is None:
        raise ValueError('mask_name is required when pred_savepath is set.')

    save_dir = os.path.join(pred_savepath, mask_name)
    os.makedirs(save_dir, exist_ok=True)

    pred_cpu = pred.detach().cpu()
    for sample_idx, name in enumerate(names):
        pred_np = pred_cpu[sample_idx].numpy().squeeze().astype(np.uint8)
        pred_np[pred_np == 4] = 3
        if expected_shape is not None and tuple(pred_np.shape) != tuple(expected_shape):
            raise ValueError(
                'Prediction for {} has shape {}, expected {} before saving.'.format(
                    name, pred_np.shape, expected_shape))

        save_path = os.path.join(save_dir, '{}.npy'.format(name))
        np.save(save_path, pred_np)
        logging.info('Saved prediction mask: {} shape={} dtype={}'.format(
            save_path, pred_np.shape, pred_np.dtype))

def test_softmax(
        test_loader,
        model,
        dataname = 'BRATS2020',
        feature_mask=None,
        mask_name=None,
        crop_size=80,
        pred_savepath=None,
        save_cropped_pred=False,
        metrics_rows=None):

    H, W, T = 240, 240, 155
    model.eval()
    vals_evaluation = AverageMeter()
    vals_separate = AverageMeter()
    crop_size = 128 if dataname == 'wmh' else crop_size
    weight_tensor = torch.ones(1, crop_size, crop_size, crop_size).float().cuda()
    # weight_tensor = create_weighted_tensor(crop_size, 0.5, 1.0).float().cuda()

    if dataname == 'wmh':
        num_cls = 1
        class_evaluation = 'wmh',
    elif dataname in BRATS_4CLASS_DATASETS:
        num_cls = 4
        class_evaluation= 'WT', 'TC', 'ET', 'ET_postpro'
        class_separate = 'ncr_net', 'edema', 'enhancing'
    elif dataname == 'BRATS2015':
        num_cls = 5
        class_evaluation= 'WT', 'TC', 'ET', 'ET_postpro'
        class_separate = 'necrosis', 'edema', 'non_enhancing', 'enhancing'
    else:
        raise ValueError(f'Unsupported dataname for evaluation: {dataname}')


    for i, data in enumerate(test_loader):
        target = data[1].cuda()
        x = data[0].cuda()
        names = data[-1]
        if feature_mask is not None:
            mask = torch.from_numpy(np.array(feature_mask))
            mask = torch.unsqueeze(mask, dim=0).repeat(len(names), 1)
        else:
            mask = data[2]
        mask = mask.cuda()
        _, _, H, W, Z = x.size()
        #########get h_ind, w_ind, z_ind for sliding windows
        h_cnt = np.int64(np.ceil((H - crop_size) / (crop_size * (1 - 0.5))))
        h_idx_list = range(0, h_cnt)
        h_idx_list = [h_idx * np.int64(crop_size * (1 - 0.5)) for h_idx in h_idx_list]
        h_idx_list.append(H - crop_size)

        w_cnt = np.int64(np.ceil((W - crop_size) / (crop_size * (1 - 0.5))))
        w_idx_list = range(0, w_cnt)
        w_idx_list = [w_idx * np.int64(crop_size * (1 - 0.5)) for w_idx in w_idx_list]
        w_idx_list.append(W - crop_size)

        z_cnt = np.int64(np.ceil((Z - crop_size) / (crop_size * (1 - 0.5))))
        z_idx_list = range(0, z_cnt)
        z_idx_list = [z_idx * np.int64(crop_size * (1 - 0.5)) for z_idx in z_idx_list]
        z_idx_list.append(Z - crop_size)

        #####compute calculation times for each pixel in sliding windows
        weight1 = torch.zeros(1, 1, H, W, Z).float().cuda()
        for h in h_idx_list:
            for w in w_idx_list:
                for z in z_idx_list:
                    weight1[:, :, h:h+crop_size, w:w+crop_size, z:z+crop_size] += weight_tensor
        weight = weight1.repeat(len(names), num_cls, 1, 1, 1)

        #####evaluation
        pred = torch.zeros(len(names), num_cls, H, W, Z).float().cuda()
        model.module.is_training=False
        for h in h_idx_list:
            for w in w_idx_list:
                for z in z_idx_list:
                    x_input = x[:, :, h:h+crop_size, w:w+crop_size, z:z+crop_size]
                    pred_part = model(x_input, mask)
                    # pred[:, :, h:h+crop_size, w:w+crop_size, z:z+crop_size] += pred_part
                    pred[:, :, h:h+crop_size, w:w+crop_size, z:z+crop_size] += pred_part * weight_tensor
        pred = pred / weight
        b = time.time()
        # pred = pred[:, :, :H, :W, :T]
        if num_cls > 1:
            pred = torch.argmax(pred, dim=1)
        else:
            pred = (pred > 0.5).float().squeeze(1)

        # save_visual (保存为npy，然后在ipynb进一步保存为图片)
        # if mask_name is not None:
        #     save_name = f'Subject_{i+1}_mask_{mask_name}.npy'
        #     # save_path = os.path.join('/mnt/data2/xxx/output/hetero/visual', save_name)
        #     save_path = os.path.join('visual', save_name)
        #     save_data = pred.cpu().numpy()
        #     np.save(save_path, save_data)

        expected_shape = None if save_cropped_pred else (
            (240, 240, 155) if dataname in BRATS_4CLASS_DATASETS else None)
        save_pred_masks(pred, names, pred_savepath, mask_name, expected_shape=expected_shape)

        if dataname in BRATS_4CLASS_DATASETS:
            scores_separate, scores_evaluation = softmax_output_dice_class4(pred, target)
        elif dataname == 'BRATS2015':
            scores_separate, scores_evaluation = softmax_output_dice_class5(pred, target)
        elif dataname == 'wmh':
            scores_separate, scores_evaluation = softmax_output_dice_class_wmh(pred, target)
        iou_hd95_values = compute_iou_hd95_metrics(
            pred, target, dataname, class_evaluation)
        for k, name in enumerate(names):
            msg = 'Subject {}/{}, {}/{}'.format((i+1), len(test_loader), (k+1), len(names))
            msg += '{:>20}, '.format(name)

            vals_separate.update(scores_separate[k])
            vals_evaluation.update(scores_evaluation[k])
            msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, scores_evaluation[k])])
            #msg += ',' + ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_separate, scores_separate[k])])

            logging.info(msg)

            if metrics_rows is not None:
                row = {
                    'case_name': str(name),
                    'mask_name': mask_name if mask_name is not None else '',
                }
                dice_values = np.asarray(scores_evaluation[k], dtype=np.float64).reshape(-1)
                for metric_idx, region_name in enumerate(class_evaluation):
                    region_metrics = iou_hd95_values[k][region_name]
                    row['dice_{}'.format(region_name)] = float(dice_values[metric_idx])
                    row['iou_{}'.format(region_name)] = region_metrics['iou']
                    row['hd95_{}'.format(region_name)] = region_metrics['hd95']
                metrics_rows.append(row)
    if mask_name is not None:
        msg = 'Mask {} Average scores: '.format(mask_name)
    else:
        msg = 'Average scores: '
    msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, vals_evaluation.avg)])
    #msg += ',' + ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_separate, vals_evaluation.avg)])
    # print (msg)
    logging.info(msg)
    model.train()
    return vals_evaluation.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
