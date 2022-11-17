import argparse
import logging
import math
import os
import random
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# import test  # import test.py to get mAP after each epoch
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr
from utils.loss import ComputeLoss, ComputeLossOTA
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, is_parallel

logger = logging.getLogger(__name__)


def train(hyp, opt, device, tb_writer=None):
    logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    save_dir, epochs, batch_size, total_batch_size, weights, freeze = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.freeze

    # Directories
    wdir = save_dir / 'weights'  # weight를 저장할 폴더
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    results_file = save_dir / 'results.txt'

    # Save run settings

    # 하이퍼파라미터 저장
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)  # dump는 hyp라는 파이썬 오브젝트를 f파일에 저장하는 함수

    # opt(옵션) 저장
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    # Configure
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict

    nc = 60  # number of classes
    names = data_dict['names']  # class names

    # 모델 정의
    model = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
    check_dataset(data_dict)  # check
    train_path = data_dict['train'] #train_data 경로
    print(train_path)
    test_path = data_dict['val'] #test_data 경로

    # 반복문을 통해 freeze된 파라미터는 requires_grad를 false로 바꿔 파라미터 업데이트를 안되게 막음
    # 나머지 파라미터들은 반복문을 통해 requires_grad를 true로 바꿔 파라미터 업데이트를 하게 해줌
    freeze = [f'model.{x}.' for x in
              (freeze if len(freeze) > 1 else range(freeze[0]))]  # parameter names to freeze (full or partial)
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print('freezing %s' % k)
            v.requires_grad = False

    # Optimizer

    # weight_decay를 이런식으로 batch 사이즈에 맞게 정규화? 해줌
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay
    logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay
        if hasattr(v, 'im'):
            if hasattr(v.im, 'implicit'):
                pg0.append(v.im.implicit)
            else:
                for iv in v.im:
                    pg0.append(iv.implicit)
        if hasattr(v, 'imc'):
            if hasattr(v.imc, 'implicit'):
                pg0.append(v.imc.implicit)
            else:
                for iv in v.imc:
                    pg0.append(iv.implicit)
        if hasattr(v, 'imb'):
            if hasattr(v.imb, 'implicit'):
                pg0.append(v.imb.implicit)
            else:
                for iv in v.imb:
                    pg0.append(iv.implicit)
        if hasattr(v, 'imo'):
            if hasattr(v.imo, 'implicit'):
                pg0.append(v.imo.implicit)
            else:
                for iv in v.imo:
                    pg0.append(iv.implicit)
        if hasattr(v, 'ia'):
            if hasattr(v.ia, 'implicit'):
                pg0.append(v.ia.implicit)
            else:
                for iv in v.ia:
                    pg0.append(iv.implicit)
        if hasattr(v, 'attn'):
            if hasattr(v.attn, 'logit_scale'):
                pg0.append(v.attn.logit_scale)
            if hasattr(v.attn, 'q_bias'):
                pg0.append(v.attn.q_bias)
            if hasattr(v.attn, 'v_bias'):
                pg0.append(v.attn.v_bias)
            if hasattr(v.attn, 'relative_position_bias_table'):
                pg0.append(v.attn.relative_position_bias_table)
        if hasattr(v, 'rbr_dense'):
            if hasattr(v.rbr_dense, 'weight_rbr_origin'):
                pg0.append(v.rbr_dense.weight_rbr_origin)
            if hasattr(v.rbr_dense, 'weight_rbr_avg_conv'):
                pg0.append(v.rbr_dense.weight_rbr_avg_conv)
            if hasattr(v.rbr_dense, 'weight_rbr_pfir_conv'):
                pg0.append(v.rbr_dense.weight_rbr_pfir_conv)
            if hasattr(v.rbr_dense, 'weight_rbr_1x1_kxk_idconv1'):
                pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_idconv1)
            if hasattr(v.rbr_dense, 'weight_rbr_1x1_kxk_conv2'):
                pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_conv2)
            if hasattr(v.rbr_dense, 'weight_rbr_gconv_dw'):
                pg0.append(v.rbr_dense.weight_rbr_gconv_dw)
            if hasattr(v.rbr_dense, 'weight_rbr_gconv_pw'):
                pg0.append(v.rbr_dense.weight_rbr_gconv_pw)
            if hasattr(v.rbr_dense, 'vector'):
                pg0.append(v.rbr_dense.vector)

    optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum

    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    if opt.linear_lr:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    else:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # EMA
    ema = ModelEMA(model)

    # # Resume
    start_epoch, best_fitness = 0, 0.0

    # Image sizes
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples

    # Trainloader
    dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt,
                                            hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect,
                                            workers=opt.workers,
                                            image_weights=opt.image_weights, quad=opt.quad, prefix=colorstr('train: '))
    mlc = 59 # max label class
    nb = len(dataloader)  # number of batches
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)

    # Process 0
    # testloader = create_dataloader(test_path, imgsz_test, batch_size * 2, gs, opt,  # testloader
    #                                hyp=hyp, cache=opt.cache_images and not opt.notest, rect=True, rank=-1,
    #                                world_size=opt.world_size, workers=opt.workers,
    #                                pad=0.5, prefix=colorstr('val: '))[0]

    if not opt.resume:
        labels = np.concatenate(dataset.labels, 0)
        c = torch.tensor(labels[:, 0])  # classes
        # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
        # model._initialize_biases(cf.to(device))
        # Anchors
        if not opt.noautoanchor:
            check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
        model.half().float()  # pre-reduce anchor precision

    # Model parameters
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names

    # Start training
    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=True)
    compute_loss_ota = ComputeLossOTA(model)  # init loss class
    compute_loss = ComputeLoss(model)  # init loss class
    logger.info(f'Image sizes {imgsz} train, {imgsz_test} test\n'
                f'Using {dataloader.num_workers} dataloader workers\n'
                f'Logging results to {save_dir}\n'
                f'Starting training for {epochs} epochs...')
    torch.save(model, wdir / 'init.pt')
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()

        # Update image weights (optional)
        if opt.image_weights:
            # Generate indices
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx
        # Update mosaic border
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(4, device=device)  # mean losses
        pbar = enumerate(dataloader)
        logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'labels', 'img_size'))
        pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward
            with amp.autocast(enabled=True):  # 모델의 속도를 높여줌
                pred = model(imgs)  # forward
                if 'loss_ota' not in hyp or hyp['loss_ota'] == 1:
                    loss, loss_items = compute_loss_ota(pred, targets.to(device), imgs)  # loss scaled by batch_size
                else:
                    loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                if opt.quad:
                    loss *= 4.

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni % accumulate == 0:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            # Print
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' * 2 + '%10.4g' * 6) % (
                '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
            pbar.set_description(s)
            # end batch ------------------------------------------------------------------------------------------------
        # end epoch ----------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for tensorboard
        scheduler.step()

        # DDP process 0 or single-GPU

        # mAP
        ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
        final_epoch = epoch + 1 == epochs
        # if not opt.notest or final_epoch:  # Calculate mAP
        #     results, maps, times = test.test(data_dict,
        #                                      batch_size=batch_size * 2,
        #                                      imgsz=imgsz_test,
        #                                      model=ema.ema,
        #                                      single_cls=opt.single_cls,
        #                                      dataloader=testloader,
        #                                      save_dir=save_dir,
        #                                      verbose=nc < 50 and final_epoch,
        #                                      compute_loss=compute_loss,
        #                                      v5_metric=opt.v5_metric)

        # Write
        with open(results_file, 'a') as f:
            f.write(s + '%10.4g' * 7 % results + '\n')  # append metrics, val_loss
        if len(opt.name) and opt.bucket:
            os.system('gsutil cp %s gs://%s/results/results%s.txt' % (results_file, opt.bucket, opt.name))

        # Log
        tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
                'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                'x/lr0', 'x/lr1', 'x/lr2']  # params
        for x, tag in zip(list(mloss[:-1]) + list(results) + lr, tags):
            if tb_writer:
                tb_writer.add_scalar(tag, x, epoch)  # tensorboard

        # Update best mAP
        fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
        if fi > best_fitness:
            best_fitness = fi

        # Save model
        if (not opt.nosave) or (final_epoch and not opt.evolve):  # if save
            ckpt = {'epoch': epoch,
                    'best_fitness': best_fitness,
                    'training_results': results_file.read_text(),
                    'model': deepcopy(model.module if is_parallel(model) else model).half(),
                    'ema': deepcopy(ema.ema).half(),
                    'updates': ema.updates,
                    'optimizer': optimizer.state_dict()}

            # Save last, best and delete
            torch.save(ckpt, last)
            if best_fitness == fi:
                torch.save(ckpt, best)
            if (best_fitness == fi) and (epoch >= 200):
                torch.save(ckpt, wdir / 'best_{:03d}.pt'.format(epoch))
            if epoch == 0:
                torch.save(ckpt, wdir / 'epoch_{:03d}.pt'.format(epoch))
            elif ((epoch + 1) % 25) == 0:
                torch.save(ckpt, wdir / 'epoch_{:03d}.pt'.format(epoch))
            elif epoch >= (epochs - 5):
                torch.save(ckpt, wdir / 'epoch_{:03d}.pt'.format(epoch))
            del ckpt

        # end epoch ----------------------------------------------------------------------------------------------------

        # end training
        logger.info('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))

        # Strip optimizers
        final = best if best.exists() else last  # final model
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
        if opt.bucket:
            os.system(f'gsutil cp {final} gs://{opt.bucket}/weights')  # upload
    torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    '''
    python3 argparse_test.py --target=테스트 --env=local 
    이런식으로 매개변수를 줄 수 있는데 여기서의 매개변수를 관리하게 해주는 객체가 parser임
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolo7.pt', help='initial weights path')  # weight 경로
    parser.add_argument('--cfg', type=str, default='cfg/training/yolov7-e6e.yaml', help='model.yaml path')  #
    # data.yaml의 경로
    # data.yaml 안에는 클래스의 개수, train 데이터셋의 개수 등이 적혀 있음
    parser.add_argument('--data', type=str, default='data/data.yaml', help='data.yaml path')
    # 하이퍼 파라미터 경로
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.p5.yaml', help='hyperparameters path')
    # epoch 값
    parser.add_argument('--epochs', type=int, default=300)
    # batch size 값
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    # image size 값
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')

    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')

    # 학습 진행시키며 hyperparameter evolve 할지 적어줌
    # store_true: 인자 값을 적으면 true, 적지 않으면 false라는 뜻
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='cuda:1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0],
                        help='Freeze layers: backbone of yolov7=50, first3=0 1 2')
    parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')

    # parser 객체에 저장된 값들을 딕셔너리 형태로 변환
    opt = parser.parse_args()

    # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
    opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
    assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
    opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
    opt.name = 'evolve' if opt.evolve else opt.name
    opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)  # increment run

    opt.total_batch_size = opt.batch_size
    # 하이퍼파라미터 로딩
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps
    device = opt.device

    # Train
    logger.info(opt)  # opt에 적힌 정보들 log
    if not opt.evolve:  # 만약 하이퍼파라미터를 evolve할 게 아니라면
        prefix = colorstr('tensorboard: ')
        logger.info(f"{prefix}Start with 'tensorboard --logdir {opt.project}', view at http://localhost:6006/")
        tb_writer = SummaryWriter(opt.save_dir)  # Tensorboard, 학습 관련 정보들을 저장함.
        train(hyp, opt, device, tb_writer)  # train 진행

    # Evolve hyperparameters (optional)
    # else:
    #     # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
    #     meta = {'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
    #             'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
    #             'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
    #             'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
    #             'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
    #             'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
    #             'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
    #             'box': (1, 0.02, 0.2),  # box loss gain
    #             'cls': (1, 0.2, 4.0),  # cls loss gain
    #             'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
    #             'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
    #             'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
    #             'iou_t': (0, 0.1, 0.7),  # IoU training threshold
    #             'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
    #             'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
    #             'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
    #             'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
    #             'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
    #             'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
    #             'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
    #             'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
    #             'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
    #             'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
    #             'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
    #             'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
    #             'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
    #             'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
    #             'mixup': (1, 0.0, 1.0),  # image mixup (probability)
    #             'copy_paste': (1, 0.0, 1.0),  # segment copy-paste (probability)
    #             'paste_in': (1, 0.0, 1.0)}  # segment copy-paste (probability)
    #
    #     with open(opt.hyp, errors='ignore') as f:
    #         hyp = yaml.safe_load(f)  # load hyps dict
    #         if 'anchors' not in hyp:  # anchors commented in hyp.yaml
    #             hyp['anchors'] = 3
    #
    #     assert opt.local_rank == -1, 'DDP mode not implemented for --evolve'
    #     opt.notest, opt.nosave = True, True  # only test/save final epoch
    #     # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
    #     yaml_file = Path(opt.save_dir) / 'hyp_evolved.yaml'  # save best result here
    #     if opt.bucket:
    #         os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  # download evolve.txt if exists
    #
    #     for _ in range(300):  # generations to evolve
    #         if Path('evolve.txt').exists():  # if evolve.txt exists: select best hyps and mutate
    #             # Select parent(s)
    #             parent = 'single'  # parent selection method: 'single' or 'weighted'
    #             x = np.loadtxt('evolve.txt', ndmin=2)
    #             n = min(5, len(x))  # number of previous results to consider
    #             x = x[np.argsort(-fitness(x))][:n]  # top n mutations
    #             w = fitness(x) - fitness(x).min()  # weights
    #             if parent == 'single' or len(x) == 1:
    #                 # x = x[random.randint(0, n - 1)]  # random selection
    #                 x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
    #             elif parent == 'weighted':
    #                 x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination
    #
    #             # Mutate
    #             mp, s = 0.8, 0.2  # mutation probability, sigma
    #             npr = np.random
    #             npr.seed(int(time.time()))
    #             g = np.array([x[0] for x in meta.values()])  # gains 0-1
    #             ng = len(meta)
    #             v = np.ones(ng)
    #             while all(v == 1):  # mutate until a change occurs (prevent duplicates)
    #                 v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
    #             for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
    #                 hyp[k] = float(x[i + 7] * v[i])  # mutate
    #
    #         # Constrain to limits
    #         for k, v in meta.items():
    #             hyp[k] = max(hyp[k], v[1])  # lower limit
    #             hyp[k] = min(hyp[k], v[2])  # upper limit
    #             hyp[k] = round(hyp[k], 5)  # significant digits
    #
    #         # Train mutation
    #         results = train(hyp.copy(), opt, device)
    #
    #         # Write mutation results
    #         print_mutation(hyp.copy(), results, yaml_file, opt.bucket)
    #
    #     # Plot results
    #     plot_evolution(yaml_file)
    #     print(f'Hyperparameter evolution complete. Best results saved as: {yaml_file}\n'
    #           f'Command to train a new model with these hyperparameters: $ python train.py --hyp {yaml_file}')
