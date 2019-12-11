from __future__ import print_function

import random
import os
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as datautils
import torchvision.transforms as transforms
import blend
import common
import data
import models as models

import time


args = {

    'sun_root': '/path/to/background_dataset',
    'voc_root': '/path/to/reflection_dataset',

    'test_root': '/path/to/sirr_test_dataset',
    'val_root': './demo/input',
    'edge_R': './demo/edge_R',
    'edge_B': './demo/edge_B',

    'chkpt_root': 'chkpt_root',

    'pretrained_vgg16': '/path/to/vgg16_model',
    'saved_logs': 'logs',

    'saved_deblend_pth': '',
    'saved_Inp_pth': '',
    'saved_Dis_pth': '',

    'start_epoch': 1,
    'cuda_device': 0,
    'batch_size': 6,
    'num_workers': 8,
    'num_epochs': 100,

    'save_frequency': 1,
    'print_frequency': 50,

    'partial_swap_ratio': 1.0/3.0,
    'fgbg_swap_rate': 0.5,
    'edge_swap_rate': 0.7,

    'num_training': 1000,  # 1000
    'num_samples': 50,
    'input_resolution': 256,

    'deblend_lr': 5e-4,
    'critic_lr': 5e-4,

    'weight_adver': 1e-3,
    'weight_vgg16': 1e-1,
    'weight_recon': 1.0,

    'weight_ipix': 8e-1,
    'weight_ipec': 2e-1,
    'weight_iadv': 1e-3,
    'weight_ireg': 1e-7,
}


def get_loaders():
    image_size = args['input_resolution']
    transform = {
        'scale_size': image_size + args['input_resolution'] / 4,
        'crop_size': image_size,
        'horizontal_flip': True
    }

    # lsun_classes = ['bridge', 'church_outdoor', 'tower']
    sun_train_factor = 0.8
    voc_train_factor = 0.8

    sun_train_dataset, sun_val_dataset = data.get_sun_datasets(args['sun_root'], sun_train_factor, transform)
    voc_train_dataset, voc_val_dataset = data.get_voc_datasets(args['voc_root'], voc_train_factor, transform)

    min_length = args['num_training'] * args['batch_size']
    bg_train_dataset = data.MixedDataset([sun_train_dataset], min_length, equalize=True)
    bg_val_dataset = data.MixedDataset([sun_val_dataset], min_length, equalize=True)
    fg_train_dataset = data.MixedDataset([voc_train_dataset], min_length, equalize=True)
    fg_val_dataset = data.MixedDataset([voc_val_dataset], min_length, equalize=True)

    bg_train_loader, bg_val_loader, fg_train_loader, fg_val_loader = map(
        lambda dataset: datautils.DataLoader(dataset, batch_size=args['batch_size'], shuffle=True,
                                             num_workers=args['num_workers']),
        [bg_train_dataset, bg_val_dataset, fg_train_dataset, fg_val_dataset]
    )
    return bg_train_loader, bg_val_loader, fg_train_loader, fg_val_loader

def val_loaders():
    transform = None

    real_val_dataset = data.get_val_datasets(args['val_root'])
    edgeR_val_dataset = data.get_val_datasets(args['edge_R'])
    edgeB_val_dataset = data.get_val_datasets(args['edge_B'])

    min_length = 1
    real_val_dataset = data.MixedDataset([real_val_dataset], min_length, equalize=True)
    edgeR_val_dataset = data.MixedDataset([edgeR_val_dataset], min_length, equalize=True)
    edgeB_val_dataset = data.MixedDataset([edgeB_val_dataset], min_length, equalize=True)


    real_val_loader, edgeR_val_loader, edgeB_val_loader = map(
        lambda dataset: datautils.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args['num_workers']),
        [real_val_dataset, edgeR_val_dataset, edgeB_val_dataset]
    )
    return real_val_loader, edgeR_val_loader, edgeB_val_loader

def test_loaders():
    transform = None

    _, sirr_val_dataset = data.get_SIRR_datasets(args['test_root'], transform)
    # x, y, z = sirr_train_dataset.__getitem__(1)
    # # print(x.type)

    min_length = args['num_training'] * args['batch_size']
    sirr_val_dataset = data.MixedDataset([sirr_val_dataset], min_length, equalize=False)

    sirr_val_loader = datautils.DataLoader(sirr_val_dataset, batch_size=1, shuffle=True,
                                             num_workers=args['num_workers'])

    return sirr_val_loader


def build_networks():
    vgg16_net = models.VGG16Net().cuda()
    deblend_net = models.multiDeblendNet().cuda()
    Inp_net = models.InpNet().cuda()
    Dis_net = models.CriticNet().cuda()

    # start learning from breakpoint
    chkpt_dir = os.path.join(args['chkpt_root'], 'samples')
    if os.path.exists(chkpt_dir):
        folder_max = 0
        for i in os.listdir(chkpt_dir):
            i = int(i)
            folder_max = max(folder_max, i)

        if folder_max > 0:
            args['start_epoch'] = folder_max +1
            args['saved_deblend_pth'] = os.path.join(chkpt_dir, str(folder_max), 'models/deblend.pth')
            args['saved_Inp_pth'] = os.path.join(chkpt_dir, str(folder_max), 'models/Inp.pth')
            args['saved_Dis_pth'] = os.path.join(chkpt_dir, str(folder_max), 'models/Dis.pth')
            print('load pth fold: ', os.path.join(chkpt_dir, str(folder_max)))

    if args['pretrained_vgg16']:
        vgg16_net.load_pretrained(args['pretrained_vgg16'])
        print('load vgg16 sucess!')
    if args['saved_deblend_pth']:
        deblend_net.load_state_dict(torch.load(args['saved_deblend_pth']))
        print('load deblend_net sucess!')
    if args['saved_Inp_pth']:
        Inp_net.load_state_dict(torch.load(args['saved_Inp_pth']))
        print('load Inp_net sucess!')
    if args['saved_Dis_pth']:
        Dis_net.load_state_dict(torch.load(args['saved_Dis_pth']))
        print('load Dis_net sucess!')


    return vgg16_net, deblend_net, Inp_net, Dis_net


def get_optimizers(deblend_net, Inp_net, Dis_net):
    deblend_optimizer = optim.Adam(deblend_net.parameters(), lr=args['deblend_lr'])
    Inp_optimizer = optim.Adam(Inp_net.parameters(), lr=args['deblend_lr'])
    Dis_optimizer = optim.Adam(Dis_net.parameters(), lr=args['critic_lr'])

    return deblend_optimizer, Inp_optimizer, Dis_optimizer


def normalize_for_vgg16(xs):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(xs[0])
    std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(xs[0])
    mean = autograd.Variable(mean).cuda()
    std = autograd.Variable(std).cuda()

    return [(x - mean) / std for x in xs]


def calc_vgg16_loss(vgg16_net, criterion, x, y):
    x, y = normalize_for_vgg16([x, y])
    x_feat, y_feat = [vgg16_net(z) for z in (x, y)]
    x_feat = x_feat.detach()
    loss = criterion(y_feat, x_feat)
    return loss


def train(bg_loader, fg_loader, mask_loader, vgg16_net, deblend_net,Inp_net, Dis_net,
          deblend_optimizer, Inp_optimizer, Dis_optimizer, epoch):
    B_loss = common.AverageMeter()
    I_loss = common.AverageMeter()
    D_loss = common.AverageMeter()

    deblend_net.train()
    Inp_net.train()
    Dis_net.train()

    # real means original
    # fake means deblended
    real_label = torch.ones(args['batch_size'])
    fake_label = torch.zeros(args['batch_size'])

    real_label = autograd.Variable(real_label).cuda()
    fake_label = autograd.Variable(fake_label).cuda()

    critic_criterion = nn.BCELoss().cuda()
    mse_criterion = nn.MSELoss().cuda()

    bg_iter = iter(bg_loader)
    fg_iter = iter(fg_loader)

    batch_count = 0
    total_count = args['num_training']
    while batch_count < total_count:

        batch_count += 1

        if random.random() > args['fgbg_swap_rate']:
            x3 = next(bg_iter)
            x4 = next(fg_iter)
        else:
            x3 = next(fg_iter)
            x4 = next(bg_iter)

        if x3.size(0) != x4.size(0):
            break
        #############################################################################
        #
        # image systhesis
        #
        #############################################################################
        mask = torch.zeros(args['batch_size'], 3, args['input_resolution'], args['input_resolution'])
        for b in range(0, args['batch_size']):
            mask_num = mask_loader.size()[0]
            mask_id_1 = random.randint(0, mask_num - 1)
            mask_id_2 = random.randint(0, mask_num - 1)
            mask[b] = torch.clamp(mask_loader[mask_id_1] + mask_loader[mask_id_2], min=0, max=1.0)
        x, x1, x2 = blend.blend_gauss_part(x3, x4, mask)

        if epoch < 20:
            scale_lower = 1.0 - 0.5*(epoch/20.0)
            th_upper = 0.5 + 0.5*(epoch/20.0)
        else:
            scale_lower = 0.5
            th_upper = 1.0

        scale_ratio = random.uniform(scale_lower, 1.0)
        th_ratio = random.uniform(0.5, th_upper)

        partial_ratio = random.random()
        # print(partial_ratio, args['partial_swap_ratio'])
        if partial_ratio < args['partial_swap_ratio']:
            edge_x1 = torch.ones(x.size(0), 1, x.size(2), x.size(3))
            edge_x2 = blend.H_map(x2, e_type='canny', th_ratio=th_ratio)
            edge_x2 = blend.Partial_Map(edge_x2, size_ratio=1.0, scale_ratio=scale_ratio)
        elif partial_ratio > 2.0*args['partial_swap_ratio']:
            edge_x1 = blend.H_map(x1, e_type='canny', th_ratio=th_ratio)
            edge_x1 = blend.Partial_Map(edge_x1, size_ratio=1.0, scale_ratio=scale_ratio)
            edge_x2 =  torch.ones(x.size(0), 1, x.size(2), x.size(3))
        else:
            edge_x1 = blend.H_map(x1, e_type='canny', th_ratio=th_ratio)
            edge_x1 = blend.Partial_Map(edge_x1, size_ratio=1.0, scale_ratio=scale_ratio)
            edge_x2 = blend.H_map(x2, e_type='canny', th_ratio=th_ratio)
            edge_x2 = blend.Partial_Map(edge_x2, size_ratio=1.0, scale_ratio=scale_ratio)
        edge_x1, edge_x2 = blend.Edge_Sparse(edge_x1, edge_x2, mask)

        x1, x2, x3, x4, x, mask, edge_x1, edge_x2 = [autograd.Variable(z).cuda() for z in (x1, x2, x3, x4, x, mask, edge_x1, edge_x2)]

        #############################################################################
        #
        # train deblend_net
        #
        #############################################################################
        deblend_optimizer.zero_grad()

        y1, y2, Rmap = deblend_net(x, edge_x1, edge_x2)

        # train with vgg16
        vgg16_loss_2 = calc_vgg16_loss(vgg16_net, mse_criterion, x2, y2)
        vgg16_loss_3 = calc_vgg16_loss(vgg16_net, mse_criterion, x1, y1)
        error_vgg16 = args['weight_vgg16'] * (vgg16_loss_2 + vgg16_loss_3)

        # train with pixel
        pixel_loss_2 = mse_criterion(y2, x2)
        pixel_loss_3 = mse_criterion(y1, x1)
        pixel_loss_4 = mse_criterion(Rmap, mask.mean(1))
        error_pixel = args['weight_recon'] * (pixel_loss_2 + pixel_loss_3 + pixel_loss_4)

        deblend_loss = error_vgg16 + error_pixel
        deblend_loss.backward(retain_graph=True)

        deblend_optimizer.step()

        #############################################################################
        #
        # train inp_net
        #
        #############################################################################
        Inp_optimizer.zero_grad()

        # y1, y2 = y1.detach(), y2.detach()

        gray_y2 = blend.ToGray(y2, True)
        y1_1 = torch.cat((y1, gray_y2), 1)

        y3_f = Inp_net(y1_1)
        ap_mask = blend.ToGray(Rmap.expand_as(y3_f), True)
        y3_f = y3_f*ap_mask
        y3 = y3_f + y1

        # train with critic
        fake_pred = Dis_net(y3_f)
        error_dis = args['weight_iadv'] * critic_criterion(fake_pred, real_label)

        # train with vgg16
        vgg16_loss_inp = calc_vgg16_loss(vgg16_net, mse_criterion, x3, y3)
        error_vgg16_inp = args['weight_ipec'] * vgg16_loss_inp

        # train with pixel
        pixel_loss_inp = mse_criterion(y3, x3)
        error_pixel_inp = args['weight_ipix'] * pixel_loss_inp

        inp_loss = error_dis + error_vgg16_inp + error_pixel_inp
        inp_loss.backward()

        Inp_optimizer.step()

        #############################################################################
        #
        # train Dis_net refinement
        #
        #############################################################################
        Dis_optimizer.zero_grad()

        # train with real
        x3_f = x3 - y1
        real_pred = Dis_net(x3_f.detach())  # t
        dis_real = critic_criterion(real_pred, real_label)

        # train with fake
        fake_pred = Dis_net(y3_f.detach())
        dis_fake = critic_criterion(fake_pred, fake_label)

        dis_loss = dis_real + dis_fake
        dis_loss.backward()

        Dis_optimizer.step()

        #############################################################################
        #
        # update loss
        #
        #############################################################################
        B_loss.update(deblend_loss.data[0], x1.size(0))
        I_loss.update(inp_loss.data[0], x1.size(0))
        D_loss.update(dis_loss.data[0], x1.size(0))

        if batch_count % args['print_frequency'] == 1:
            print('    epoch: {}   train: {}/{}\t'
                '    deblend_loss: {deblend_loss.avg:.4f}\t'
                '    inp_loss: {inp_loss.avg:.4f}   dis_loss: {dis_loss.avg:.4f}'.format(epoch, batch_count, total_count,
                                                                                 deblend_loss=B_loss, inp_loss=I_loss, dis_loss=D_loss))


def sample(bg_loader, fg_loader, raw_val_loader, edgeR_val_loader, edgeB_val_loader, sirr_test_loader, mask_loader, deblend_net, Inp_net, epoch):
    deblend_net.eval()
    Inp_net.eval()

    transform = transforms.Compose([
        transforms.Lambda(lambda z: z.clamp(0, 1)),
        transforms.ToPILImage(),
    ])

    image_samples = []
    bg_iter = iter(bg_loader)
    fg_iter = iter(fg_loader)

    raw_iter = iter(raw_val_loader)
    edgeR_iter = iter(edgeR_val_loader)
    edgeB_iter = iter(edgeB_val_loader)

    sirr_iter = iter(sirr_test_loader)
    #############################################################################
    #
    # test part
    #
    #############################################################################
    # print('==========> testing <==========')
    # NCC = 0.0
    # sLMSE = 0.0
    # SSIM = 0.0
    # SI = 0.0
    #
    # NCC2 = 0.0
    # sLMSE2 = 0.0
    # SSIM2 = 0.0
    # SI2 = 0.0
    #
    # val_count = 0
    # total_val = len(sirr_val_loader)
    # while val_count < total_val:
    #     val_count += 1
    #
    #     x, x1, x2 = next(sirr_iter)
    #
    #     assert x.size(2) in (400, 540)
    #     if x.size(2) == 400 and x.size(3) == 540:
    #         x_pad = torch.ones(x.size(0), x.size(1), x.size(2), 4)
    #         x1 = torch.cat((x1, x_pad), 3)
    #         x2 = torch.cat((x2, x_pad), 3)
    #         x = torch.cat((x, x_pad), 3)
    #     elif x.size(2) == 540 and x.size(3) == 400:
    #         x_pad = torch.ones(x.size(0), x.size(1), 4, x.size(3))
    #         x1 = torch.cat((x1, x_pad), 2)
    #         x2 = torch.cat((x2, x_pad), 2)
    #         x = torch.cat((x, x_pad), 2)
    #
    #     edge_x1 = blend.H_map(x1, e_type='canny', th_ratio=0.5)
    #     # edge_x1 = blend.Partial_Map(edge_x1, size_ratio=1.0, scale_ratio=1.0)
    #     edge_x2 = blend.H_map(x2, e_type='canny', th_ratio=0.5)
    #     # edge_x2 = blend.Partial_Map(edge_x2, size_ratio=1.0, scale_ratio=1.0)
    #     # h = torch.cat((x, edge_x1), 1)
    #     # h = torch.cat((h, edge_x2), 1)
    #
    #     x1, x2, x, edge_x1, edge_x2 = [autograd.Variable(z).cuda() for z in (x1, x2, x, edge_x1, edge_x2)]
    #
    #     #############################################################################
    #     #
    #     # train deblend_net
    #     #
    #     #############################################################################
    #     y1, y2, Rmap = deblend_net(x, edge_x1, edge_x2)
    #
    #     gray_y2 = blend.ToGray(y2, True)
    #     y1_1 = torch.cat((y1, gray_y2), 1)
    #
    #     y3_f = Inp_net(y1_1)
    #     ap_mask = blend.ToGray(Rmap.expand_as(y3_f), True)
    #     y3_f = y3_f*ap_mask
    #     y3 = y3_f + y1
    #
    #     if x.size(2) == 400 and x.size(3) == 540:
    #         x1, x2, y1, y2, y3 = [z[:, :, :, 0:540] for z in (x1, x2, y1, y2, y3)]
    #     elif x.size(2) == 540 and x.size(3) == 400:
    #         x1, x2, y1, y2, y3 = [z[:, :, 0:540, :] for z in (x1, x2, y1, y2, y3)]
    #     #############################################################################
    #     #
    #     # validate metrics
    #     #
    #     #############################################################################
    #     ssimMetric, siMetric = common.SSIM(x1, y3)
    #     nccMetric = common.NCC(x1, y3)
    #     slmseMetric = common.sLMSE(x1, y3, x2, y2)
    #     NCC += nccMetric.data[0]
    #     sLMSE += slmseMetric.data[0]
    #     SSIM += ssimMetric.data[0]
    #     SI += siMetric.data[0]
    #
    #     ssimMetric2, siMetric2 = common.SSIM(x1, y1)
    #     nccMetric2 = common.NCC(x1, y1)
    #     slmseMetric2 = common.sLMSE(x1, y1, x2, y2)
    #     NCC2 += nccMetric2.data[0]
    #     sLMSE2 += slmseMetric2.data[0]
    #     SSIM2 += ssimMetric2.data[0]
    #     SI2 += siMetric2.data[0]
    #
    #     # print('    val_epoch: {}\t'
    #     #       '    ssimMetric: {ssimMetric:.4f}   siMetric: {siMetric:.4f}\t'
    #     #       '    nccMetric: {nccMetric:.4f}   slmseMetric: {slmseMetric:.4f}\t'.format(epoch,
    #     #                                                                                  ssimMetric=ssimMetric.data[0],
    #     #                                                                                  siMetric=siMetric.data[0],
    #     #                                                                                  nccMetric=nccMetric.data[0],
    #     #                                                                                  slmseMetric=slmseMetric.data[0]))
    # #############################################################################
    # #
    # # # ============ TensorBoard logging ============#
    # #
    # #############################################################################
    # info = {
    #
    #     'ssimMetric': float(SSIM)/total_val,
    #     'siMetric': float(SI)/total_val,
    #     'nccMetric': float(NCC)/total_val,
    #     'slmseMetric': float(sLMSE)/total_val,
    #
    #     'ssimMetric2': float(SSIM2) / total_val,
    #     'siMetric2': float(SI2) / total_val,
    #     'nccMetric2': float(NCC2) / total_val,
    #     'slmseMetric2': float(sLMSE2) / total_val,
    # }
    #
    # for tag, value in info.items():
    #         logger.scalar_summary(tag, value, epoch)
    #
    # print('    val_epoch: {}\t'
    #         '    ssimMetric: {ssimMetric:.4f}   siMetric: {siMetric:.4f}\t'
    #         '    nccMetric: {nccMetric:.4f}   slmseMetric: {slmseMetric:.4f}\t'.format(epoch,
    #                                                                                      ssimMetric=float(SSIM)/total_val,
    #                                                                                      siMetric=float(SI)/total_val,
    #                                                                                      nccMetric=float(NCC)/total_val,
    #                                                                                      slmseMetric=float(sLMSE)/total_val))
    #############################################################################
    #
    # validate and display part
    #
    #############################################################################
    print('==========> validation <==========')
    print('            emmm...            ')
    batch_count = 0
    test_count = 0
    total_count = min(len(bg_loader), len(fg_loader))
    while batch_count < total_count:
        iter_source = 'real'
        if random.random() > 0.3 or test_count >= len(raw_val_loader):
            iter_source = 'synthesis'

        batch_count += 1
        # start_time = time.time()

        if iter_source == 'real':
            test_count += 1
            x = next(raw_iter)
            edge_x_R = next(edgeR_iter)
            edge_x_B = next(edgeB_iter)

            edge_x1, edge_x2 = torch.mean(edge_x_B, 1).unsqueeze(0), torch.mean(edge_x_R, 1).unsqueeze(0)

            if epoch < 20:
                scale_lower = 1.0 - 0.5 * (epoch / 20.0)
            else:
                scale_lower = 0.5
            scale_ratio = random.uniform(scale_lower, 1.0)
            partial_ratio = random.random()
            if partial_ratio < args['partial_swap_ratio']:
                edge_x1 = torch.ones(x.size(0), 1, x.size(2), x.size(3))
                edge_x2 = blend.Partial_Map(edge_x2, size_ratio=1.0, scale_ratio=scale_ratio)
            elif partial_ratio > 2.0 * args['partial_swap_ratio']:
                edge_x1 = blend.Partial_Map(edge_x1, size_ratio=1.0, scale_ratio=scale_ratio)
                edge_x2 = torch.ones(x.size(0), 1, x.size(2), x.size(3))
            else:
                edge_x1 = blend.Partial_Map(edge_x1, size_ratio=1.0, scale_ratio=scale_ratio)
                edge_x2 = blend.Partial_Map(edge_x2, size_ratio=1.0, scale_ratio=scale_ratio)

            x1 = x
            x2 = x

            x, edge_x1, edge_x2 = [autograd.Variable(z).cuda() for z in (x, edge_x1, edge_x2)]

            #############################################################################
            #
            # train deblend_net
            #
            #############################################################################
            y1, y2, Rmap = deblend_net(x, edge_x1, edge_x2)

            gray_y2 = blend.ToGray(y2, True)
            y1_1 = torch.cat((y1, gray_y2), 1)

            y3_f = Inp_net(y1_1)
            ap_mask = blend.ToGray(Rmap.expand_as(y3_f), True)
            y3_f = y3_f * ap_mask
            y3 = y3_f + y1
            y3 = torch.clamp(y3, min=0, max=1)

        elif iter_source == 'synthesis':
            if random.random() > args['fgbg_swap_rate']:
                x3 = next(bg_iter)
                x4 = next(fg_iter)
            else:
                x3 = next(fg_iter)
                x4 = next(bg_iter)

            if x3.size(0) != x4.size(0):
                break
            #############################################################################
            #
            # image systhesis
            #
            #############################################################################
            mask = torch.zeros(args['batch_size'], 3, args['input_resolution'], args['input_resolution'])
            for b in range(0, args['batch_size']):
                mask_num = mask_loader.size()[0]
                mask_id_1 = random.randint(0, mask_num - 1)
                mask_id_2 = random.randint(0, mask_num - 1)
                mask[b] = torch.clamp(mask_loader[mask_id_1] + mask_loader[mask_id_2], min=0, max=1.0)
            x, x1, x2 = blend.blend_gauss_part(x3, x4, mask)

            if epoch < 20:
                scale_lower = 1.0 - 0.5 * (epoch / 20.0)
                th_upper = 0.5 + 0.5 * (epoch / 20.0)
            else:
                scale_lower = 0.5
                th_upper = 1.0

            scale_ratio = random.uniform(scale_lower, 1.0)
            th_ratio = random.uniform(0.5, th_upper)

            partial_ratio = random.random()
            if partial_ratio < args['partial_swap_ratio']:
                edge_x1 = torch.ones(x.size(0), 1, x.size(2), x.size(3))
                edge_x2 = blend.H_map(x2, e_type='canny', th_ratio=th_ratio)
                edge_x2 = blend.Partial_Map(edge_x2, size_ratio=1.0, scale_ratio=scale_ratio)
            elif partial_ratio > 2.0 * args['partial_swap_ratio']:
                edge_x1 = blend.H_map(x1, e_type='canny', th_ratio=th_ratio)
                edge_x1 = blend.Partial_Map(edge_x1, size_ratio=1.0, scale_ratio=scale_ratio)
                edge_x2 = torch.ones(x.size(0), 1, x.size(2), x.size(3))
            else:
                edge_x1 = blend.H_map(x1, e_type='canny', th_ratio=th_ratio)
                edge_x1 = blend.Partial_Map(edge_x1, size_ratio=1.0, scale_ratio=scale_ratio)
                edge_x2 = blend.H_map(x2, e_type='canny', th_ratio=th_ratio)
                edge_x2 = blend.Partial_Map(edge_x2, size_ratio=1.0, scale_ratio=scale_ratio)

            x, edge_x1, edge_x2 = [autograd.Variable(z).cuda() for z in (x, edge_x1, edge_x2)]

            #############################################################################
            #
            # train deblend_net
            #
            #############################################################################
            y1, y2, Rmap = deblend_net(x, edge_x1, edge_x2)

            gray_y2 = blend.ToGray(y2, True)
            y1_1 = torch.cat((y1, gray_y2), 1)

            y3_f = Inp_net(y1_1)
            ap_mask = blend.ToGray(Rmap.expand_as(y3_f), True)
            y3_f = y3_f * ap_mask
            y3 = y3_f + y1
            y3 = torch.clamp(y3, min=0, max=1)

        #############################################################################
        #
        # save sample images
        #
        #############################################################################
        x, y1, y2, y3 = map(lambda z: z.cpu().data, (x, y1, y2, y3))

        # end_time = time.time()
        # print('forward time: ', end_time - start_time)

        batch_size = x.size(0)
        for i in range(batch_size):
            images = [transform(z[i]) for z in (x1, x2, x, y1, y2, y3)]
            image_samples.append(images)
            if len(image_samples) >= args['num_samples']:
                return image_samples



def main():
    # for training
    bg_train_loader, bg_val_loader, fg_train_loader, fg_val_loader = get_loaders()
    # for testing
    sirr_test_loader = test_loaders()
    # for validation
    raw_val_loader, edgeR_val_loader, edgeB_val_loader = val_loaders()

    vgg16_net, deblend_net, Inp_net, Dis_net = build_networks()
    deblend_optimizer, Inp_optimizer, Dis_optimizer = get_optimizers(deblend_net, Inp_net, Dis_net)

    #############################################################################
    #
    # you can pre-generate the mask!
    #
    #############################################################################
    # mask_patch = blend.build_masks()
    # torch.save(mask_patch, './Mask/mask.pth')
    mask_loader = torch.load('./Mask/mask.pth')
    print('load mask success!')

    # make ckpt dir
    if not os.path.exists(args['chkpt_root']):
        os.makedirs(args['chkpt_root'])

    #############################################################################
    #
    # train & validate
    #
    #############################################################################
    for epoch in range(args['start_epoch'], args['num_epochs']):
        print('epoch: {}/{}'.format(epoch, args['num_epochs']))
        train(bg_train_loader, fg_train_loader, mask_loader, vgg16_net,
              deblend_net, Inp_net, Dis_net, deblend_optimizer, Inp_optimizer, Dis_optimizer, epoch)

        if epoch % args['save_frequency'] == 0:
            image_names = ('x1', 'x2', 'x', 'y1', 'y2', 'y3')
            image_samples = sample(bg_val_loader, fg_val_loader, raw_val_loader, edgeR_val_loader, edgeB_val_loader, sirr_test_loader, mask_loader,
                               deblend_net, Inp_net, epoch)
            common.save_checkpoint(args['chkpt_root'], epoch, image_samples, image_names, deblend_net,Inp_net, Dis_net)


if __name__ == '__main__':
    with torch.cuda.device(args['cuda_device']):
        main()
