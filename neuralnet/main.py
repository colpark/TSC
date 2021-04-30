import os
import cv2
import argparse
import numpy as np
import nibabel as nib

import wandb

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import torchvision.transforms as transforms

from dataloader import *
from utils import *
from unet import *

def main(config):
    # Create model saving path
    if not os.path.exists(os.path.join(config.model_save_dir, config.version)):
        os.makedirs(os.path.join(config.model_save_dir, config.version))
    if not os.path.exists(config.result_save_dir):
        os.makedirs(config.result_save_dir)

    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda:{}".format(config.n_cuda))
    all_idx = [2, 3, 10, 12, 15, 20, 29, 39, 40, 42, 44, 45, 46, 47, 50, 52, 59, 62, 63, 66, 68]

    ### Leave-one-out validation
    if not (config.infer or config.validate or config.test):
        wandb.init(project="TSC")        
        # from fold_start to all cross validation
        for k in range(config.fold_start, len(all_idx)):
            model = UNet(in_channels=2, out_channels=config.n_class, init_features=config.init_features)
            model.to(device)
            print('Number of model parameters: {}'.format(count_parameters(model)))
            
            train_idx = all_idx[:k] + all_idx[k+1:]
            test_idx = [(all_idx[k])]
            train_loader, test_loader = MRI_Loader(config.data_dir,
                                                   config.mask_version,
                                                   train_idx,
                                                   test_idx,
                                                   coronal = config.coronal,
                                                   batch_size = config.batch_size,
                                                   modality=config.modality,
                                                   cystic=config.cystic)

            ### Training
            train(config, model, device, train_loader, test_loader, k, wandb)

    ### Sampling with validation data
    elif config.validate:
        for k in range(len(all_idx)):
            model = UNet(in_channels=2, out_channels=config.n_class, init_features=config.init_features)
            model.to(device)
            print('Number of model parameters: {}'.format(count_parameters(model)))
            validate(config, device, model, all_idx, k)
    
    ### Sampling with unseen or selected data
    elif config.infer:        
        for sub_id in ['sub1']:
            for k in range(len(all_idx)):
                model = UNet(in_channels=2, out_channels=config.n_class, init_features=config.init_features)
                model.to(device)
                print('Number of model parameters: {}'.format(count_parameters(model)))
                infer(config, device, model, sub_id, k)
    
    ### Sampling with test data
    elif config.test:
        for k in range(len(all_idx)):
            model = UNet(in_channels=2, out_channels=config.n_class, init_features=config.init_features)
            model.to(device)
            print('Number of model parameters: {}'.format(count_parameters(model)))
            
                # All the processed brain, T1 and FLAIR
            import json
            with open('/home/socrates/david/tsc/testset/test_data_dict_ready', 'r') as f:
                test_data_dict = json.load(f)

            for sid in tqdm(test_data_dict.keys()):
                test(config, test_data_dict, sid, device, model, all_idx, k)

    return 0



def train(config, model, device, train_loader, test_loader, k,  wandb):
    if not os.path.exists(os.path.join(config.model_save_dir, config.version, 'fold{}'.format(k))):
        os.makedirs(os.path.join(config.model_save_dir, config.version, 'fold{}'.format(k)))

    best_loss = np.array([0,0,0]).astype('float32')
    best_list = {0:'0',1:'1',2:'2'}

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 40], gamma=config.gamma,
                                                     last_epoch=-1)

    for epoch in range(config.total_epoch):
        model.train()
        for i, (img, mask) in enumerate(train_loader):
            if config.multimodal_dropout:
                img = multimodal_dropout(img, config.both_prob)

            # Move tensors to the configured device
            img = img.to(device)
            mask = mask.to(device).squeeze(1)
            outputs = model(img)
            loss = criterion(outputs, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, config.total_epoch, i + 1, len(train_loader), loss.item()))

        model.eval()

        val_CEloss = AverageMeter()
        val_jaccard = AverageMeter()
        val_dice = AverageMeter()
        val_hd = AverageMeter()

        for i, (img, mask) in enumerate(test_loader):
            # Move tensors to the configured device

            img = img.to(device)
            mask = mask.to(device).squeeze(1)
            # Forward pass
            outputs = model(img)
            loss = criterion(outputs, mask)

            #========= Computing the metrics ==========#
            n = mask.size(0)
            val_CEloss.update(loss.item(), n)
            output_1dim = (F.softmax(outputs, dim=1)[:, 1] > 1 / config.n_class)

            val_jaccard.update(JACCARD_new(mask.detach().cpu(), output_1dim.detach().cpu()), n)
            val_dice.update(DICE_new(mask.detach().cpu(), output_1dim.detach().cpu()), n)

            # hausedorff metric
            with torch.no_grad():
                # defalut using compute_dtm; however, compute_dtm01 is also worth to try;
                gt_dtm_npy = compute_dtm(mask.squeeze().cpu().numpy(), outputs.shape)
                gt_dtm = torch.from_numpy(gt_dtm_npy).float().cuda(outputs.device.index)
                seg_dtm_npy = compute_dtm(outputs[:, 1, :, :].cpu().numpy() > 0.5, outputs.shape)
                seg_dtm = torch.from_numpy(seg_dtm_npy).float().cuda(outputs.device.index)
            val_hd.update(hd_loss(outputs, mask.squeeze(), seg_dtm, gt_dtm).item(), n)

            # ==========================================#

        scheduler.step()
        wandb.log({'fold{}, CE loss'.format(k): val_CEloss.avg,
                   'fold{}, Jaccard'.format(k): val_jaccard.avg,
                   'fold{}, Dice'.format(k): val_dice.avg,
                   'fold{}, Hausedorff'.format(k): val_hd.avg})

        print('FOLD{}, VALIDATION Epoch [{}/{}], Loss: {:.4f}, '
              'Jaccard: {:.4f}, Dice: {:.4f}, hd: {:.4f}'.format(k, epoch + 1, config.total_epoch,
                                                                      val_CEloss.avg,
                                                                      val_jaccard.avg,
                                                                      val_dice.avg,
                                                                      val_hd.avg))

        # Model save code
        if np.array(best_loss).min() < val_dice.avg :

            model_path = os.path.join(config.model_save_dir, config.version, 'fold{}'.format(k),
                                      'Epoch{}_Loss{:.4f}_Jac{:.4f}_Dice{:.4f}_hd{:.4f}.ckpt'.format(epoch + 1,
                                                                                                val_CEloss.avg,
                                                                                                val_jaccard.avg,
                                                                                                val_dice.avg,
                                                                                                val_hd.avg))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict()}, model_path)
            b_idx = np.argmin(best_loss)

            # Remove the model not in the lowest 5 models
            if os.path.exists(best_list[b_idx]):
                os.remove(best_list[b_idx])

            best_list[b_idx] = model_path
            best_loss[b_idx] = val_dice.avg

    return 0

def infer(config, device, model, sub_id, k):
    """
    Given a pre-processed nifti file, generate a mask nifti file in the same spatial size as the input
    """

    sample_dataset = Infer_Dataset(sub_id)
    sample_loader = torch.utils.data.DataLoader(dataset=sample_dataset,
                                              batch_size=config.batch_size,
                                              num_workers=0,
                                              drop_last=False,
                                              shuffle=False)

        # loading the checkpoint
    model_path = os.path.join(config.model_save_dir, config.version, 'fold{}'.format(k))
    model_chosen = choose_model(model_path)
    checkpoint = torch.load(model_chosen)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    print('loading {}, epoch: {}'.format(model_chosen, epoch))

    out_img = np.ones(sample_dataset.np_fl.shape)
    temp_img = np.ones((out_img.shape[2], 256, 256))
    count = 0
    for i, (img) in enumerate(sample_loader):
        img = img.to(device)
        outputs = model(img)
        outputs = F.softmax(outputs, dim=1)
        for j in range(img.size(0)):
            temp_img[count,:,:] = (outputs[j][1] > 1 / config.n_class).detach().cpu().numpy()
            count += 1

    out_img = transforms.Resize((out_img.shape[0], out_img.shape[1]), interpolation=0)(torch.from_numpy(temp_img)).permute(1,2,0)

    result_img = nib.Nifti1Image(out_img.numpy().astype('int32'), sample_dataset.meta.affine, sample_dataset.meta.header)
    filename = sample_dataset.fl.split('/')[-1][:-7]

    save_dir = os.path.join(config.result_save_dir, 'infer', '{}/{}/{}'.format( sub_id, config.version, k))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    nib.save(result_img, os.path.join(save_dir, '{}_predictedmask.nii'.format(filename)))
    return 0

def validate(config, device, model, all_idx, k):
    """
    Given a pre-processed nifti file, generate a mask nifti file in the same size as input
    """

    sample_dataset = TSC_Dataset(data_dir=config.data_dir,
                               subject_idx=[all_idx[k]],
                               mask_version=config.mask_version,
                               coronal=config.coronal,
                               modality=config.modality,
                               train=False)

    sample_loader = torch.utils.data.DataLoader(dataset=sample_dataset,
                                              batch_size=config.batch_size,
                                              num_workers=0,
                                              drop_last=False,
                                              shuffle=False)

        # loading the checkpoint
    model_path = os.path.join(config.model_save_dir, config.version, 'fold{}'.format(k))
    model_chosen = choose_model(model_path)
    checkpoint = torch.load(model_chosen)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    print('loading {}, epoch: {}'.format(model_chosen, epoch))

    out_img = np.ones(sample_dataset.np_fl.shape)
    temp_img = np.ones((out_img.shape[2], 256, 256))
    count = 0
    for i, (img, _) in enumerate(sample_loader):
        img = img.to(device)
        outputs = model(img)
        outputs = F.softmax(outputs, dim=1)
        for j in range(img.size(0)):
            temp_img[count,:,:] = (outputs[j][1] > 1 / config.n_class).detach().cpu().numpy()
            count += 1

    out_img = transforms.Resize((out_img.shape[0], out_img.shape[1]), interpolation=0)(torch.from_numpy(temp_img)).permute(1,2,0)

    result_img = nib.Nifti1Image(out_img.numpy().astype('int32'), sample_dataset.meta.affine, sample_dataset.meta.header)
    filename = sample_dataset.fl.split('/')[-1][:-7]

    save_dir = os.path.join(config.result_save_dir, 'val', '{}/{}'.format(config.version, all_idx[k]))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    nib.save(result_img, os.path.join(save_dir, '{}_predictedmask.nii'.format(filename)))

def test(config, test_data_dict, sid, device, model, all_idx, k):
    """
    Given a pre-processed nifti file, generate a mask nifti file in the same size as input
    """

    sample_dataset = Test_Dataset(data_dir=config.data_dir,
                               test_data = test_data_dict[sid],
                               modality=config.modality,
                               coronal=config.coronal)

    sample_loader = torch.utils.data.DataLoader(dataset=sample_dataset,
                                              batch_size=config.batch_size,
                                              num_workers=0,
                                              drop_last=False,
                                              shuffle=False)

        # loading the checkpoint
    model_path = os.path.join(config.model_save_dir, config.version, 'fold{}'.format(k))
    model_chosen = choose_model(model_path)
    checkpoint = torch.load(model_chosen)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    print('loading {}, epoch: {}'.format(model_chosen, epoch))

    out_img = np.ones(sample_dataset.np_fl.shape)
    temp_img = np.ones((out_img.shape[2], 256, 256))
    count = 0
    for i, (img, _) in enumerate(sample_loader):
        img = img.to(device)
        outputs = model(img)
        outputs = F.softmax(outputs, dim=1)
        for j in range(img.size(0)):
            temp_img[count,:,:] = (outputs[j][1] > 1 / config.n_class).detach().cpu().numpy()
            count += 1

    out_img = transforms.Resize((out_img.shape[0], out_img.shape[1]), interpolation=0)(torch.from_numpy(temp_img)).permute(1,2,0)

    result_img = nib.Nifti1Image(out_img.numpy().astype('int32'), sample_dataset.meta.affine, sample_dataset.meta.header)
    filename = sample_dataset.fl.split('/')[-1][:-7]
    
    save_dir = os.path.join(config.result_save_dir, 'test', '{}/{}/{}'.format(config.version, all_idx[k], sid))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    nib.save(result_img, os.path.join(save_dir, '{}_predictedmask.nii'.format(filename)))

    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/socrates/david/tsc/codebase/data')
    parser.add_argument('--model_save_dir', type=str, default='/home/socrates/david/tsc/codebase/results/models')
    parser.add_argument('--result_save_dir', type=str, default='/home/socrates/david/tsc/codebase/results/samples')
    parser.add_argument('--version', type=str, default='v2')
    parser.add_argument('--mask_version', type=str, default='v3')
    parser.add_argument('--fold_start', type=int, default=0)
    parser.add_argument('--modality', type=str, default='both')
    parser.add_argument('--coronal', type=bool, default=False)
    parser.add_argument('--cystic', type=bool, default=False)
    parser.add_argument('--init_features', type=int, default=32)

    # Multimodal dropout
    parser.add_argument('--multimodal_dropout', type=bool, default=False)
    parser.add_argument('--both_prob', type=float, default=0.5)

    # Sampling and inference
    parser.add_argument('--test', type=bool, default=False)
    parser.add_argument('--validate', type=bool, default=False)
    parser.add_argument('--infer', type=bool, default=False)
    parser.add_argument('--model_ckpt', type=str, default='')

    # Multiclass
    parser.add_argument('--n_class', type=int, default=2)

    # TSCseg_Epoch41_DiceLoss0.4494.ckpt
    parser.add_argument('--total_epoch', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_cuda', type=int, default=1)
    parser.add_argument('--print_iter', type=int, default=10)

    config = parser.parse_args()
    print(config)
    main(config)