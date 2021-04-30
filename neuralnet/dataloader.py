import os
import re
import sys
import glob
import json
import torch
import random
import numpy as np
import nibabel as nib
from tqdm import tqdm
sys.path.append('..')
from misc_utils import *
import torch.utils.data as data
import torchvision.transforms as transforms

def eye_to_zero(img):
    img[img == 7.0] = 0.0
    return img

def nonzero_to_one(img):
    img[img != 0.0] = 1.0
    return img

def znormalize(brain):
    brain = (brain - brain.mean()) / brain.std()
    return brain

def getNumbers(str):
    array = re.findall(r'[0-9]+', str)
    return int(array[0])

def augment(data):
    data = transforms.RandomHorizontalFlip(p=0.5)(data)
    data = transforms.RandomVerticalFlip(p=0.5)(data)
    angle = random.choice([1, 2, 3])
    data = torch.rot90(data, k=angle, dims=[1, 2])

    return data

def mask_search_preproc(mask_dir, s):
    found = False
    for addr_ in glob.glob(mask_dir + '/*'):
        if getNumbers(addr_.split('/')[-1]) == s:
            # make sure it's binary
            m1 = nonzero_to_one(eye_to_zero(nib.load(addr_).get_fdata()))
            found = True
            break
    return m1, found

class TSC_Dataset(data.Dataset):
    def __init__(self, data_dir, subject_idx, mask_version, input_size=256,
                 coronal = False,
                 modality = 'both',
                 cystic = False,
                 train=True):

        """
        mask_version: ['v3'~'v7']
        subject_idx: list containing subject integers
        input_size: data input size
        """

        self.train = train
        self.input_size = input_size
        self.data = []

        # All the processed brain, T1 and FLAIR
        with open('/home/socrates/david/tsc/TSCseg/T1_experiments/3.BET/data_sixfold.json', 'r') as f:
            t1_added_data = json.load(f)
            
        with open('/home/socrates/david/tsc/TSCseg/INTEG_ADDR.json', 'r') as f:
            INTEG_ADDR = json.load(f)
            
        with open('/home/socrates/david/tsc/codebase/data/NEW_INTEG.json', 'r') as f:
            NEW_INTEG = json.load(f)
        
        meta_info = misc_meta()
    

        for sid in tqdm(subject_idx):
            s = str(sid)
                            
            if mask_version == 'v3':
                mask_inter = NEW_INTEG['train'][s]['mask_orig1']
                self.np_mask = nonzero_to_one(eye_to_zero(nib.load(mask_inter).get_fdata()))
             
            elif mask_version == 'v4':
                mask_inter = NEW_INTEG['train'][s]['mask_orig2']
                self.np_mask = nonzero_to_one(eye_to_zero(nib.load(mask_inter).get_fdata()))
                
            # Using only the intersection from the two masks
            elif mask_version == 'v5':                
                mask_inter = NEW_INTEG['train'][s]['mask_inter']
                self.np_mask = nonzero_to_one(eye_to_zero(nib.load(mask_inter).get_fdata()))
                
                if cystic:
                    if s in meta_info['cysticpid']:
                        cys_mask = INTEG_ADDR['train'][s]['mask_cystic_inter']
                        cys_mask_np =  nonzero_to_one(eye_to_zero(nib.load(cys_mask).get_fdata()))
                        self.np_mask = np.logical_or(self.np_mask, cys_mask_np)

            # Using union of the two masks
            elif mask_version == 'v6':
                mask_union = NEW_INTEG['train'][s]['mask_union']
                self.np_mask = nonzero_to_one(eye_to_zero(nib.load(mask_union).get_fdata()))

            # Using both data independently
            elif mask_version == 'v7':
                
                mask_orig1 = NEW_INTEG['train'][s]['mask_orig1']
                mask_orig2 = NEW_INTEG['train'][s]['mask_orig2']
                m1 = nonzero_to_one(eye_to_zero(nib.load(mask_orig1).get_fdata()))
                m2 = nonzero_to_one(eye_to_zero(nib.load(mask_orig2).get_fdata()))
                rand_idx = random.randint(0, 1)
                if rand_idx == 0:
                    self.np_mask = m1
                else:
                    self.np_mask = m2

            else:
                raise RuntimeError('No mask found for version: {}'.format(mask_version))

            self.fl = NEW_INTEG['train'][s]['FL_brain']
            self.t1 = NEW_INTEG['train'][s]['T1_reg2FL_brain']

            # Normalization of input brain
            self.np_fl = znormalize(nib.load(self.fl).get_fdata())
            self.np_t1 = znormalize(nib.load(self.t1).get_fdata())

            # Use only t1 or flair
            if modality == 'flair':
                print('setting t1 to zero!')
                self.np_t1 = np.zeros((self.np_t1.shape))

            elif modality == 't1':
                print('setting flair to zero!')
                self.np_fl = np.zeros((self.np_fl.shape))

            elif modality == 'both':
                print('keeping both modalities!')
                pass

            # Saving the data to the memory. If data gets larger, saving it to the hard disk may be considered.
            dim_z = self.np_fl.shape[2]
            for i in range(dim_z):
                data_ = np.concatenate([np.expand_dims(self.np_fl[:, :, i], 2),
                                         np.expand_dims(self.np_t1[:, :, i], 2),
                                         np.expand_dims(self.np_mask[:, :, i], 2)
                                         ], axis=2)

                self.data.append(data_)

        # Use coronal as data augmentation, only for training
        ##############################
        if train and coronal:
            print('get coronal slides ready for training dataloader')
            for pid in meta_info['coronalpid']:
                
                np_fl = znormalize(nib.load(NEW_INTEG['train'][pid]['FL_coronal_brain']).get_fdata())
                np_t1 = znormalize(nib.load(NEW_INTEG['train'][pid]['T1_reg2FL_coronal_brain']).get_fdata())
                np_mask = nonzero_to_one(eye_to_zero(nib.load(NEW_INTEG['train'][pid]['mask_coronal']).get_fdata()))
                
                # Use only t1 or flair
                if modality == 'flair':
                    np_t1 = np.zeros((np_t1.shape))

                elif modality == 't1':
                    np_fl = np.zeros((np_fl.shape))

                elif modality == 'both':
                    pass
                
                dim_z = np_fl.shape[2]
                for i in range(dim_z):
                    data_ = np.concatenate([np.expand_dims(np_fl[:, :, i], 2),
                                            np.expand_dims(np_t1[:, :, i], 2),
                                            np.expand_dims(np_mask[:, :, i], 2)
                                            ], axis=2)
                    self.data.append(data_)
        ##############################
        
        if len(subject_idx) == 1:
            self.meta= nib.load(self.fl)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        data_np = self.data[idx]
        data_t = transforms.ToTensor()(data_np).permute(0, 1, 2)

        if self.train:
            data_t = augment(data_t)

        img = data_t[:2]
        img = transforms.Resize((self.input_size, self.input_size), interpolation=2)(img)

        mask = data_t[2].unsqueeze(0)
        mask = transforms.Resize((self.input_size, self.input_size), interpolation=0)(mask)

        return img.float(), mask.long()

def MRI_Loader(data_dir, mask_version, train_idx, test_idx, batch_size=16, coronal = False, modality = 'both', cystic = False):
    train_dataset = TSC_Dataset(data_dir=data_dir, subject_idx=train_idx, mask_version = mask_version,
                                coronal=coronal,
                                modality=modality,
                                cystic=cystic,
                                train=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               num_workers=0,
                                               drop_last=False,
                                               shuffle=True)
    print('train_loader ready')
    test_dataset = TSC_Dataset(data_dir=data_dir, subject_idx=test_idx, mask_version = mask_version,
                               coronal=coronal,
                               modality=modality,
                               cystic=cystic,
                               train=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              num_workers=0,
                                              drop_last=False,
                                              shuffle=False)
    print('test_loader ready')
    print('dataset length - train: {} test: {}'.format(len(train_dataset), len(test_dataset)))

    return train_loader, test_loader


class Infer_Dataset(data.Dataset):
    def __init__(self, sub_id, input_size=256):
        # How am i saving the intermediate results from the individual-level inference?
        with open('/home/socrates/david/tsc/codebase/data/inference/processed/infer_integ.json', 'r') as f:
            infer_integ = json.load(f)

        self.fl = infer_integ[sub_id]['FL_brain']
        self.t1 = infer_integ[sub_id]['T1_reg2FL_brain']

        # normalization
        self.np_fl = znormalize(nib.load(self.fl).get_fdata())
        self.np_t1 = znormalize(nib.load(self.t1).get_fdata())

        self.input_size = input_size
        self.meta = nib.load(self.fl)

        dim_z = self.np_fl.shape[2]
        self.data = []
        for i in range(dim_z):
            pair = np.concatenate([np.expand_dims(self.np_fl[:, :, i], 2),
                                   np.expand_dims(self.np_t1[:, :, i], 2)], axis=2)
            self.data.append(pair)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_np = self.data[idx]
        data_t = transforms.ToTensor()(data_np).permute(0, 1, 2)

        img = data_t[:2]
        img = transforms.Resize((self.input_size, self.input_size), interpolation=2)(img)
        return img.float()
    
    
class Test_Dataset(data.Dataset):
    def __init__(self, data_dir,  test_data, input_size=256, coronal = False, modality = 'both'):

        """
        input_size: data input size
        """
        self.input_size = input_size
        self.data = []        
           
        # Find the mask corresponding to the subject
        mask_dir = test_data['tuber_mask']
        self.np_mask = nonzero_to_one(eye_to_zero(nib.load(mask_dir).get_fdata()))            

        self.fl = test_data['FL_brain']
        self.t1 = test_data['T1_reg_brain']

        # Normalization of input brain
        self.np_fl = znormalize(nib.load(self.fl).get_fdata())
        self.np_t1 = znormalize(nib.load(self.t1).get_fdata())

        # Use only t1 or flair
        if modality == 'flair' or coronal:
            print('setting t1 to zero!')
            self.np_t1 = np.zeros((self.np_t1.shape))

        elif modality == 't1':
            print('setting flair to zero!')
            self.np_fl = np.zeros((self.np_fl.shape))

        elif modality == 'both':
            print('keeping both modalities!')
            pass

        # Saving the data to the memory. If data gets larger, saving it to the hard disk may be considered.
        dim_z = self.np_fl.shape[2]
        for i in range(dim_z):
            data_ = np.concatenate([np.expand_dims(self.np_fl[:, :, i], 2),
                                     np.expand_dims(self.np_t1[:, :, i], 2),
                                     np.expand_dims(self.np_mask[:, :, i], 2)
                                     ], axis=2)

            self.data.append(data_)

        
        self.meta= nib.load(self.fl)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        data_np = self.data[idx]
        data_t = transforms.ToTensor()(data_np).permute(0, 1, 2)

        img = data_t[:2]
        img = transforms.Resize((self.input_size, self.input_size), interpolation=2)(img)

        mask = data_t[2].unsqueeze(0)
        mask = transforms.Resize((self.input_size, self.input_size), interpolation=0)(mask)

        return img.float(), mask.long()
