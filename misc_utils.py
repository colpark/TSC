import os
import re
import glob
import json
import shutil
import seaborn as sns
import cc3d
import numpy as np
import matplotlib.pyplot as plt
from nilearn import plotting
import numpy as np
import mlxtend
import pandas as pd
from tqdm.notebook import tqdm
import nibabel as nib
import warnings

def misc_meta():
    # Template and atlas
    template = '/usr/share/data/fsl-mni152-templates/MNI152_T1_1mm.nii.gz'
    atlas = '/usr/share/fsl/data/atlases/MNI/MNI-maxprob-thr0-1mm.nii.gz'
    atlas_lr = '/home/socrates/david/tsc/TSCseg/data/atlas/atlas_lf.nii'
    idx2atlas = {'1': 'Caudate', 
                 '2': 'Cerebellum', 
                 '3': 'Frontal Lobe', 
                 '4': 'Insula',
                 '5': 'Occipital Lobe', 
                 '6': 'Parietal Lobe', 
                 '7': 'Putamen', 
                 '8': 'Temporal Lobe', 
                 '9': 'Thalamus',
                }
    idx2atlas_lr= {'1': 'Left Caudate', 
             '3': 'Left Cerebellum', 
             '5': 'Left Frontal Lobe', 
             '7': 'Left Insula',
             '9': 'Left Occipital Lobe', 
             '11': 'Left Parietal Lobe', 
             '13': 'Left Putamen', 
             '15': 'Left Temporal Lobe', 
             '17': 'Left Thalamus',
             '2': 'Right Caudate', 
             '4': 'Right Cerebellum', 
             '6': 'Right Frontal Lobe', 
             '8': 'Right Insula',
             '10': 'Right Occipital Lobe', 
             '12': 'Right Parietal Lobe', 
             '14': 'Right Putamen', 
             '16': 'Right Temporal Lobe', 
             '18': 'Right Thalamus',}
    
    test2id = {
        '100': '6529080',
        '101': '6138501',
        '102': '7168419',
        '103': '9174876',
        '104': '6628113',
        '105': '9340864',
        '106': '9599587',
        '107': '9761127',
    }
    testid2orig = {
        '6529080': '1',
        '6138501': '25',
        '7168419': '41',
        '9174876': '51',
        '6628113': '21',
        '9340864': '8',
        '9599587': '6',
        '9761127': '31',
    }
    pid2eeglobe = {
        '100': ['9'],
        '2': ['15'],
        '3': ['16', '12'],
        '106': ['11', '9'],
        '105': ['15'],
        '10': ['5', '6'],
        '12': ['15'],
        '15': ['5'],
        '104': ['6'],
        '29': ['5'],
        '107': ['12'],
        '39': ['6', '16'],
        '40': ['12', '10'],
        '102': ['6'],
        '44': ['6'],
        '45': ['5', '15'],
        '46': ['11', '9'],
        '50': ['6', '12'],
        '59': ['16'],
        '62': ['15'],
        '63': ['16'],
         }
    
    pid2leftright = {
        '100': ['9'],
        '2': ['15'],
        '3': ['16', '12'],
        '106': ['11', '9'],
        '105': ['15'],
        '12': ['15'],
        '15': ['5'],
        '104': ['6'],
        '29': ['5'],
        '107': ['12'],
        '39': ['6', '16'],
        '40': ['12', '10'],
        '102': ['6'],
        '44': ['6'],
        '45': ['5', '15'],
        '46': ['11', '9'],
        '50': ['6', '12'],
        '59': ['16'],
        '62': ['15'],
        '63': ['16'],
         }
    
    cysticpid = ['2', '15', '39', '44', '46', '63', '106', '107']
    coronalpid = ['10', '12', '15', '3', '40', '42', '44', '45', '47', '59', '62', '63', '68']
    
    pred_key_connect = {'datav5_both':'pred_inter_both', 
                        'datav5_noT1':'pred_inter_FL', 
                        'datav5_noFL':'pred_inter_T1',
                        'datav5_cystic': 'pred_inter_cysadded',
                        'datav3_both': 'pred_orig1_both',
                        'datav4_both': 'pred_orig2_both',
                        'datav3_noT1': 'pred_orig1_FL',
                        'datav4_noT1': 'pred_orig2_FL',
                        'datav6_noT1': 'pred_union_FL',
                        'datav7_noT1': 'pred_orig12_FL',   
                        'datav7_both':'pred_orig12_both', 
                        }
    
    return {'template':template, 'atlas':atlas, 'atlas_lr':atlas_lr,
            'idx2atlas':idx2atlas, 'idx2atlas_lr':idx2atlas_lr, 
            'test2id':test2id, 'testid2orig': testid2orig, 'pid2eeglobe':pid2eeglobe, 
            'pid2leftright':pid2leftright, 'cysticpid': cysticpid, 'coronalpid': coronalpid,
            'pred_key_connect':pred_key_connect}

def select(df, column, key):
    return df[df[column] == key]

def append_name(addr, adding):
    folder = '/'.join(addr.split('/')[:-1])
    name = addr.split('/')[-1].split('.')[0]
    updated_name = name + adding
    return os.path.join(folder, updated_name)

def znormalize(brain):
    brain = (brain - brain.mean()) / brain.std()
    return brain

import matplotlib.pyplot as plt

def makedir(addr):
    if not os.path.exists(addr):
        os.makedirs(addr)
    return 0

def save_scan(addr, np_array, refer, verbose=True):
    nifty = nib.Nifti1Image(np_array.astype('int32'), refer.affine, refer.header)
    nib.save(nifty, addr)
    if verbose:
        print('save done: {}'.format(addr))
    return 0
    
def display_row(imgs, user_cut, axis_idx, title):
    f = plt.figure(figsize=(15, 4))
    for i, img in enumerate(imgs):
        if user_cut is None:
            cut = int(img.shape[axis_idx]/2) 
        else:
            cut = user_cut
        f.add_subplot(1, len(imgs), i+1)
        plt.imshow(img.take([cut], axis=axis_idx).squeeze(), cmap='gray')
#         plt.colorbar()
        plt.gca().set_axis_off()
        if title is not None:
            plt.title(title[i])
    plt.show()
    return 0
   
def np_dropnan(x):
    """
    from a numpy array, drop all nan. Mainly fro calculating statistics without making all members nan
    """
    return x[np.logical_not(np.isnan(x))]

def list2str(L):
    return '/'.join(L)

def plot_many(*images, axis='z', Title=None, if_numpy=False, cut=None):
    """
    Input: 
        images: a number of numpy image addresses, optimal below 5
        if_numpy: True if numpy array. Nibabel import is omitted when True.
        Title: title for the plotting. If none, pass.
        cut: explicitly say which slice to look
        axis: which axis to see from ('x', 'y', 'z')
    """
    axis_dict = {'x':0, 'y':1, 'z':2}
    imgs = []
    if if_numpy==True:
        imgs = images
    else:
        for img in images:
            nib_img = nib.load(img).get_fdata()
            print(img.split('/')[-1],nib_img.shape)
            imgs.append(nib_img)
        
    if axis=='all':
        for axis in axis_dict.keys():
            display_row(imgs, cut, axis_dict[axis], Title)      
    elif axis in axis_dict.keys():
        display_row(imgs, cut, axis_dict[axis], Title)

def plot_seg(img, mask):
    print('image: {}'.format(img))
    f = plt.figure(figsize=(15, 5))
    plotting.plot_img(img, figure=f, display_mode='z', cmap='gray', cut_coords=(16, 30, 45, 60, 70))
    plt.show()
    print('mask: {}'.format(mask))
    f = plt.figure(figsize=(15, 5))
    plotting.plot_roi(mask, img, figure=f, alpha=0.6, display_mode='z', cmap='jet',  cut_coords=(16, 30, 45, 60, 70))
    plt.show()

"""Compute volume of non-zero voxels in a nifti image."""

def compute_volumn(orig_nifty, brain_region=[]):
    """
    Given the nifty file and the region of interest (binary), return number of voxels and the volume
        orig_nifty: nifty addr
        brain_region: numpy array
    """
    INPUT = orig_nifty
    nii = nib.load(INPUT)
    if len(brain_region) == 0:
        img = nii.get_fdata()
    else:
        img = brain_region
    voxel_dims = (nii.header["pixdim"])[1:4]

    # Compute volume
    n_voxel = np.count_nonzero(img)
    voxel_volume = np.prod(voxel_dims)
    volume  = n_voxel * voxel_volume

    return n_voxel, volume



def compute_voxel_volumn(orig_nifty):
    """
    Given the nifty file and the region of interest (binary), return the number of voxels and the volume
        orig_nifty: nifty addr
        brain_region: numpy array
    """
    nii = nib.load(orig_nifty)
    voxel_dims = (nii.header["pixdim"])[1:4]
    voxel_volume = np.prod(voxel_dims)
    return voxel_volume

def get_dice(addr1, addr2, k=1):
    img1 = nonzero_to_one(eye_to_zero(nib.load(addr1).get_fdata())).astype('int')
    img2 = nonzero_to_one(eye_to_zero(nib.load(addr2).get_fdata())).astype('int')
    dice = np.sum(np.logical_and(img1==k, img2==k))*2.0 / (np.sum(img1[img1==k]==k) + np.sum(img2[img2==k]==k))
#     print ('Dice similarity score is {}'.format(dice))
    return dice
    
def eye_to_zero(img):
    img[img==7.0] = 0.0
    return img    
    
def nonzero_to_one(img):
    img[img!=0.0] = 1.0
    return img

def getNumbers(str):
    array = re.findall(r'[0-9]+', str)
    return int(array[0])

def id_search(mask_dir, s):
    found = False
    for addr_ in glob.glob(mask_dir + '/*'):
        if getNumbers(addr_.split('/')[-1]) == s:
            # make sure it's binary
            addr = addr_
            found = True
            break
    if not found:
        print('subject not found: {}'.format(s))
    return addr

def get_file_dict(result_dir, subject_id):
    sub_file_dict = {}
    for s in subject_id:
        addr = id_search(result_dir, s)
        sub_file_dict[str(s)] = addr
    return sub_file_dict


def get_tuber_dice(addr1, addr2, background=0, threshold=1):
    """
    given two mask addresses, perform CCA, and get the tuber_dice
    addr1 = predicted
    addr2 = true mask
    threshold = volume threshold
    """
    voxel_volume = compute_voxel_volumn(addr1)
    one_np = nonzero_to_one(eye_to_zero(nib.load(addr1).get_fdata())).astype('int')
    two_np = nonzero_to_one(eye_to_zero(nib.load(addr2).get_fdata())).astype('int')
    cca_one = cc3d.connected_components(one_np) 
    cca_two = cc3d.connected_components(two_np) 
    tub_indice_pred = np.unique(cca_one)[background+1:]
    tub_indice_gt = np.unique(cca_two)[background+1:]
    sensitivity_ = 0
    specificity_ = 0
    agreed_ = 0
    all_pred = len(tub_indice_pred)
    all_gt = len(tub_indice_gt)
    for t1 in tub_indice_pred:
        for t2 in tub_indice_gt:
            if voxel_volume * np.logical_and(cca_one==t1, cca_two==t2).sum()>=threshold:
                specificity_ += 1
                agreed_+=1
                break
    for t2 in tub_indice_gt:
        for t1 in tub_indice_pred:
            if voxel_volume * np.logical_and(cca_one==t1, cca_two==t2).sum()>=threshold:
                sensitivity_ += 1
                agreed_+=1
                break                
    return agreed_/(all_gt + all_pred), sensitivity_ /all_gt , specificity_/all_pred 

def get_which_tuber_covered(pred, gt, background=0, threshold = 1):
    """
    Get two addresses and return cca outputs and covered gt tuber indice  
    """
    voxel_volume = compute_voxel_volumn(pred)
    one_np = nonzero_to_one(eye_to_zero(nib.load(pred).get_fdata())).astype('int')
    two_np = nonzero_to_one(eye_to_zero(nib.load(gt).get_fdata())).astype('int')
    cca_pred = cc3d.connected_components(one_np) 
    cca_gt = cc3d.connected_components(two_np) 
    tub_indice_pred = np.unique(cca_pred)[background+1:]
    tub_indice_gt = np.unique(cca_gt)[background+1:]
    covered_gt_tub_indice = []

    for t2 in tub_indice_gt:
        for t1 in tub_indice_pred:
            if voxel_volume * np.logical_and(cca_pred==t1, cca_gt==t2).sum()>=threshold:
                covered_gt_tub_indice.append(t2)
                break                
    return cca_pred, cca_gt, covered_gt_tub_indice

def get_area_per_lobe(reg2tp_mask, if_lr = True, if_numpy = False):
    """
    Get cumic milimeter (mm3) volume of mask corresponding to each MNI structural atlas
    """
    if if_lr:
        # left and right distinction
        atlas = '/home/socrates/david/tsc/TSCseg/data/atlas/atlas_lf.nii'
        idxs = range(1,18+1)
    else:
        # just 9 lobes
        atlas = '/usr/share/fsl/data/atlases/MNI/MNI-maxprob-thr0-1mm.nii.gz'
        idxs = range(1,9+1)
        
    lobe_result = {}
    if not if_numpy:
        mask_FL_reg2tp = nib.load(reg2tp_mask).get_fdata() 
    else:
        mask_FL_reg2tp = reg2tp_mask
    for lobe_id in idxs:        
        atlas_lobe = nib.load(atlas).get_fdata()==lobe_id
        lobe_result[str(lobe_id)] = np.logical_and(mask_FL_reg2tp, atlas_lobe).sum()
    return lobe_result

def get_dominating_lobe(assignment, threshold=0.5):
    """
    Get the dominating lobe, given the assignment dictionary for a single tuber.
    Over 50% of tuber area belong to the dominating lobe
    return corresponding list of tubers  
    """
    total_volume = np.array(list(assignment.values())).sum()
    lobe_list = []
    found = False
    
    for lobe in assignment.keys():
        if assignment[lobe] / total_volume > threshold:
            found = True
            lobe_list.append(lobe)
    assert len(lobe_list) <= 1
    
    if not found:
        for lobe in assignment.keys():
            if assignment[lobe] > 100:
                lobe_list.append(lobe)        
    
    if len(lobe_list) == 0:
        lobe_list.append('0')        
    assert len(lobe_list) > 0
    return lobe_list
        

