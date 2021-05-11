import os
import re
import glob
import json
import cc3d
import shutil
import mlxtend
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import nibabel as nib
from nilearn import plotting
import statsmodels.api as sm
from typing import Dict, List
from scipy import stats
from nilearn import plotting
from tqdm.notebook import tqdm
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_linear_regression
import xml.etree.ElementTree as ET

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
    major_lobes = {'5', '6', '9', '10', '11', '12', '15', '16', }
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
    
    orig2test = {
        '1':'100',
        '25':'101',
        '41':'102',
        '51':'103',
        '21':'104',
        '8':'105',
        '6':'106',
        '31':'107',
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
            'test2id':test2id, 'testid2orig': testid2orig, 'orig2test':orig2test, 'pid2eeglobe':pid2eeglobe, 
            'pid2leftright':pid2leftright, 'cysticpid': cysticpid, 'coronalpid': coronalpid,
            'pred_key_connect':pred_key_connect, 'major_lobes':major_lobes}

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
    atlas_np = nib.load(atlas).get_fdata()
    for lobe_id in idxs:        
        atlas_lobe = atlas_np==lobe_id
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
        

class Tuber_EXP:
    """
    Class to integrate all experiments, cache intermediate outputs, and save results
    """
    def __init__(self, INTEG_ADDR: Dict[str, str], meta_info: Dict[str, str]):
        self.INTEG = INTEG_ADDR
        self.meta = meta_info
        self.all_subs = list(self.INTEG['train'].keys()) 
        self.all_subs += list(self.INTEG['test'].keys())
        # compute volume per lobe from the template
        self.tuber_stats = {}
           
    
    def get_variables(self, 
                      lobes: List[str], 
                      mod: str = 'mask_FL_reg2tp_thr0.5',
                      var: str = 'area',
                      znorm: bool = False,
                      granule: str = 'mni',
                      vnorm: bool = True,
                      cystic: bool = False, ) -> Dict[str, int]:
        """
        Given meta informations: INTEG_ADDR / meta_info, retrieve corresponding variables per subject
        """        
        assert var in ['area', 'ntuber', 'intensity']
        self.lobe_vol, self.hemis_vol = self.get_lobe_vol(lobes)
        
        # version control.
        version = '{}|{}|vnorm{}|znorm{}|{}'.format(var, granule, vnorm, znorm, mod)
        if not version in self.tuber_stats.keys():
            self.tuber_stats[version] = {}
            if var == 'intensity':
                self.tuber_stats[version]= {'mean': {}, 'std': {}, '75p': {}, '90p': {}}
        else:
            overwrite = input('experiment name duplicated. overwrite? [y/n]: ')
            overwrite = True if overwrite == 'y' else False
            if overwrite:
                self.tuber_stats[version] = {}
                if var == 'intensity':
                    self.tuber_stats[version]= {'mean': {}, 'std': {}, '75p': {}, '90p': {}}
            else:
                raise NameError('Duplicated version name')   
        
        # iterate through subjects
        if cystic:
            self.all_subs = self.meta['cysticpid']
            
        for sid in tqdm(self.all_subs):
            dataset = self.train_or_test(sid)
            sub_addr = self.INTEG[dataset][sid][mod]
            fl_addr = self.INTEG[dataset][sid]['FL_reg2tp']
                                               
            # conditioning on the variables of focus
            if var == 'area':
                area_per_lobe = self.get_area(sub_addr, granule, lobes)
                if vnorm:
                    area_per_lobe = self.normalize_vol(area_per_lobe, granule)                    
                self.tuber_stats[version][sid] = area_per_lobe
            elif var == 'ntuber':
                ntuber_per_lobe = self.get_ntubers(sub_addr, granule, lobes)
                if vnorm:
                    ntuber_per_lobe = self.normalize_vol(ntuber_per_lobe, granule)                    
                self.tuber_stats[version][sid] = ntuber_per_lobe
            elif var == 'intensity':                
                intensity_stats  = self.get_intensity_stats(fl_addr, sub_addr, granule, lobes)
                if vnorm:
                    for stat_ in intensity_stats.keys():
                        intensity_stats[stat_]  = self.normalize_vol(intensity_stats[stat_], granule)
                
                for stat_ in intensity_stats.keys():
                    self.tuber_stats[version][stat_][sid] = intensity_stats[stat_]      
                
        if znorm and var in ['area', 'ntuber']:
            self.tuber_stats[version] = self.normalize_z(self.tuber_stats[version])
        elif znorm and var in ['intensity']:
            for stat_ in intensity_stats.keys():
                self.tuber_stats[version][stat_] = self.normalize_z(self.tuber_stats[version][stat_])
        return 0
    
    def normalize_vol(self, per_lobe: Dict[str, int], granule: str):
        normalized = {}
        norm_dict = self.hemis_vol if granule == 'lr' else self.lobe_vol
        for lo in per_lobe:
            normalized[lo] = per_lobe[lo] / norm_dict[lo]
        return normalized
        
    def normalize_z(self, tuber_stats: Dict[str, str]):
        tuber_stats_normed = {sid: {} for sid in self.all_subs}
        lobes = tuber_stats[self.all_subs[0]].keys()
        area_mean_std = {lo:{'mean':{}, 'std': {}} for lo in lobes}
        for lo in lobes:
            lobe_np = np.array([tuber_stats[sid][lo] for sid in self.all_subs])
            lobe_np = lobe_np[np.logical_not(np.isnan(lobe_np))]   
            mean, std = lobe_np.mean(), lobe_np.std()
            area_mean_std[lo]['mean'] = mean
            area_mean_std[lo]['std'] = std
        for sid in self.all_subs:
            for lo in lobes:
                tuber_stats_normed[sid][lo] = (tuber_stats[sid][lo] - area_mean_std[lo]['mean']) / area_mean_std[lo]['std']
        return tuber_stats_normed
    
    def train_or_test(self, sid):
        return 'train' if int(sid) < 100 else 'test'
    
    def get_lobe_vol(self, lobes: List[str]) -> Dict[str, int]:
        atlas_lr_addr = self.meta['atlas_lr']
        atlas_lr = nib.load(atlas_lr_addr).get_fdata()
        lobe_vol = {}
        for lo in lobes:        
            lobe_sum = (atlas_lr==int(lo)).sum()
            lobe_vol[lo] = lobe_sum
        hemis_vol = {'left':0, 'right':0}
        for lo in lobe_vol.keys():
            hemis = 'left' if int(lo)%2 == 1 else 'right'                
            hemis_vol[hemis] += lobe_vol[lo]
        return lobe_vol, hemis_vol
    
    def get_area(self, sub_addr: str, granule: str, lobes: List[str]) -> Dict[str, int]:        
        area_per_all = get_area_per_lobe(sub_addr, if_lr=True, if_numpy=False)
        area_per_focus = {lo:area_per_all[lo] for lo in lobes}
        
        if granule == 'lr': 
            area_per_hemis = {'left':0, 'right':0}
            for lo in area_per_focus.keys():
                hemis = 'left' if int(lo)%2 == 1 else 'right'                
                area_per_hemis[hemis] += area_per_focus[lo]
            area_per_focus = area_per_hemis
        area_per_focus = {k:int(area_per_focus[k]) for k in area_per_focus.keys()}
        return area_per_focus
    
    def get_ntubers(self, sub_addr: str, granule: str, lobes: List[str]) -> Dict[str, int]:        
        mask_np = nib.load(sub_addr).get_fdata().astype('int')
        mask_cc3d = cc3d.connected_components(mask_np)
        ntuber_stats = {lo:0 for lo in lobes}
        for label in np.unique(mask_cc3d)[1:]:
            booled = mask_cc3d==label
            assignment = get_area_per_lobe(booled, if_lr=True, if_numpy=True)
            assignment_d = get_dominating_lobe(assignment)        
            for L in assignment_d:
                if L in lobes:
                    ntuber_stats[L] += 1
                    
        if granule == 'lr': 
            area_per_hemis = {'left':0, 'right':0}
            for lo in ntuber_stats.keys():
                hemis = 'left' if int(lo)%2 == 1 else 'right'                
                area_per_hemis[hemis] += ntuber_stats[lo]
            ntuber_stats = area_per_hemis
        ntuber_stats = {k:int(ntuber_stats[k]) for k in ntuber_stats.keys()}
        return ntuber_stats
    
    def get_intensity_stats(self, fl_addr: str, sub_addr: str, granule: str, lobes: List[str]) -> Dict[str, int]:        
        """
        Get lobe-level mean, variance, 75-, 90-percentile intensities of all corresponding tubers 
        """
        flair_np =znormalize(nib.load(fl_addr).get_fdata())       
        mask_np = nib.load(sub_addr).get_fdata().astype('int')
        mask_cc3d = cc3d.connected_components(mask_np)
        if granule == 'lr': 
            t_dict = {'left':[], 'right':[]}
            intensity_stats = {lo:{'left':0, 'right':0} for lo in ['mean', 'std', '75p', '90p']}          
        else:
            t_dict = {lo:[] for lo in lobes}
            intensity_stats = {lo:{L:0 for L in lobes} for lo in ['mean', 'std', '75p', '90p']}   
            
        for label in np.unique(mask_cc3d)[1:]:
            booled = mask_cc3d==label
            masked_flair = flair_np * booled
            tuber_intensities = masked_flair[masked_flair!=0]            
            assignment = get_area_per_lobe(booled, if_lr=True, if_numpy=True)
            assignment_d = get_dominating_lobe(assignment)        
            for L in assignment_d:
                if L in lobes:                    
                    if granule == 'lr':
                        hemis = 'left' if int(L)%2 == 1 else 'right'                
                        t_dict[hemis] += list(tuber_intensities)
                    else:
                        t_dict[L] += list(tuber_intensities)
        
        for lo in intensity_stats['mean'].keys():
            np_ = np.array(t_dict[lo])
            if len(np_) > 1:      
                intensity_stats['mean'][lo] = np_.mean() 
                intensity_stats['std'][lo] = np_.std() 
                intensity_stats['75p'][lo] = np.percentile(np_, 75) 
                intensity_stats['90p'][lo] = np.percentile(np_, 90)    
            else:   
                intensity_stats['mean'][lo] = np.nan
                intensity_stats['std'][lo] = np.nan
                intensity_stats['75p'][lo] = np.nan
                intensity_stats['90p'][lo] = np.nan    
                
        return intensity_stats
    
    def save_stats(self, addr='./data/stats', version = 'v1'):
        save_addr = addr+'_{}.json'.format(version)
        with open(save_addr, 'w') as f:
            json.dump(self.tuber_stats, f)
        print('saved to {}'.format(save_addr))
            
    def load_stats(self, addr='./data/stats', version = 'v1'):
        load_addr = addr+'_{}.json'.format(version)
        with open(load_addr, 'r') as f:
            self.tuber_stats = json.load(f)
        print('loaded from {}'.format(load_addr))
    
    def get_ofmax_classify_accuracy(self, lobes, var_):
        correct_ = []
        wrong_ = []
        if var_.split('|')[0] in ['ntuber', 'area']:
            var_stat = self.tuber_stats[var_]
        else:
            var_stat = self.tuber_stats[var_]['std']
        for s in var_stat.keys():
            if s in self.meta['pid2eeglobe'].keys():
                eight = np.zeros(len(lobes))
                for i, lo in enumerate(lobes):
                    eight[i] = var_stat[s][lo]
                eight_max = eight.max()
                eight_argmax = eight.argmax()
                if lobes[eight_argmax] in EXP.meta['pid2eeglobe'][s]:
                    correct_.append(s)
                else:
                    wrong_.append(s)
        accuracy = len(correct_) / (len(correct_) + len(wrong_))
        return accuracy, correct_, wrong_
    
    def znormalize_list(self, in_list: List[int]) -> List[int]:
        """
        Given a list, remove nan values and z-normalize the whole list.
        """
        in_list_np = np.array(in_list)                          
        np_mean, np_std = dropnan(in_list_np).mean(), dropnan(in_list_np).std()
        return list( (in_list_np - np_mean) / np_std)

    def get_logistic_regress(self, lobes, var_, 
                             control_lobe = False, 
                             summary = False, 
                             merge_lr = False, 
                             show_img = False, 
                             save_img=False,
                             save_dir = '/home/socrates/david/tsc/codebase/results'):
        set_vars = {}      
        
        dep_ = []
        indep_ = []
        dummy_lobe = []
        if var_.split('|')[0] in ['ntuber', 'area']:
            var_stat = self.tuber_stats[var_]
        else:
            var_stat = self.tuber_stats[var_]['std']
        for s in var_stat.keys():
            if s in self.meta['pid2eeglobe'].keys():
                eight = np.zeros(len(lobes))
                for i, lo in enumerate(lobes):
                    eight[i] = var_stat[s][lo]
                indep_ += list(eight)
                # dummy to control for lobe location
                dummy_lobe += lobes
                dep_temp = [1 if lo in self.meta['pid2eeglobe'][s] else 0 for lo in lobes]
                dep_ += list(dep_temp)
    
        set_vars[var_] = indep_
        if control_lobe:
            if not merge_lr:
                set_vars['dummy_lobe'] = np.asarray(dummy_lobe).astype('int')
            else:
                merge_target = ['6', '10', '12', '16']
                dummy_lobe = [str(int(lo) -1) if lo in merge_target else lo for lo in dummy_lobe ]
                set_vars['dummy_lobe'] = np.asarray(dummy_lobe).astype('int')
            
        # make dataframe for statsmodel
        set_vars['dep'] = dep_
        df = pd.DataFrame(data=set_vars).dropna()
        
        if control_lobe:
            X, y= df[[var_, 'dummy_lobe']], df['dep']
        else:
            X, y= df[var_], df['dep']
        X = sm.add_constant(X)
        est = sm.OLS(y, X).fit()
        if summary:
            print(est.summary())
        effect_size, pvalue = est.params[var_], est.pvalues[var_]
        
        if show_img:
            label_key = var_.split('|')[0]
            labeling_dict = {'area': 'tuber size', 'intensity':'intensity std', 'ntuber': 'number of tubers' }
            fig_title = 'Logistic regression of lobe attribution, pvalue= {:.4f}'.format(pvalue)
            
            if control_lobe:                
                ax = sns.lmplot(x=var_, y="dep", hue="dummy_lobe", data=df, height=7, aspect=10/7, logistic=True)
                new_title = 'lobar attribution'                
                ax._legend.set_title(new_title)
                new_legend = ['frontal lobe', 'occipital lobe', 'parietal lobe', 'temporal lobe']
                for i, t in enumerate(ax._legend.texts): 
                    t.set_text(new_legend[i])
            
            else:
                ax = sns.lmplot(x=var_, y="dep", data=df, height=7, aspect=10/7, logistic=True)
#             ax.set_title(fig_title)
            ax.set(xlabel=labeling_dict[label_key], ylabel='epileptogenic', title=fig_title)
            if save_img:
                makedir(os.path.join(save_dir, 'figures', 'group_stat_test'))
                save_addr =  os.path.join(save_dir, 'figures', 'group_stat_test', 'stattest_{}_controllobe{}_mergelr{}.png'.format(var_,
                                                                                                                   control_lobe,
                                                                                                                   merge_lr,
                                                                                                                   ))
                
                ax.savefig(save_addr)
            
            plt.show()
        
        return effect_size, pvalue
    
    def get_rank_correlation(self, lobes, var_):
        dep_ = []
        indep_ = []        
        if var_.split('|')[0] in ['ntuber', 'area']:
            var_stat = self.tuber_stats[var_]
        else:
            var_stat = self.tuber_stats[var_]['std']
        for s in var_stat.keys():
            if s in self.meta['pid2eeglobe'].keys():
                dep_eight = np.array([1 if lo in self.meta['pid2eeglobe'][s] else 0 for lo in lobes])
                
                eight = np.zeros(len(lobes))
                for i, lo in enumerate(lobes):
                    eight[i] = var_stat[s][lo] 

                if np.isnan(eight).any():  
                    indep_eight = eight[np.logical_not(np.isnan(eight))]
                    dep_eight = dep_eight[np.logical_not(np.isnan(eight))]
                    assert len(indep_eight) == len(dep_eight)
                else:
                    indep_eight =eight

                rank_eight = indep_eight.argsort() + 1                   
                
                indep_ += list(rank_eight)                
                dep_ += list(dep_eight)
                
        corr, pvalue = spearmanr(dep_, indep_)
        return corr, pvalue