import os
import re
import glob
import json
import shutil
import warnings
import subprocess
import numpy as np
import pandas as pd
import seaborn as sns
import nibabel as nib
import matplotlib.pyplot as plt
from nipype.interfaces import fsl
from nipype import SelectFiles, Node
from nipype.interfaces.utility import Function
from nipype.interfaces.ants import Registration
from nipype.interfaces.ants import ApplyTransforms
from nipype import DataGrabber, Workflow, Node, MapNode, Function, DataSink, IdentityInterface

from misc_utils import *
warnings.filterwarnings("ignore")

class Preproc_TSC():
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.raw_dir = os.path.join(base_dir, 'data/inference/raw')
        self.proc_dir = os.path.join(base_dir, 'data/inference/processed')
        self.infer_dir = os.path.join(base_dir, 'results/samples/infer')
        self.mni_dir = os.path.join(base_dir, 'results/registration')
        self.template = os.path.join(self.base_dir, 'data/template/MNI152_T1_1mm.nii.gz')
        self.meta = misc_meta()
        self.sub_dict = {}
    
    def base_preproc(self, sub_id):
        self.sub_dict[sub_id] = {}
        
        # Registration
        print('perform registration')
        t1_addr, flair_addr, reg_file, reg_mat = self.reg_t1_to_flair(sub_id)
        self.sub_dict[sub_id]['T1_orig'] = t1_addr
        self.sub_dict[sub_id]['FL_orig'] = flair_addr
        self.sub_dict[sub_id]['T1_reg2FL'] = reg_file
        self.sub_dict[sub_id]['mat_T12FL'] = reg_mat
        print('Generated and saved: T1_orig, FL_orig, T1_reg2FL, mat_T12FL')
        
        # Brain extraction
        print('perform brain extraction')
        brain_file, brain_mask = self.flair_brain_extract(sub_id)
        self.sub_dict[sub_id]['FL_brain'] = brain_file
        self.sub_dict[sub_id]['mask_FL_brain'] = brain_mask
        self.sub_dict[sub_id]['T1_reg2FL_brain'] = self.apply_brainmask_to_t1(sub_id)
        print('Generated and saved: FL_brain, mask_FL_brain, T1_reg2FL_brain')
        return 0
    
    def get_pred(self, sub_id, pred_version= 'datav5_both', vote_n = 10):
        # Get prediction
        newkey = self.meta['pred_key_connect'][pred_version]
        self.perform_prediction(sub_id, pred_version)
        predictions = self.retr_prediction(sub_id, pred_version)
        pred_addr = self.get_best_prediction(sub_id, predictions, vote_n = vote_n)        
        self.sub_dict[sub_id][newkey] = pred_addr
        return 0
    
    def mni_link_flair(self, sub_id):
        """
        Registration of T1 / FLAIR to MNI     
        """
        
        work_dir = os.path.join(self.mni_dir, sub_id)
        output_dir = os.path.join(self.mni_dir, sub_id, 'output')
        makedir(work_dir)
        makedir(output_dir)
        n_procs = 2
        version = sub_id+'_wf'
        
        infosource = Node(IdentityInterface(fields=['sid', 'data']), name='SubInfo')
        infosource.iterables = [('sid', [sub_id])]
        infosource.inputs.data = self.sub_dict
        
        select_node = Node(name='pickup', interface= Function(input_names=["sid", 'data'],  
                           output_names=['T1_orig', 'FL_orig', 'FL_brain', 'mat_T12FL'],
                           function=_pickup_data_basic))

        # Registration
        reg_node = self._get_reg_node()

        # convert the mat to inverse
        invt_node = Node(name='convert', interface=fsl.ConvertXFM())
        invt_node.inputs.invert_xfm = True

        # datasink
        ds_node = Node(DataSink(base_directory= os.path.join(output_dir, 'experiment', version)), name="DataSink")
        ds_node.inputs.substitutions = [('..TSC_', ''), ('_file_name_TSC_', '')]

        # flair to t1
        f2t_fl_node = Node(name='ft2_flair', interface=fsl.preprocess.ApplyXFM())
        f2t_fl_node.inputs.apply_xfm = True
        # flair_brain to t1
        f2t_fb_node = Node(name='ft2_flair_brain', interface=fsl.preprocess.ApplyXFM())
        f2t_fb_node.inputs.apply_xfm = True

        # apply ants registration
        t2temp_fl_node = Node(name='t2temp_flair', interface = ApplyTransforms())
        t2temp_fl_node.inputs.reference_image = self.template
        t2temp_fb_node = Node(name='t2temp_flair_brain', interface = ApplyTransforms())
        t2temp_fb_node.inputs.reference_image = self.template

        wf = Workflow(name=version, base_dir=work_dir)
        wf.connect([
            (infosource, select_node, [('sid', 'sid'), ('data', 'data')]),
            (select_node, reg_node, [('T1_orig', 'moving_image')]),
            (select_node, f2t_fl_node, [('T1_orig', 'reference'), ('FL_orig', 'in_file')]),
            (select_node, f2t_fb_node, [('T1_orig', 'reference'), ('FL_brain', 'in_file')]),

            (reg_node, t2temp_fl_node, [('composite_transform', 'transforms')]),
            (reg_node, t2temp_fb_node, [('composite_transform', 'transforms')]),

            (select_node, invt_node, [('mat_T12FL', 'in_file')]),
            (invt_node, f2t_fl_node, [('out_file', 'in_matrix_file')]),
            (invt_node, f2t_fb_node, [('out_file', 'in_matrix_file')]),

            (f2t_fl_node, t2temp_fl_node, [('out_file', 'input_image')]),
            (f2t_fb_node, t2temp_fb_node, [('out_file', 'input_image')]),

            (t2temp_fl_node, ds_node, [('output_image', 'results@flair_warped')]),
            (t2temp_fb_node, ds_node, [('output_image', 'results@flair_brain_warped')]), 
            (reg_node, ds_node, [('composite_transform', 'results@mat_T12mni')]),
            (reg_node, ds_node, [('warped_image', 'results@T1_warped')])
                        ])

        wf.run('MultiProc', plugin_args={'n_procs': n_procs})
        print('Registration successful.')
        self._get_reg_results(sub_id, version)
        return 0
        
    
    
    def _get_reg_node(self):
            reg_node = Node(interface = Registration(), name='anstregistration')
            reg_node.inputs.fixed_image = self.template
            reg_node.inputs.transforms = ['Affine', 'SyN']
            reg_node.inputs.transform_parameters = [(2.0,), (0.25, 3.0, 0.0)]
            reg_node.inputs.number_of_iterations = [[1500, 200], [100, 50, 30]]
            reg_node.inputs.dimension = 3
            reg_node.inputs.write_composite_transform = True
            reg_node.inputs.collapse_output_transforms = False
            reg_node.inputs.initialize_transforms_per_stage = False
            reg_node.inputs.metric = ['Mattes']*2
            reg_node.inputs.metric_weight = [1]*2 # Default (value ignored currently by ANTs)
            reg_node.inputs.radius_or_number_of_bins = [32]*2
            reg_node.inputs.sampling_strategy = ['Random', None]
            reg_node.inputs.sampling_percentage = [0.05, None]
            reg_node.inputs.convergence_threshold = [1.e-8, 1.e-9]
            reg_node.inputs.convergence_window_size = [20]*2
            reg_node.inputs.smoothing_sigmas = [[1,0], [2,1,0]]
            reg_node.inputs.sigma_units = ['vox'] * 2
            reg_node.inputs.shrink_factors = [[2,1], [3,2,1]]
            reg_node.inputs.use_estimate_learning_rate_once = [True, True]
            reg_node.inputs.use_histogram_matching = [True, True] # This is the default
            reg_node.inputs.output_warped_image = 'T1_warped_image.nii.gz'
            return reg_node
    
    def mni_register_pred(self, sub_id, pred_version= 'datav5_both', vote_n = 10):
        """
        Registration of all other preds to MNI    
        """
        newkey = self.meta['pred_key_connect'][pred_version]        
        
        
        work_dir = os.path.join(self.mni_dir, sub_id)
        output_dir = os.path.join(self.mni_dir, sub_id, 'output')
        n_procs = 2
        version = sub_id+'_pred_wf'
        
        infosource = Node(IdentityInterface(fields=['sid', 'data', 'predkey']), name='SubInfo')
        infosource.iterables = [('sid', [sub_id])]
        infosource.inputs.data = self.sub_dict
        infosource.inputs.predkey = newkey
        
        select_node = Node(name='pickup', interface= Function(input_names=["sid", 'data', 'predkey'],  
                           output_names=[newkey, 'T1_orig', 'mat_T12FL', 'mat_T12mni'],
                           function=_pickup_data_pred))


        # convert the mat to inverse
        invt_node = Node(name='convert', interface=fsl.ConvertXFM())
        invt_node.inputs.invert_xfm = True

        # datasink
        ds_node = Node(DataSink(base_directory= os.path.join(output_dir, 'experiment', version)), name="DataSink")
        ds_node.inputs.substitutions = [('..TSC_', ''), ('_file_name_TSC_', '')]

        # flair_brain to t1
        pred2t1_node = Node(name='pred2t1', interface=fsl.preprocess.ApplyXFM())
        pred2t1_node.inputs.apply_xfm = True

        # apply ants registration
        predt12mni_node = Node(name='predt12mni', interface = ApplyTransforms())
        predt12mni_node.inputs.reference_image = self.template

        wf = Workflow(name=version, base_dir=work_dir)
        wf.connect([
            (infosource, select_node, [('sid', 'sid'), ('data', 'data'), ('predkey', 'predkey')]),
            (select_node, pred2t1_node, [('T1_orig', 'reference'), (newkey, 'in_file')]),
            (select_node, predt12mni_node, [('mat_T12mni', 'transforms')]),
            (select_node, invt_node, [('mat_T12FL', 'in_file')]),            
            (invt_node, pred2t1_node, [('out_file', 'in_matrix_file')]),

            (pred2t1_node, predt12mni_node, [('out_file', 'input_image')]),
            (predt12mni_node, ds_node, [('output_image', 'results@{}'.format(newkey))]), 
                        ])

        wf.run('MultiProc', plugin_args={'n_procs': n_procs})
        print('Prep registration successful.')
        updated_key = self._get_predreg_results(sub_id, version, newkey)
        self._threshold_mask_save(sub_id, updated_key)
        return 0
    
    def _get_reg_results(self, sub_id, version):
        output_base = os.path.join(self.mni_dir, sub_id, 'output', 'experiment', version)
        result_link = {'results@flair_warped':'FL_reg2tp',
                      'results@flair_brain_warped': 'FL_brain_reg2tp',
                      'results@mat_T12mni': 'mat_T12mni',
                      'results@T1_warped': 'FL_reg2tp'}
        for k in result_link.keys():
            self.sub_dict[sub_id][result_link[k]] = get_unique_file(os.path.join(output_base, '{}'.format(k), '_sid_{}'.format(sub_id)))
        print('Generated and saved: FL_reg2tp, FL_brain_reg2tp, mat_T12mni, FL_reg2tp')        
        return 0
    
    def _get_predreg_results(self, sub_id, version, newkey):
        output_base = os.path.join(self.mni_dir, sub_id, 'output', 'experiment', version)
        result_link = {'results@{}'.format(newkey):newkey+'_reg2tp'}
        for k in result_link.keys():
            self.sub_dict[sub_id][result_link[k]] = get_unique_file(os.path.join(output_base, '{}'.format(k), '_sid_{}'.format(sub_id)))
        print('Generated and saved: {}'.format(result_link[k]))        
        return result_link[k]
    
    def _threshold_mask_save(self, sub_id, updated_key):
        ## Resaving the pred_reg2tp by thresholding
        threshold = 0.5
        pred_addr = self.sub_dict[sub_id][updated_key]
        pred = nib.load(pred_addr)
        th_pred = (pred.get_fdata() > threshold).astype('int')
        save_addr= append_name(pred_addr, adding='_threshold05.nii')
        save_dir = os.path.join(self.proc_dir, sub_id, save_addr.split('/')[-1])
        save_scan(save_dir, th_pred, pred)
        self.sub_dict[sub_id]['{}_thr0.5'.format(updated_key)] = save_dir
        print('Generated and saved: {}'.format('{}_thr0.5'.format(updated_key)))    
        return 0
    
    def perform_prediction(self, sub_id, model_version = 'datav5_both'):
        """
        evaluate the prediction command
        """
        command = "python ./neuralnet/main.py --infer True --version datav5_both --mask_version v5 --n_cuda 0 --modality both --batch_size 12"
        print('evaluating.. {}'.format(command))
        command_list = command.split(' ')
        result = subprocess.run(command_list, stdout=subprocess.PIPE)
        result_lines = result.stdout.splitlines()
        for line in result_lines:
            print(line.decode())        
        return 0
    
    def retr_prediction(self, sub_id, model_version):
        """
        Given the model version and the subject id, retrieve already inferred prediction addresses
        """
        version_dir = os.path.join(self.infer_dir, sub_id, model_version)
        folders = glob.glob(version_dir + '/*')        
        predictions = [get_unique_file(f) for f in folders]        
        return predictions
    
    def get_best_prediction(self, sub_id, predictions, vote_n = 10):
        """
        Inputs:
            predictions: a list of predicted addresses containing 21 predictions
            method: choices = ['vote', 'median']
        return:
            an address of the best prediction. for voting, a new address will be created
        """
        scout = nib.load(predictions[0])
        summary = np.zeros((len(predictions), *scout.get_data().shape))
        summary[0] = scout.get_data()
        for i in range(1, summary.shape[0]):
            summary[i] = nib.load(predictions[i]).get_data()
        save_dir = os.path.join(self.base_dir, 'results/samples/vote', sub_id)
        makedir(save_dir)
        save_addr = os.path.join(save_dir, 'vote_{}.nii'.format(vote_n))          
        
        majority_voting = ( summary.mean(axis=0) >= vote_n / summary.shape[0] )
        save_scan(save_addr, majority_voting, scout)
        
        return save_addr
         
        
    
    
    def save_dict(self, filename='infer_integ.json'):
        """
        save self.sub_dict to the processed dir
        """
        with open(os.path.join(self.proc_dir, filename), 'w') as f:
            json.dump(self.sub_dict, f)
        print('Dict saving done: {}'.format(os.path.join(self.proc_dir, filename)))
        return 0
                  
    def load_dict(self, filename='infer_integ.json'):
        with open(os.path.join(self.proc_dir, filename), 'r') as f:
            self.sub_dict = json.load(f)
        print('Dict loaded from: {}'.format(os.path.join(self.proc_dir, filename)))
        return 0
    
    def reg_t1_to_flair_dynamic(self, t1_addr, flair_addr, reg_addr, regmat_addr):  
        flt = fsl.FLIRT(bins=640, cost_func='mutualinfo')
        flt.inputs.in_file = t1_addr
        flt.inputs.reference = flair_addr
        flt.inputs.out_file = reg_addr
        flt.inputs.out_matrix_file = regmat_addr
        res = flt.run() 
        print('flirt done for {}'.format(t1_addr))
        return 0
    
    def reg_t1_to_flair(self, sub_id):
        sub_dir = os.path.join(self.raw_dir, sub_id)
        t1_addr = get_unique_file(os.path.join(sub_dir, 't1'))
        flair_addr = get_unique_file(os.path.join(sub_dir, 'flair'))
        reg_file, reg_mat = reg_outfiles(t1_addr, self.proc_dir, sub_id)

        flt = fsl.FLIRT(bins=640, cost_func='mutualinfo')
        flt.inputs.in_file = t1_addr
        flt.inputs.reference = flair_addr
        flt.inputs.out_file = reg_file
        flt.inputs.out_matrix_file = reg_mat
        res = flt.run() 
        print('Registration of t1 to flair done: {}'.format(sub_id))
        return t1_addr, flair_addr, reg_file, reg_mat
    
    def apply_brainmask_to_t1(self, sub_id):
        t1 = self.sub_dict[sub_id]['T1_reg2FL']
        mask = self.sub_dict[sub_id]['mask_FL_brain']
        t1_meta = nib.load(t1)
        t1_brain = nib.load(t1).get_fdata() * nib.load(mask).get_fdata()
        t1_brain_save = nib.Nifti1Image(t1_brain.astype('int32'), t1_meta.affine, t1_meta.header)
        save_addr = os.path.join(self.proc_dir, sub_id, t1.split('/')[-1].split('.')[0]+'_brain.nii')
        nib.save(t1_brain_save, save_addr)
        print('Generation of T1_reg2FL_brain done.')
        return save_addr
    
    def flair_brain_extract(self, sub_id):
        sub_dir = os.path.join(self.raw_dir, sub_id)
        flair_addr = get_unique_file(os.path.join(sub_dir, 'flair'))
        bet_outfile, bet_outmask = bet_outfiles(flair_addr, self.proc_dir, sub_id)   
        filename = flair_addr.split('/')[-1]
        print('processing {} ...'.format(filename))

        ans = run_bet(flair_addr, bet_outfile, bet_outmask, cbool=False, fbool=False)    
        if ans == 'y':
            print('Brain extraction successful at once!')
        else:
            while ans != 'y':
                print('Give either [center of gravity] and/or [frac]')
                center_val = input("center [3D points comma-separated] or 0 : ")
                center = False if center_val == 0 else True
                center_val = [int(coord) for coord in center_val.split(',')]
                frac_val = float(input("frac (0.0,1.0) or 0: "))
                frac = False if frac_val == 0 else True
                print(center, frac, center_val, frac_val)
                ans = run_bet(flair_addr, bet_outfile, bet_outmask, cbool=center, fbool=frac, center_of_gravity=center_val, frac=frac_val)
            print('Brain extracted with center: {}, frac: {}'.format(center_val, frac_val))
        return bet_outfile, bet_outmask
    
    def flair_brain_extract_dynamic(self, flair_addr, bet_outfile, bet_outmask):
        filename = flair_addr.split('/')[-1]
        print('processing {} ...'.format(filename))

        ans = run_bet(flair_addr, bet_outfile, bet_outmask, cbool=False, fbool=False)    
        if ans == 'y':
            print('Brain extraction successful at once!')
        else:
            while ans != 'y':
                print('Give either [center of gravity] and/or [frac]')
                center_val = input("center [3D points comma-separated] or 0 : ")
                center = False if center_val == 0 else True
                center_val = [int(coord) for coord in center_val.split(',')]
                frac_val = float(input("frac (0.0,1.0) or 0: "))
                frac = False if frac_val == 0 else True
                print(center, frac, center_val, frac_val)
                ans = run_bet(flair_addr, bet_outfile, bet_outmask, cbool=center, fbool=frac, center_of_gravity=center_val, frac=frac_val)
            print('Brain extracted with center: {}, frac: {}'.format(center_val, frac_val))
        return 0
    
    
def _pickup_data_basic(sid, data):
    t1 = data[sid]['T1_orig']
    flair = data[sid]['FL_orig']
    flair_brain = data[sid]['FL_brain']
    mat_T12FL = data[sid]['mat_T12FL']  
    return t1, flair, flair_brain, mat_T12FL    

def _pickup_data_pred(sid, data, predkey):
    pred = data[sid][predkey]
    t1 = data[sid]['T1_orig']
    mat_T12FL = data[sid]['mat_T12FL'] 
    mat_T12mni = data[sid]['mat_T12mni']      
    return pred, t1, mat_T12FL, mat_T12mni

    
def run_bet(in_file, out_file, out_mask, cbool=False, fbool=False, center_of_gravity=None, frac=None):
    bet = fsl.BET(mask=True)
    bet.inputs.in_file = in_file
    bet.inputs.out_file = out_file
    if cbool:
        bet.inputs.center = center_of_gravity
    if fbool:
        bet.inputs.frac = frac
    res = bet.run()
    
    # visualize three cuts
    z_dim = nib.load(out_file).header['dim'][3]
    for cut in range((z_dim//5)*1, (z_dim//5)*4, z_dim//5):
        plot_many(in_file, out_file, out_mask, Title = ['orig', 'brain', 'mask'], cut=cut)
    ans = input("Well extracted? [y/n] : ")
    return ans

def get_unique_file(addr):
    """
    From the target folder, assert only one nifti (or mat) file exists, and return the nifti file.
    """
    all_ = []
    extension = ['nii', 'nii.gz', 'mat', 'h5']
    for ext in extension:
        all_ += glob.glob(addr + '/*.{}'.format(ext))
    assert len(all_) == 1, 'a single file allowed'
    return all_[0]

def reg_outfiles(in_file, processed_dir, sub_id):
    out_dir = os.path.join(processed_dir, sub_id)
    makedir(out_dir)
    filename = in_file.split('/')[-1]
    out_file = os.path.join(out_dir, '{}_reg.nii.gz'.format(filename.split('.')[0]))
    out_mat = os.path.join(out_dir, '{}_reg.mat'.format(filename.split('.')[0]))
    return out_file, out_mat

def bet_outfiles(in_file, processed_dir, sub_id):
    out_dir = os.path.join(processed_dir, sub_id)
    makedir(out_dir)
    filename = in_file.split('/')[-1]
    out_file = os.path.join(out_dir, '{}_brain.nii.gz'.format(filename.split('.')[0]))
    out_mask = os.path.join(out_dir, '{}_brain_mask.nii.gz'.format(filename.split('.')[0]))
    return out_file, out_mask

