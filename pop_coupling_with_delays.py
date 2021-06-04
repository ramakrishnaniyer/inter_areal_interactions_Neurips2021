import numpy as np
import os,sys
import yaml
import pandas as pd
import time
import random
import argparse

from allensdk.brain_observatory.ecephys.ecephys_session import EcephysSession

from pop_cpl_util_funcs import load_stimulus_filtered_array, fit_lasso, obtain_and_bin_spt_for_given_cstr, fit_lasso_on_test, LSTM_fit

class PC_sandbox:
    def __init__(self, yaml_fname = 'sg_a2a_wd_si.yaml', expt_id = 715093703, \
                 cstr = 'VISp', unit_id = 950930407, stim_scale = 1, flag_del = False, flag_inst_and_del = False, flag_fit_on_test = False, flag_permute = False):
        
        config = yaml.load(open(yaml_fname, 'r'))
        self.expt_id = str(expt_id)
        self.data_dir = config['data_dir']
        self.nwb_path = os.path.join(self.data_dir,'ecephys_session_'+ self.expt_id + '.nwb')
        self.yaml_fname = yaml_fname
        
        self.cstr = cstr 
        self.unit_id = unit_id

        self.frac = config['frac']
        self.stim_name = config['stim_name']
        
        self.stim_scale = stim_scale
        self.cstr_scale = config['cstr_scale']
        self.flag_del = flag_del
        self.flag_inst_and_del = flag_inst_and_del
        self.flag_fit_on_test = flag_fit_on_test
        self.flag_permute = flag_permute
                
        self.fit_type = config['fit_type']
        self.glm_method = config['glm_method']
        
        self.bin_start = config['bin_start']
        self.bin_end = config['bin_end']
        self.bin_dt = config['bin_dt']
        self.bin_edges = np.arange(self.bin_start, self.bin_end + 0.001, self.bin_dt)
        
        self.prs_savedir =  os.path.join(config['prs_savedir'], self.expt_id, self.cstr, \
            self.stim_name, 't1_'+str(self.bin_start) + '_t2_'+str(self.bin_end) + '_dt_'+str(self.bin_dt))

        if not os.path.exists(self.prs_savedir):
            os.makedirs(self.prs_savedir)
        
        separator = '_'
        prs_save_str = separator.join([str(self.unit_id), 'fr_scale',str(self.cstr_scale), 'stim_scale', str(self.stim_scale)]) + '.pkl'
        self.prs_savename = os.path.join(self.prs_savedir,prs_save_str)
        
        if not os.path.isfile(self.prs_savename):
            ##### MAIN STEPS
            self.load_session_and_get_fr_df()
            
            #self.make_output_vector()
            self.make_cstr_input_matrix()
            self.make_stim_input_matrix()
            
            self.input_X = np.concatenate((self.X_cstr, self.X_stim.T), axis=1)
            print('Shape of input is :', self.input_X.shape)

            self.get_prs_df_col_names()
            self.get_glm_fit_prs_new()
            #################
        else:
            print('Params file for this unit already exists')

    def get_prs_df_col_names(self):
        self.col_names = ['sim_prs_yml','unit_id','alpha_val','mse','nnz_coef','cstr_scale','stim_scale','train_corr','test_corr']
        if self.cstr_scale != 0: 
            self.col_names = self.col_names + list(self.tot_fr_df.T.columns.values)
        
        if np.shape(self.X_cstr)[1] > np.shape(self.tot_fr_df.T)[1]:
            del_col_names = [str(nn)+'-del' for nn in self.tot_fr_df.T.columns.values]
            self.col_names = self.col_names + del_col_names
        
        if self.stim_scale != 0:
            self.col_names = self.col_names + list(range(self.X_stim.shape[0]))   

    def load_session_and_get_fr_df(self):
        session = EcephysSession.from_nwb_path(self.nwb_path, api_kwargs={
        "amplitude_cutoff_maximum": 0.1,
        "presence_ratio_minimum": 0.9,
        "isi_violations_maximum": 0.5
            })

        self.units_df = session.units
        
        if 'spontaneous' in self.stim_name:
            spt_df = pd.DataFrame(sorted(session.spike_times.items()),index = session.units.index.values,columns=['unit_id','spike_times'])
            spt_df1 = pd.merge(session.units,spt_df,how='inner',on=['unit_id'])
            spt_df1.set_index('unit_id',inplace=True)
            
            stim_epochs = session.get_stimulus_epochs()
            if 'combined' in self.stim_name:
                spont_epochs = stim_epochs[(stim_epochs.stimulus_name=='spontaneous')|(stim_epochs.stimulus_name=='static_gratings')]
            else:
                spont_epochs = stim_epochs[stim_epochs.stimulus_name=='spontaneous']

            stt_l = spont_epochs.start_time.values
            endt_l = spont_epochs.stop_time.values
            self.tot_fr_df = obtain_and_bin_spt_for_given_cstr(spt_df1,stt_l,endt_l, self.bin_dt, narg_out = 1)
            self.y = self.tot_fr_df.T[self.unit_id].values.astype(float)
            
        else:
            self.stim_table = session.get_stimulus_table(self.stim_name)
            stim_pres_ids = self.stim_table.index.values
            tmp_binned_spt = session.presentationwise_spike_counts\
                (bin_edges = self.bin_edges, stimulus_presentation_ids=stim_pres_ids, unit_ids = session.units.index.values)
            
            num_pres,num_bins,num_cells = tmp_binned_spt.shape
            print(num_pres,num_bins,num_cells)
            tot_arr_fr_all = np.reshape(tmp_binned_spt.values, (num_pres*num_bins,num_cells))
            self.tot_fr_df = pd.DataFrame(tot_arr_fr_all.T, index = session.units.index.values)
            self.y = self.tot_fr_df.T[self.unit_id].values.astype(float)
            
            if self.flag_permute == True:
                print('Permuting trials')
                
                ### Trying permutation without caring about stimulus condition on Dec 1, 2020
                
                #tmp_binned_spt_perm = self.permute_fr_arr(tmp_binned_spt) #(this line orig with tracking stim cond)
                
                #tmp_binned_spt_perm1 = np.random.permutation(tmp_binned_spt) #(this line had the wrong shuffle control, each neuron in the shuffled data was still looking at the same input grating).
                
#                 #Making a change on May 20, 2021 after discussion
#                 tmp_binned_spt_perm = np.zeros(tmp_binned_spt.shape)
#                 for cc in range(tmp_binned_spt_perm.shape[-1]):
#                     tmp_binned_spt_perm[:,:,cc] = np.random.permutation(tmp_binned_spt[:,:,cc])
                    
#                 ####
                
#                 tot_arr_fr_all_perm = np.reshape(tmp_binned_spt_perm, (num_pres*num_bins,num_cells))  

                ### Making a change on May 26, 2021 to try an extreme version of shuffling
                tot_arr_fr_all_perm_pre = np.reshape(tmp_binned_spt.values, (num_pres*num_bins,num_cells))
                tot_arr_fr_all_perm = np.zeros(tot_arr_fr_all_perm_pre.shape)
                for nn in range(tot_arr_fr_all_perm.shape[1]):
                    tot_arr_fr_all_perm[:,nn] = np.random.permutation(tot_arr_fr_all_perm_pre[:,nn])
                
                
                self.tot_fr_df = pd.DataFrame(tot_arr_fr_all_perm.T, index = session.units.index.values)
                print('Shape of permuted FR array is :', self.tot_fr_df.shape)
                #self.y = self.tot_fr_df.T[self.unit_id].values.astype(float)
                
            
            del tmp_binned_spt      

        del session
        
    def permute_fr_arr(self, binned_spt):
        print('Permuting trials using function')
        stim_cond_id_list = self.stim_table.stimulus_condition_id.unique()
        tmp_binned_spt_perm = np.zeros(binned_spt.shape)
        for cc in range(tmp_binned_spt_perm.shape[-1]):
            for scid in stim_cond_id_list:
                inds = np.where(self.stim_table.stimulus_condition_id == scid)[0]
                tmp_binned_spt_perm[inds,:,cc] = binned_spt[np.random.permutation(inds),:,cc]
                
        return tmp_binned_spt_perm
        
       
    #def make_output_vector(self):
    #    self.y = self.tot_fr_df.T[self.unit_id].values.astype(float)

    def make_cstr_input_matrix(self):
        self.X_cstr = np.array([]).reshape(self.tot_fr_df.shape[1],0)
        if self.cstr_scale != 0:
            df_X = self.tot_fr_df.T.copy()
            if self.fit_type == 'all_to_all':
                
                if self.flag_del == False:
                    df_X[self.unit_id] = np.zeros(self.y.shape)
                    self.X_cstr = self.cstr_scale*np.array(df_X)
                else:
                    
                    #To add delay ONLY TO CSTR, not STIM (modified on Jul 2, 2020)
                    #Not sure why I zeroed out self activity with delay, so putting it back (modified Feb 3, 2021)
                    #Also padding went y0 = x1 (future) instead of y1 = x0 (previous bin predicts current)
                    #So changing padding and checking
                    
                    num_del_pts = 1
                    #df_X[self.unit_id] = np.zeros(self.y.shape) #(modified on Feb 3, 2021)
                    print('Max output unit FR is :', np.amax(df_X[self.unit_id].values.astype(float)))
                    self.X_cstr = self.cstr_scale*np.array(df_X)
                    
                    if self.flag_inst_and_del == False:
                        self.X_cstr = np.pad(self.X_cstr, ((num_del_pts,0),(0,0)), 'constant', constant_values = 0.)
                        self.X_cstr = self.X_cstr[:-num_del_pts,:]
                        #self.X_cstr = np.pad(self.X_cstr, ((0,num_del_pts),(0,0)), 'constant', constant_values = 0.)
                        #self.X_cstr = self.X_cstr[num_del_pts:,:]
                    else:
                        X_cstr_del = np.pad(self.X_cstr, ((num_del_pts,0),(0,0)), 'constant', constant_values = 0.)
                        X_cstr_del = X_cstr_del[:-num_del_pts,:]
                        self.X_cstr = np.hstack((self.X_cstr,X_cstr_del))


            del df_X
        print('Cstr shape is :', self.X_cstr.shape)

    def make_stim_input_matrix(self):
        if 'combined' in self.stim_name:
            stim_arr_fname =  'gabor_filtered_static_gratings_stim_combined_with_spont.npy' 
            self.X_stim = self.stim_scale*np.load(stim_arr_fname)
        else:
            stim_arr_fname =  'gabor_filtered_'+self.stim_name+'_stim_corr.npy'
            self.X_stim = np.array([]).reshape(0,self.tot_fr_df.shape[1])
            if self.stim_scale != 0:
                stim_durn = self.bin_end - self.bin_start
                self.X_stim = self.stim_scale*load_stimulus_filtered_array(stim_arr_fname, stim_durn, self.bin_dt)
                
        print('Stim shape is :', self.X_stim.shape)

    def get_glm_fit_prs_new(self):
        
        print('Starting to analyze unit ',  self.unit_id)

        print('Maximum input is: ', np.amax(self.input_X))
        print('Maximum output FR is: ', np.amax(self.y))
        prs_df_unit = pd.DataFrame(index=range(1),columns = self.col_names)

        if self.flag_fit_on_test == False:
            alpha_val, train_corr, true_test_corr, params, nnz_coef, mse = fit_lasso(self.input_X, self.y, self.frac, cv = 10)
        else:
            alpha_val, train_corr, true_test_corr, params, nnz_coef, mse = fit_lasso_on_test(self.input_X, self.y, self.frac, cv = 10)
        
        
        print('Finished analyzing unit ',  self.unit_id)
        print(self.unit_id, train_corr, true_test_corr, nnz_coef, mse)
        print('Optimal alpha_val: ',alpha_val)
        
        prs_df_unit.iloc[0]['sim_prs_yml'] = self.yaml_fname
        prs_df_unit.iloc[0]['unit_id'] = self.unit_id
        prs_df_unit.iloc[0]['alpha_val'] = alpha_val
        prs_df_unit.iloc[0]['mse'] = mse
        prs_df_unit.iloc[0]['nnz_coef'] = nnz_coef
        prs_df_unit.iloc[0]['cstr_scale'] = self.cstr_scale
        prs_df_unit.iloc[0]['stim_scale'] = self.stim_scale
        prs_df_unit.iloc[0]['train_corr'] = train_corr
        prs_df_unit.iloc[0]['test_corr'] = true_test_corr
        prs_df_unit.iloc[0,9:] = params

        print('Number of non-zero stim coeffs: ', np.count_nonzero(prs_df_unit.iloc[0,-120:].values))
        print(np.amax(prs_df_unit.iloc[0,-120:].values))
        print(np.amin(prs_df_unit.iloc[0,-120:].values))
        
        self.prs_df_unit = prs_df_unit

        prs_df_unit.to_pickle(self.prs_savename)
        
 
if __name__ == '__main__':

    random.seed(9)
    
    ## Flag for delay added on Jul 8, 2020
    flag_del = True
    flag_inst_and_del = False
    flag_fit_on_test = True #True#False#True #False
    flag_permute = True #True #False

    # Parse arguments:
    parser = argparse.ArgumentParser(description='Number of clusters')

    parser.add_argument('--expt_id', type=str, required=True)
    parser.add_argument('--cstr', type = str, required = True)
    parser.add_argument('--unit_id', type = str, required = True)
    parser.add_argument('--prs_yaml_fname', type = str, required = True)
    parser.add_argument('--stim_scale', type = float, required = True)

    #Parse all arguments provided
    args = parser.parse_args()
    eid = args.expt_id
    uid = int(args.unit_id)
    cstr = args.cstr
    yaml_fname = args.prs_yaml_fname
    stim_scale = args.stim_scale

    sim_stt = time.time()
    pc = PC_sandbox(yaml_fname = yaml_fname, expt_id = eid, cstr = cstr, unit_id = uid, stim_scale = stim_scale, flag_del = flag_del, flag_inst_and_del = flag_inst_and_del, flag_fit_on_test = flag_fit_on_test, flag_permute = flag_permute)
    sim_endt = time.time()

    print('Time taken to fit: ', sim_endt-sim_stt)
