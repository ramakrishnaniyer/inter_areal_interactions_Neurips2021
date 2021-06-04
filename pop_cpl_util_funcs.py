import numpy as np
import pandas as pd

from sklearn.preprocessing import scale 
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression, PLSSVD
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LassoCV, Lasso, LinearRegression
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict

from matplotlib import pyplot as plt

def make_metrics_df_sub_for_tgt_cstr(df_sub, experiment_id, cstr = None):#, src_cstr):
    
    c1 = (df_sub.ecephys_session_id==experiment_id)
    c2 = (df_sub.amplitude_cutoff<0.1)
    c3 = (df_sub.presence_ratio>0.9)
    c4 = (df_sub.isi_violations<0.5)
    
    if cstr is not None:
        c5 = (df_sub.ecephys_structure_acronym==cstr)
        df_sub_for_src_tgt_cstr = df_sub[c1 & c2 & c3 & c4 & c5]
    else:
        df_sub_for_src_tgt_cstr = df_sub[c1 & c2 & c3 & c4]
    
    return df_sub_for_src_tgt_cstr

def bin_spt_for_expt(x,tbins=np.arange(0.,10000.,0.03333)):
    
    spt_binned,bins = np.histogram(x,tbins)
    return spt_binned

def calc_tot_FR_len(start_times_list,end_times_list,dt):
    deltaT_l = [int(ii) for ii in end_times_list-start_times_list]
    total_T = np.sum(deltaT_l)
    
    total_FR_len = int(total_T/dt)
    print(total_T,dt,total_FR_len)
    
    return total_FR_len

def obtain_and_bin_spt_for_given_cstr(spt_df1_sub,start_times_list,end_times_list, dt, narg_out = 3):
    
    tot_FR_len = calc_tot_FR_len(start_times_list,end_times_list,dt)
    num_units = spt_df1_sub.shape[0]
    
    event_times_list_per_trial = {}
    binned_spt_per_trial_dict = {}
    total_fr_arr = np.zeros((num_units,tot_FR_len))
    
    for jj,unit in enumerate(spt_df1_sub.index.values):
        
        spt = spt_df1_sub.loc[unit].spike_times
        event_times_list_per_trial[unit] = []
        binned_spt_per_trial = []
        for tt in range(len(start_times_list)):
            
            stt_time = start_times_list[tt]
            end_time = end_times_list[tt]
            durn = int(end_time-stt_time)
            
            spt_in_interval = spt[(spt>stt_time)&(spt < end_time)]-stt_time
            binned_spt_per_trial.append(bin_spt_for_expt(spt_in_interval,tbins=np.arange(0,durn+dt,dt)))
                      
            if len(spt_in_interval)>0:
                event_times_list_per_trial[unit].append(spt_in_interval)
            else:
                event_times_list_per_trial[unit].append([])
            
            binned_spt_per_trial_dict[unit] = binned_spt_per_trial
                
        total_fr_arr[jj,:] = np.hstack(binned_spt_per_trial)
    
    total_fr_arr_df = pd.DataFrame(total_fr_arr, index = spt_df1_sub.index.values)
    
    if narg_out == 3:
        return event_times_list_per_trial, binned_spt_per_trial_dict, total_fr_arr
    else:
        return total_fr_arr_df


def load_stimulus_filtered_array(stim_arr_fname, stim_durn, dt):
    upsmp_fac = round(stim_durn/dt) 
    print(stim_durn, dt, upsmp_fac)
    stim_filt_arr = np.load(stim_arr_fname)
    print(stim_filt_arr.shape)

    stim_filt_arr_upsmp = np.repeat(stim_filt_arr.T, upsmp_fac, axis=1)

    return stim_filt_arr_upsmp

def LSTM_fit(X, y, split_frac=0.5, epoch=50, batch_size=96, nhidden_neurons=2, loss='mae', optimizer='adam'):
    # Split into training and test sets
    X_train, X_test , y_train, y_test = model_selection.train_test_split(X, y, test_size=split_frac, random_state=1)
    
    # reshape the data
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    
    # design network
    model = Sequential()
    model.add(LSTM(nhidden_neurons, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1))
    model.compile(loss=loss, optimizer=optimizer)
    
    # fit network
    history = model.fit(X_train, y_train, epochs=epoch, batch_size=batch_size, validation_data=(X_test, y_test), verbose=2, shuffle=False)
    # plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
    

def fit_lasso(X, y, split_frac = 0.5, cv = 10):
    
    # Split into training and test sets
    X_train, X_test , y_train, y_test = model_selection.train_test_split(X, y, test_size=split_frac, random_state=1)

    #Lasso CV
    lasso = Lasso(max_iter = 10000, normalize = False, fit_intercept=True)#False)
    lassocv = LassoCV(alphas = None, cv = cv, max_iter = 10000, normalize = False, fit_intercept=True)#False)
    lassocv.fit((X_train), y_train)

    lasso.set_params(alpha=lassocv.alpha_)
    lasso.fit((X_train), y_train)
    pred_train = np.squeeze(lasso.predict((X_train)))
    pred = np.squeeze(lasso.predict((X_test)))

    train_corr = np.corrcoef(y_train,pred_train)[0,1]
    mse = mean_squared_error(y_test, pred)
    test_corr = np.corrcoef(y_test,pred)[0,1]
    nnz_coef = len(lasso.coef_[np.abs(lasso.coef_>1e-10)])

    return lassocv.alpha_, train_corr, test_corr, lasso.coef_, nnz_coef, mse



def fit_lasso_on_test(X, y, split_frac = 0.5, cv = 10):
    
    # Split into training and test sets
    X_train, X_test , y_train, y_test = model_selection.train_test_split(X, y, test_size=split_frac, random_state=1)

    #Lasso CV
    lasso = Lasso(max_iter = 10000, normalize = False, fit_intercept=False)
    lassocv = LassoCV(alphas = None, cv = cv, max_iter = 10000, normalize = False, fit_intercept=False)
    lassocv.fit((X_test), y_test)

    lasso.set_params(alpha=lassocv.alpha_)
    lasso.fit((X_test), y_test)
    pred_test = np.squeeze(lasso.predict((X_test)))
    pred = np.squeeze(lasso.predict((X_train)))

    train_corr = np.corrcoef(y_test,pred_test)[0,1]
    mse = mean_squared_error(y_train, pred)
    test_corr = np.corrcoef(y_train,pred)[0,1]
    nnz_coef = len(lasso.coef_[np.abs(lasso.coef_>1e-10)])

    return lassocv.alpha_, train_corr, test_corr, lasso.coef_, nnz_coef, mse