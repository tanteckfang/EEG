# FILE Name: analyze_trial.py
import numpy as np
from scipy.io import loadmat
import os
import matplotlib.pyplot as plt



def extract_data(randomized_id, mat_file_list, out_file_list, offset_s = 0.5, duration_s=3.0, unshuffle=False):
    """
    Function to extract own static experiments to format for DL 
    Argument:
        randomized_id: type int. ID of the test subject. All trials will be in this folder
        mat_file_list: type list. List of .mat file to decode
        out_file_list: type list. List of .out file to decode
        offset_s: type float. offset (in seconds) from pulse signal to start collecting the samples
        duration_s: type float. duration (in seconds) of EEG samples to collect
        unshuffle: type bool
            if True, perform unshuffle for dataset. I.e. extract class from file.out, then unshuffle them such that classes are in class 0,1,2,3,0,1,2,...,2,3
            So that all analysis scripts that assumes trials are in order of class 0,1,2,3,0,1,... can still work
    Return:
        data: type eeg_data 
            data.X: np.ndarray of shape (subject_num, trial_num, sample_num, channel_num)
            data.Y: np.ndarray of shape (subject_num, trial_num, 1)
            data.acc: np.ndarray of shape (subject_num, trial_num, sample_num, channel_num =3)
            data.sampleRateHz: type int of value 250
    """
#     MAT_FILE_STR[264] = 'Run344.mat'; OUT_FILE_STR[264] = 'Run344.out'
    
    # Check validity of input before further processing
    assert_extract_data(randomized_id, mat_file_list, out_file_list, offset_s, duration_s, unshuffle)
    
    # Create eeg_data class for output
    class eeg_data:
        pass
    output = eeg_data()
    
        
    for i in range(len(mat_file_list)):
        if i == 0:
            # Load .mat and .out file into memory
            mat_file = load_ssvep_mat(randomized_id, mat_file_list[i])
            label_file = load_ssvep_out_label(randomized_id, out_file_list[i])

            # Retrieve respective fields from .mat file
            data = get_fields(mat_file, 'data')

            # get valid EEG samples and label
            eeg_sample = get_valid_samples(data,offset_s=offset_s,duration_s=duration_s)
            eeg_label = get_labels(label_file)

        else:
            # Load .mat and .out file into memory
            mat_file = load_ssvep_mat(randomized_id, mat_file_list[i])
            label_file = load_ssvep_out_label(randomized_id, out_file_list[i])

            # Retrieve respective fields from .mat file
            data = get_fields(mat_file, 'data')

            # get valid EEG samples and label
            eeg_sample_temp = get_valid_samples(data,offset_s=offset_s,duration_s=duration_s)
            eeg_label_temp = get_labels(label_file)

            # Concatenate on subject_num
            eeg_sample = np.concatenate((eeg_sample, eeg_sample_temp), axis=0)
            eeg_label = np.concatenate((eeg_label, eeg_label_temp), axis=0)

                
    output.sampleRateHz = 250
    if eeg_sample.shape[3] <= 8:   # Only EEG data available
        output.X = eeg_sample
    else:
        output.X = eeg_sample[:,:,:,:8]
        output.acc = eeg_sample[:,:,:,8:]

    output.Y = eeg_label

    if unshuffle == True:  # Perform unshuffling of dataset
        for i in range(output.X.shape[0]):   # For each subject, do unshuffle

            X_copy = np.copy(output.X[i])
            Y_copy = np.copy(output.Y[i])

            # Get index based on ascending order, then reshape to interleave classes to 0,1,2,3,0,1,...
            index = np.argsort(Y_copy, 0)
            index = np.reshape(index, (4,Y_copy.shape[0]//4))  # By Default, order='C'
            index = np.reshape(index, -1, order='F')

            # Using sorted index to sort the data
            Y_sorted = np.take(Y_copy, index, axis=0)
            X_sorted = np.take(X_copy, index, axis=0)

            # For accelerometer reading (if applicable)
            if eeg_sample.shape[3] > 8:   # Has accelerometer readings
                acc_copy = np.copy(output.acc[i])
                acc_sorted = np.take(acc_copy, index, axis=0)
                output.acc[i] = acc_sorted

            output.X[i] = X_sorted
            output.Y[i] = Y_sorted

    return output



def assert_extract_data(randomized_id, mat_file, out_file, offset_s, duration_s, unshuffle):
    assert type(randomized_id) is int, '"randomized_id" must be of int type'
    assert type(mat_file) is list, '"mat_file_list" must be of type list'
    assert type(out_file) is list, '"out_file_list" must be of type list'
    assert len(mat_file) == len(out_file), 'length of "mat_file" and "out_file" must be equal'
    assert type(unshuffle) is bool, '"unshuffle" must be of type bool'




def load_ssvep_mat(randomized_id, filename):
    CURRENT_DIR = os.path.dirname(__file__)    # Use os.getcwd() on .ipynb; os.path.dirname(__file__) on .py
    FILE_DIR_NAME = os.path.join(CURRENT_DIR, '..','..', 'WalkingWizardTrial_2022_03', str(randomized_id))
    file = loadmat(os.path.join(FILE_DIR_NAME, filename))
    
    return file


def load_ssvep_out_label(randomized_id, filename):
    CURRENT_DIR = os.path.dirname(__file__)
    FILE_DIR_NAME = os.path.join(CURRENT_DIR, '..','..', 'WalkingWizardTrial_2022_03', str(randomized_id))
    FILE_OUT = os.path.join(FILE_DIR_NAME, filename)

    with open(FILE_OUT) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    content = list(map(int, content))
    
    return content


    
    
def get_fields(mat_file, field='data'):
    """
    Function to retrieve the respective fields from the .mat file returned by function load_ssvep_mat
    Argument:
        mat_file: the output from load_ssvep_mat, in dict type
        field: type string. type of data to retrieve
            'data': retrieve the data, return will be in numpy.ndarray type with shape (10, num_samples)
                data[0,:]: Time stamp with interval 0.004s correspondingto sampling rate of 250Hz
                data[1,:]: Pulse signal corresponding to pressing of spacebar
                data[2:9,:]: EEG signal of electrodes 0 to 7
            'FirstName',... 'Medication': retrieve the metadata. return will be in string format
        
    Return:
        Respective output as determined by 'field'. If 'field' == 'data', output of numpy.ndarray type, else is string type
    """
    if field == 'data':
        return mat_file['y']
    elif field == 'FirstName':
        return str(mat_file['PAT'][0][0][0][0])
    elif field == 'LastName':
        return str(mat_file['PAT'][0][0][1][0])
    elif field == 'DateOfBirth':
        return str(mat_file['PAT'][0][0][2][0])
    elif field == 'Comment':
        return str(mat_file['PAT'][0][0][3][0])
    elif field == 'Session':
        return str(mat_file['PAT'][0][0][4][0])
    elif field == 'Run':
        return str(mat_file['PAT'][0][0][5][0])
    elif field == 'ID':
        return str(mat_file['PAT'][0][0][6][0])
    elif field == 'Sex':
        return str(mat_file['PAT'][0][0][7][0])
    elif field == 'Hand':
        return str(mat_file['PAT'][0][0][8][0])
    elif field == 'Diagnosis':
        return str(mat_file['PAT'][0][0][9][0])
    elif field == 'Medication':
        return str(mat_file['PAT'][0][0][10][0])
    else:
        assert False, 'Invalid field option provided'
        
        
def get_valid_samples(data,offset_s=0.5,duration_s=1.5):
    """
    Function to get the valid EEG sample data from the input data file
    Argument:
        data: type numpy.ndarray of shape (10, samples)
            data[0,:]: Time stamp with interval 0.004s correspondingto sampling rate of 250Hz
            data[1,:]: Pulse signal corresponding to pressing of spacebar
            data[2:9,:]: EEG signal of electrodes 0 to 7
            data[10:12,:]: Accelerometer signal Ch10: X, Ch11: Y; Ch12: Z
        offset_s: type float. offset (in seconds) from pulse signal (data[1]) to start collecting the samples
        duration_s: type float. duration (in seconds) of EEG samples to collect
    Return:
        Return the EEG samples (i.e. data[2:9]) and accelerometer (i.e. data[10:12]) that are valid. Index of samples not returned. 
            Type np.ndarray of shape (subject_num, trial_num, sample_num, channel_num), where subject_num = 1, trial_num = 40
    """
    import numpy as np
    
    # Check input validity
    assert type(data) == np.ndarray, 'Input data must be of type np.ndarray'
    assert len(data.shape) == 2, 'Input data must be of shape (10, samples)'
#     assert data.shape[0] == 10, 'Input data.shape[0] must be 10'
    assert type(offset_s) == float, 'Input offset_s must be of type float'
    assert type(duration_s) == float, 'Input duration_s must be of type float'
    
    # Setup required variables
    index_valid = np.arange(len(data[1]))[data[1]==1]
    period = data[0,1] - data[0,0]
    offset_val = index_valid + int(offset_s/period)
    duration_sample = int(duration_s/period)
    
    # Extract valid sample to return
    output = np.array([], dtype = np.float64).reshape(0, 11, duration_sample) # Empty array for stacking
    for i in offset_val:
        temp = np.expand_dims(data[2:,i:i+duration_sample],0)
        output = np.vstack((output,temp))
    output = np.expand_dims(output, 0)
    output = np.transpose(output,(0,1,3,2))
    
    return output

def get_labels(label_file):
    """
    Function to get the shape the label_file to the format accepted for training
    Argument:
        label_file: type list. output of load_ssvep_out_label 
    Return:
        output: type numpy.ndarray of shape (subject_num, trial_num, 1), where subject_num = 1, trial_num = 40, with 4 classes
    """
    output = np.expand_dims(np.expand_dims(np.array(label_file),1),0)
    return output




        
def plot_agg_fft(X, randomized_id, to_save=False, to_show=True, title='FFT Plot Aggregated by Class and Channel', file_name = 'Agg Spectral', freq=[6.0, 6.667, 7.5, 8.571], agg_params = [True, True, 'mean', True, False], subject=0, channel=0):
    """
    Function to plot and save the aggregated FFT plot, showing the classes 0 - 3. 
        X        : type np.ndarray of shape (subject, trial, sample, channel)
        randomized_id: type int. ID of the test subject. All trials will be in this folder
        to_save  : type bool. True to indicate to save
        to_show  : type bool. True to indicate to show plot
        title    : type str. Title to show on the plot.
        file_name : type str. file name to save the file.
        freq     : type list. To indicate the target frequencies to be plotted
        agg_params : type list. Provide the parameters for aggregation. [mean_channel , mean_subj , aggregate type, meas_noise_electrode , all_class]
        subject  : type int. The subject to plot (applicable if agg_params[1] == False, else only 0)
        channel  : type int. The channel to plot (applicable if agg_params[0] == False, else only 0 for EEG, 1 for Noise electrode)
    """
    # Apply FFT on data
    xf, yf  = apply_fft(X, spectrum_type='amplitude')
    
    # Aggregate the spectral data
    mean_yf =  aggregate_over_class(yf, mean_channel=agg_params[0], mean_subj=agg_params[1], aggregate=agg_params[2], meas_noise_electrode=agg_params[3], all_class=agg_params[4])
    mean_noise =  aggregate_over_class(yf, mean_channel=False, mean_subj=agg_params[1], aggregate=agg_params[2], meas_noise_electrode=agg_params[3], all_class=agg_params[4])
    
    
    # Plot data
#     fig, axs = plt.subplots(2, sharex=True, sharey=True)
    plt.rcParams["figure.figsize"] = (15,15)
    fig, axs = plt.subplots(2)
    fig.suptitle(title)
    
    
    # Plot EEG signal without noise electrode
    axs[0].plot(xf,  mean_yf [subject,0,:,channel])   # [subject, class / trial, sample, channel]
    axs[0].plot(xf,  mean_yf [subject,1,:,channel])   # [subject, class / trial, sample, channel]
    axs[0].plot(xf,  mean_yf [subject,2,:,channel])   # [subject, class / trial, sample, channel]
    axs[0].plot(xf,  mean_yf [subject,3,:,channel])   # [subject, class / trial, sample, channel]
    
    # Plot EEG signal after subtract noise electrode, channel 4 is noise electrode
    axs[1].plot(xf,  mean_yf [subject,0,:,channel] - mean_noise[subject,0,:,4])   # [subject, class / trial, sample, channel]
    axs[1].plot(xf,  mean_yf [subject,1,:,channel] - mean_noise[subject,1,:,4])   # [subject, class / trial, sample, channel]
    axs[1].plot(xf,  mean_yf [subject,2,:,channel] - mean_noise[subject,2,:,4])   # [subject, class / trial, sample, channel]
    axs[1].plot(xf,  mean_yf [subject,3,:,channel] - mean_noise[subject,3,:,4])   # [subject, class / trial, sample, channel]
    
    
    subtitle = ['Mean EEG Signal for each class', 'Mean EEG Signal - Noise for each class']
    sub_ylim = [[0,4],[-3,3]]
    plt.xlabel('Freq / Hz')
    i = 0
    
    for ax in axs.flat:
        ax.set_ylabel('Amplitude / uV')
        ax.set_title(subtitle[i])
        ax.legend(['Class 0', 'Class 1', 'Class 2', 'Class 3'])
        ax.set_ylim(sub_ylim[i])
        ax.set_xlim([0,30])
        ax.tick_params(labelrotation=315)
        ax.set_xticks([2.0,30,freq[0],freq[1],freq[2],freq[3],freq[0]*2,freq[1]*2,freq[2]*2,freq[3]*2,
                       freq[0]*3,freq[1]*3,freq[2]*3,freq[3]*3,freq[0]*4,freq[1]*4,freq[2]*4])
        ax.axvline(x=freq[0], color='b',linewidth=0.5)
        ax.axvline(x=freq[0]*2, color='b',linewidth=0.5)
        ax.axvline(x=freq[0]*3, color='b',linewidth=0.5)
        ax.axvline(x=freq[0]*4, color='b',linewidth=0.5)

        ax.axvline(x=freq[1], color='y',linewidth=0.7)
        ax.axvline(x=freq[1]*2, color='y',linewidth=0.7)
        ax.axvline(x=freq[1]*3, color='y',linewidth=0.7)
        ax.axvline(x=freq[1]*4, color='y',linewidth=0.7)

        ax.axvline(x=freq[2], color='g',linewidth=0.5)
        ax.axvline(x=freq[2]*2, color='g',linewidth=0.5)
        ax.axvline(x=freq[2]*3, color='g',linewidth=0.5)
        ax.axvline(x=freq[2]*4, color='g',linewidth=0.5)

        ax.axvline(x=freq[3], color='r',linewidth=0.5)
        ax.axvline(x=freq[3]*2, color='r',linewidth=0.5)
        ax.axvline(x=freq[3]*3, color='r',linewidth=0.5)
#         ax.axvline(x=freq[3]*4, color='r',linewidth=0.5)
        ax.grid(True)
        i = i + 1

   
    if to_save == True:
        current_dir= os.path.dirname(__file__)
        save_dir = os.path.join(current_dir,'..', '..', 'WalkingWizardTrial_2022_03', str(randomized_id), 'image')
        file_name = file_name + '.png'
        savefile = os.path.join(save_dir, file_name)
        if os.path.isdir(save_dir):  # Check if directory exists
            pass
        else:
            os.mkdir(save_dir)
        plt.savefig(savefile, orientation='portrait',transparent=True, bbox_inches=None, pad_inches=0)
    if to_show == True:
        plt.show()
    else:
        plt.close()
        
        
def plot_time_series_ch(X, rms_interval_s=1):
    """
    Function to plot/save time-series data of input data X for each EEG channel (0-3) + Noise (4).
    Argument:
        X      : type np.ndarray of shape (sample, channel)
        rms_interval_s : type float or None. 
            If None, just perform rms on axis. 
            If float, perform rms on num_to_mean samples denoted by axis. And for every num_to_mean / 2 samples onwards, perform same rms.
    Return: 
        y_rms: the y_rms data
    """
    ## Static Indexes: 239, 243, 248, 256
    ## 1.5km/hr Indexes: 240, 244, 249 (lil noisy, class 1), 250 (class 1 issue)
    ## 3.0km/hr Indexes: 238, 241, 244, 245, 251, 258
    ## 4.5km/hr Indexes: 242, 246, 252, 259
    ## 5.0km/hr Indexes: 247, 254, 255, 261
    
    # Plot the rms of time-series broken down into short intervals to detect short pulses of noise (e.g. blink)
    t = np.arange(X.shape[0]) / 250
    ch_vec =['FP2','AF8', 'F8', 'Noise_Right', 'FP1', 'AF7','F7', 'Noise_Left']

    # RMS calculation
    y_rms = get_rms(X)
    y_rms_short = get_rms(X, num_to_mean = int(rms_interval_s * 250))
    x = np.arange(1,y_rms_short.shape[0]+1) / 2 * rms_interval_s

    # Plot
    plt.rcParams["figure.figsize"] = (15,10)
    fig, axs = plt.subplots(8, sharex=True, sharey=True)
    fig.suptitle('Calibration Plot')

    axs[0].plot(t,X[:,0])
    axs[1].plot(t,X[:,1])
    axs[2].plot(t,X[:,2])
    axs[3].plot(t,X[:,3])
    axs[4].plot(t,X[:,4])
    axs[5].plot(t,X[:,5])
    axs[6].plot(t,X[:,6])
    axs[7].plot(t,X[:,7])

    axs[0].plot(x,y_rms_short[:,0])
    axs[1].plot(x,y_rms_short[:,1])
    axs[2].plot(x,y_rms_short[:,2])
    axs[3].plot(x,y_rms_short[:,3])
    axs[4].plot(x,y_rms_short[:,4])
    axs[5].plot(x,y_rms_short[:,5])
    axs[6].plot(x,y_rms_short[:,6])
    axs[7].plot(x,y_rms_short[:,7])

    plt.xlabel('Time / s')
    channel_num = 0

    for ax in axs.flat:
        ax.label_outer()
        ax.minorticks_on()
        ax.grid(which='major', color='k')
        ax.grid(which='minor', color='y', linestyle='--')
        ax.set_ylabel('Amplitude / uV')
        ax.set_title('Channel Num: ' + str(channel_num) + ' (' + ch_vec[channel_num] +')')
        ax.legend(['Time-series','RMS = ' + '{0:.3f}'.format(y_rms[0,channel_num]) + 'uV'])
        ax.set_ylim([-50,50])
        channel_num = channel_num + 1
    plt.show()
    return y_rms




def plot_time_series_ch_rms_trial(X, rms_interval_s=1):
    """
    Function to plot/save time-series data of input rms data
    Argument:
        X      : type np.ndarray of shape (sample, channel). where sample is the rms per 20s interval
        rms_interval_s : type float or None. 
            If None, just perform rms on axis. 
            If float, perform rms on num_to_mean samples denoted by axis. And for every num_to_mean / 2 samples onwards, perform same rms.
    Return: 
        y_rms: the y_rms data
    """
    ## Static Indexes: 239, 243, 248, 256
    ## 1.5km/hr Indexes: 240, 244, 249 (lil noisy, class 1), 250 (class 1 issue)
    ## 3.0km/hr Indexes: 238, 241, 244, 245, 251, 258
    ## 4.5km/hr Indexes: 242, 246, 252, 259
    ## 5.0km/hr Indexes: 247, 254, 255, 261
    
    # Plot the rms of time-series broken down into short intervals to detect short pulses of noise (e.g. blink)
#     t = np.arange(X.shape[0]) / 250
    t = np.arange(X.shape[0]) / 0.05 / 60
    ch_vec =['FP2','AF8', 'F8', 'Noise_Right', 'FP1', 'AF7','F7', 'Noise_Left']

    # RMS calculation
    y_rms = get_rms(X)
#     y_rms_short = get_rms(X, num_to_mean = int(rms_interval_s * 250))
#     x = np.arange(1,y_rms_short.shape[0]+1) / 2 * rms_interval_s

    # Plot
    plt.rcParams["figure.figsize"] = (15,10)
    fig, axs = plt.subplots(8, sharex=True, sharey=True)
    fig.suptitle('Calibration Plot')

    axs[0].plot(t,X[:,0])
    axs[1].plot(t,X[:,1])
    axs[2].plot(t,X[:,2])
    axs[3].plot(t,X[:,3])
    axs[4].plot(t,X[:,4])
    axs[5].plot(t,X[:,5])
    axs[6].plot(t,X[:,6])
    axs[7].plot(t,X[:,7]*100)

#     axs[0].plot(x,y_rms_short[:,0])
#     axs[1].plot(x,y_rms_short[:,1])
#     axs[2].plot(x,y_rms_short[:,2])
#     axs[3].plot(x,y_rms_short[:,3])
#     axs[4].plot(x,y_rms_short[:,4])
#     axs[5].plot(x,y_rms_short[:,5])
#     axs[6].plot(x,y_rms_short[:,6])
#     axs[7].plot(x,y_rms_short[:,7])

    plt.xlabel('Time / min')
    channel_num = 0

    for ax in axs.flat:
        ax.label_outer()
        ax.minorticks_on()
        ax.grid(which='major', color='k')
        ax.grid(which='minor', color='y', linestyle='--')
        ax.set_ylabel('Amplitude / uV')
        ax.set_title('Channel Num: ' + str(channel_num) + ' (' + ch_vec[channel_num] +')')
        ax.legend(['Time-series','RMS = ' + '{0:.3f}'.format(y_rms[0,channel_num]) + 'uV'])
#         ax.set_ylim([-50,50])
        ax.set_ylim([0,500])
        channel_num = channel_num + 1
    plt.show()
    return y_rms



        
def aggregate_over_class(data_X, mean_subj = False, mean_channel = False, aggregate='mean', meas_noise_electrode=False, all_class=False, is_mobile_bci=False):
    """
    Function to get the aggregate (e.g. mean) of each class. For gSSVEP experiment, class label for trials are in this sequence 0>1>2>3>0>1>... . Class label is the same every 4 other trials. Use this relationship to obtain the aggregate of the signal (or spectrum, if perform fft before calling this function) across classes.
    Argument:
        data_X: input data of type np.ndarray with shape (subject, trial, sample, channel)
        mean_subj: type bool. If True, aggregate across subjects also.
        mean_channel: type bool. If True, aggregate across channels also.
        aggregate: type str. Type of aggregation to perform. 
        meas_noise_electrode: type bool. If True | mean_channel == True, aggregate channels 0,1,2,3 (measuring electrode), and 4,5,6,7 (noise electrodes) separately.
        all_class: type bool. If true, aggregate across all classes. 
        is_mobile_bci: type bool. If true, only 3 classes, [0,0,...,1,1...,2,2...]
    Return:
        output: type np.ndarray of shape (subject, CLASS, sample, channel). Where class=4 for gSSVEP as there are 4 classes. If mean_subj == True, subject = 1. If mean_channel == True, channel = 1 (if meas_noise_electrode == False) or channel = 2 (if meas_noise_electrode = True). If all_class=True, CLASS=1
    """
    # Get array of bool value to indicate trial number corresponding to respective class
    if is_mobile_bci == False:
        class0 = np.arange(data_X.shape[1])%4 == 0
        class1 = np.arange(data_X.shape[1])%4 == 1
        class2 = np.arange(data_X.shape[1])%4 == 2
        class3 = np.arange(data_X.shape[1])%4 == 3
    else: 
        class0 = np.arange(data_X.shape[1]) < 20
        class1 = np.logical_and(np.arange(data_X.shape[1]) >= 20, np.arange(data_X.shape[1]) < 40)
        class2 = np.arange(data_X.shape[1]) >= 40
    
    # Get requested aggregation function
    if aggregate == 'mean':
        agg = np.mean
    elif aggregate == 'max':
        agg = np.max
    elif aggregate == 'median':
        agg = np.median
    elif aggregate == 'L2':
        agg = np.linalg.norm
    else: 
        assert False, 'aggregate function not available'
    
    # Aggregate across classes
    class0 = agg(data_X[:,class0,:,:], axis=1, keepdims=True)
    class1 = agg(data_X[:,class1,:,:], axis=1, keepdims=True)
    class2 = agg(data_X[:,class2,:,:], axis=1, keepdims=True)
    if is_mobile_bci == False:
        class3 = agg(data_X[:,class3,:,:], axis=1, keepdims=True)
        output = np.concatenate((class0, class1, class2, class3), axis=1)
    else:
        output = np.concatenate((class0, class1, class2), axis=1)
    
    # Aggregate across subjects and channels (if app)
    if mean_subj == True:
        output = agg(output, axis=0, keepdims=True)

    if mean_channel == True:
        if meas_noise_electrode == False:    # Aggregate all channels together
            output = agg(output, axis=3, keepdims=True)
        else:                                # Aggregate measuring and noise electrode separately
            # Get array of bool value to indicate meas, noise electrode
            meas  = np.arange(data_X.shape[3]) <= 3    # Channels 0-3 are measuring electrode
            noise = np.arange(data_X.shape[3]) >  3    # Channels 4-7 are measuring electrode
            
            # Aggregate across electrode type
            meas  = agg(output[:,:,:,meas], axis=3, keepdims=True)
            noise = agg(output[:,:,:,noise], axis=3, keepdims=True)
            output = np.concatenate((meas, noise), axis=3)
            
    # Aggregate all class (if app)
    if all_class == True:
        output = agg(output, axis=1, keepdims=True)
    
    return output


def get_rms(X, axis=0, num_to_mean=None):
    """
    Function: Get the root mean square value of the input waveform X
    Argument: 
        X    : np.ndarray of shape (subject, trial, sample, channel)
        axis : int type. Denotes the axis to perform the mean on
        num_to_mean : int type, or None. 
            If None, just perform rms on axis. 
            If int, perform rms on num_to_mean samples denoted by axis. And for every num_to_mean / 2 samples onwards, perform same rms for every num_to_mean samples. i.e. 50% overlap
    Return:
        X_rms : np.ndarray of shape (subject, trial, sample, channel), where the value is the root mean square value of the samples. 
            For num_to_mean == None, the axis denoted by axis will have shape = 1
            For num_to_mean == int type, the axis denoted by axis will have shape  int(np.ceil(X.shape[axis] / num_to_mean * 2))-1
    """
    # Square
    X = np.square(X)
    
    # Mean
    if num_to_mean is None:
        X_rms = np.mean(X, axis=axis, keepdims=True)
        
    else:     # expect int value to num_to_mean
        assert type(num_to_mean) is int, "num_to_mean must be of int type"
        num_of_mean = int(np.ceil(X.shape[axis] / num_to_mean * 2))-1
        half_mean = int(np.floor(num_to_mean / 2))
        for i in range(num_of_mean):
            if i == 0:                # First set
                if axis == 0: 
                    X_rms = np.mean(X[0:num_to_mean], axis=axis, keepdims=True)
                elif axis == 1:
                    X_rms = np.mean(X[:,0:num_to_mean], axis=axis, keepdims=True)
                elif axis == 2:
                    X_rms = np.mean(X[:,:,0:num_to_mean], axis=axis, keepdims=True)
                else:
                    X_rms = np.mean(X[:,:,:,0:num_to_mean], axis=axis, keepdims=True)
            elif i == num_of_mean -1:   # Last set
                if axis == 0: 
                    X_rms = np.concatenate((X_rms,np.mean(X[i*half_mean:], axis=axis, keepdims=True)),axis=axis)
                elif axis == 1:
                    X_rms = np.concatenate((X_rms,np.mean(X[:,i*half_mean:], axis=axis, keepdims=True)),axis=axis)
                elif axis == 2:
                    X_rms = np.concatenate((X_rms,np.mean(X[:,:,i*half_mean:], axis=axis, keepdims=True)),axis=axis)
                else:
                    X_rms = np.concatenate((X_rms,np.mean(X[:,:,:,i*half_mean:], axis=axis, keepdims=True)),axis=axis)
            else: 
                if axis == 0: 
                    X_rms = np.concatenate((X_rms,np.mean(X[i*half_mean:i*half_mean + num_to_mean], axis=axis, keepdims=True)),axis=axis)
                elif axis == 1:
                    X_rms = np.concatenate((X_rms,np.mean(X[:,i*half_mean:i*half_mean + num_to_mean], axis=axis, keepdims=True)),axis=axis)
                elif axis == 2:
                    X_rms = np.concatenate((X_rms,np.mean(X[:,:,i*half_mean:i*half_mean + num_to_mean], axis=axis, keepdims=True)),axis=axis)
                else:
                    X_rms = np.concatenate((X_rms,np.mean(X[:,:,:,i*half_mean:i*half_mean + num_to_mean], axis=axis, keepdims=True)),axis=axis)
                
    # Root
    X_rms = np.sqrt(X_rms)
    
    return X_rms



# def apply_fft(data_X, spectrum_type='power', amp_scale='linear', sample_rate = 250):
#     """
#     Apply fft function on input data_X
#     Argument:
#         data_X: input data of type np.ndarray with shape (sample, channel)
#         spectrum_type: type str
#             'power': power spectrum (units = uV^2). Uses np.abs(yf)**2
#             'amplitude': amplitude spectrum (units = uV). Uses np.abs(yf)
#             'real': amplitude spectrum, taking the real values only (units = uV). Uses yf.real
#             'imag': amplitude spectrum, taking the imaginary values only (units = uV). Uses yf.imag
#             'angle': angular value of the complex number. Uses np.angle(yf)
#         output_scale: type str
#             'linear': output y-scale is linear
#             'log': output y-scale is 10*log(yf)
#         sample_rate: type int. The sample rate in Hz
#     Return:
#         xf: type np.ndarray of shape (num_bins,) each with resolution of bin_width
#         yf: type np.ndarray of shape (subject, trial, num_bins, channel), where each element is of type np.float64 (same as input data_X). where num_bins = sample // 2 if filter_bin[0] == False. Else, to be determined by filter_bin parameters
#     """
#     import numpy.fft

#     # FFT on y
#     sample_len = data_X.shape[0]
#     bin_width_hz = sample_rate / sample_len
#     yf = numpy.fft.fft(data_X, axis=0)

#     # Get real components of yf
#     if spectrum_type == 'power':
#         yf = 4.0/sample_len/sample_len*np.abs(yf)**2
#     elif spectrum_type == 'amplitude':
#         yf = 2.0/sample_len*np.abs(yf)
#     elif spectrum_type == 'real':
#         yf = 2.0/sample_len*yf.real
#     elif spectrum_type == 'imag':
#         yf = 2.0/sample_len*yf.imag
#     elif spectrum_type == 'angle':
#         yf = np.angle(yf)
#     else:
#         assert False, 'spectrum_type input is not valid'

#     # Get xf, yf output
#     xf = np.arange(sample_len // 2) * bin_width_hz
#     if amp_scale == 'linear':
#         yf = yf[:sample_len//2,:]
#     elif amp_scale == 'log':
#         yf = 10*np.log10(yf[:sample_len//2,:])
#     else:
#         assert False, 'amp_scale input is not valid'
    
#     return xf, yf

def apply_fft(data_X, spectrum_type='power', amp_scale='linear', filter_bin = [False, 250, 10, 60]):
    """
    Apply fft function on input data_X
    Argument:
        data_X: input data of type np.ndarray with shape (subject, trial, sample, channel)
        spectrum_type: type str
            'power': power spectrum (units = uV^2). Uses np.abs(yf)**2
            'amplitude': amplitude spectrum (units = uV). Uses np.abs(yf)
            'real': amplitude spectrum, taking the real values only (units = uV). Uses yf.real
            'imag': amplitude spectrum, taking the imaginary values only (units = uV). Uses yf.imag
            'angle': angular value of the complex number. Uses np.angle(yf)
        output_scale: type str
            'linear': output y-scale is linear
            'log': output y-scale is 10*log(yf)
        filter_bin: type list. To filter the output bins if filter_bin[0] == True
            filter_bin[0]: type bool. If False, nothing. If True, filter the output xf and yf according to filter_bin[1:4]
            filter_bin[1]: type int. Sample Frequency 
            filter_bin[2]: type int. Lowest frequency to include in output
            filter_bin[3]: type int. Highest frequency to include in output
    Return:
        xf: type np.ndarray of shape (num_bins,) each with resolution of bin_width
        yf: type np.ndarray of shape (subject, trial, num_bins, channel), where each element is of type np.float64 (same as input data_X). where num_bins = sample // 2 if filter_bin[0] == False. Else, to be determined by filter_bin parameters
    """
    import numpy.fft

    # FFT on y
    sample_len = data_X.shape[2]
    sample_rate = filter_bin[1]
    bin_width_hz = sample_rate / sample_len
    yf = numpy.fft.fft(data_X, axis=2)

    # Get real components of yf
    if spectrum_type == 'power':
        yf = 4.0/sample_len/sample_len*np.abs(yf)**2
    elif spectrum_type == 'amplitude':
        yf = 2.0/sample_len*np.abs(yf)
    elif spectrum_type == 'real':
        yf = 2.0/sample_len*yf.real
    elif spectrum_type == 'imag':
        yf = 2.0/sample_len*yf.imag
    elif spectrum_type == 'angle':
        yf = np.angle(yf)
    else:
        assert False, 'spectrum_type input is not valid'

    # Get xf, yf output
    xf = np.arange(sample_len // 2) * bin_width_hz
    if amp_scale == 'linear':
        yf = yf[:,:,:sample_len//2,:]
    elif amp_scale == 'log':
        yf = 10*np.log10(yf[:,:,:sample_len//2,:])
    else:
        assert False, 'amp_scale input is not valid'


    # Filter by bin
    if filter_bin[0] == False:
        pass
    elif filter_bin[0] == True:   # Perform bin filtering
        lowest_bin = int(filter_bin[2] // bin_width_hz)
        highest_bin = int(filter_bin[3] // bin_width_hz)
        xf = xf[lowest_bin:highest_bin]
        yf = yf[:,:,lowest_bin:highest_bin,:]
    else: 
        assert False, 'filter_bin[0] input is not valid'
    
    return xf, yf

def plot_fft(xf, yf, xlim=[0,15], ylim=[0,10], xlabel='Frequency (Hz)', ylabel='Amplitude (uV)', mode='Average', channel=[0,1,2,4,5,6]):
    """
    Function to plot the spectrum of the EEG data, with aggregation function and plotting conditioning (if required)
    Argument:
        xf: type np.ndarray of shape (num_bins, )
        yf: type np.ndarray of shape (num_bins, num of channels), 
            where each element is of type np.float64 (same as input data_X), 
            where num_bins = sample // 2 if filter_bin[0] == False. Else, to be determined by filter_bin parameters
        xlim : type list. Gives the lower and upper limit for x plotting
        ylim : type list. Gives the lower and upper limit for y plotting
        mode : type str. Accepted input are 
            "Average": Function will aggregate (np.mean) the amplitude of the channels (see next field)
            "Single": Function will only extract the amplitude of the channel (i.e. one channel at channel[0]) 
        channel : type list. Gives the list of channels to use for aggregate, or for single channel, len(channel) == 1
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # If average the channels
    if mode == 'Average':
        yf = yf[:,channel]   # Get only the channels indicated
        yf = np.mean(yf,axis=1, keepdims=True)
        print(yf.shape)
    elif mode == 'Single':
        assert len(channel) == 1, 'Only 1 channel should be provided for "Single" channel mode'
        yf = yf[:,channel[0]]
        print(yf.shape)
    
    # Plot and condition the plot
    plt.plot(xf, yf)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.show()

def get_avg_fft_bins(xf, yf, num_of_bins_to_combine=1, agg='mean'):
    """
    Function to combine num_of_bins_to_combine fft bins to smoothen out the FFT plot.
    Argument:
        xf: type np.ndarray of shape (num_samples,). Gives the frequency bins of input
        yf: type np.ndarray of shape (num_samples,). Gives the amplitude of each frequency bin
        num_of_bins_to_combine: type int. If 1, output = input. Else combine the respective number of bins and output the xf and yf accordingly.
        agg: type str. If mean, use mean; if max, use max.
    Return:
        xf_out: type np.ndarray of shape (num_samples,)
        yf_out: type np.ndarray of shape (num_samples,)
    """
    # Assert xf, yf same length
    assert len(xf) == len(yf), "xf and yf should have same length!"
    
    # Find remainder to get rid of tail end of data that cannot fit into multiples of num_of_bins_to_combine
    remainder = len(xf) % num_of_bins_to_combine
    if remainder:
        xf = xf[:-remainder]
        yf = yf[:-remainder]
        
    # Reshape the array and compute the average of each num_of_bins_to_combine samples
    if agg=='mean':
        xf_out = xf.reshape(-1, num_of_bins_to_combine).min(axis=1)
        yf_out = yf.reshape(-1, num_of_bins_to_combine).mean(axis=1)
    elif agg == 'max':
        xf_out = xf.reshape(-1, num_of_bins_to_combine).min(axis=1)
        yf_out = yf.reshape(-1, num_of_bins_to_combine).max(axis=1)
    
    return xf_out, yf_out