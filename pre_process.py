"""
Source: https://github.com/robintibor/braindecode/blob/master/braindecode/datautil/signalproc.py
Signal processing block for EEG signal
"""
## Using IPython.core.debugger.set_trace for debug
# from IPython.core.debugger import set_trace
import logging
log = logging.getLogger(__name__)

import pandas as pd
import numpy as np
import scipy
import scipy.signal
from sklearn.model_selection import train_test_split



def exponential_running_standardize(
    data, factor_new=0.001, init_block_size=None, eps=1e-6
):
    """
    Perform exponential running standardization, using pandas DataFrame.ewm function, with adjust=False
    
    Compute the exponental running mean :math:`m_t` at time `t` as 
    :math:`m_t=\mathrm{factornew} \cdot mean(x_t) + (1 - \mathrm{factornew}) \cdot m_{t-1}`.
    
    Then, compute exponential running variance :math:`v_t` at time `t` as 
    :math:`v_t=\mathrm{factornew} \cdot (m_t - x_t)^2 + (1 - \mathrm{factornew}) \cdot v_{t-1}`.
    
    Finally, standardize the data point :math:`x_t` at time `t` as:
    :math:`x'_t=(x_t - m_t) / max(\sqrt{v_t}, eps)`.
    
    
    Parameters
    ----------
    data: 2darray (time, channels)
    factor_new: float
    init_block_size: int
        Standardize data before to this index with regular standardization. 
    eps: float
        Stabilizer for division by zero variance.
    Returns
    -------
    standardized: 2darray (time, channels)
        Standardized data.
    """
    df = pd.DataFrame(data)
    meaned = df.ewm(alpha=factor_new, adjust=False).mean()   # GTL added adjust=False
    demeaned = df - meaned
    squared = demeaned * demeaned
    square_ewmed = squared.ewm(alpha=factor_new, adjust=False).mean()   # GTL added adjust=False
    standardized = demeaned / np.maximum(eps, np.sqrt(np.array(square_ewmed)))
    standardized = np.array(standardized)
    if init_block_size is not None:
        other_axis = tuple(range(1, len(data.shape)))
        init_mean = np.mean(
            data[0:init_block_size], axis=other_axis, keepdims=True
        )
        init_std = np.std(
            data[0:init_block_size], axis=other_axis, keepdims=True
        )
        init_block_standardized = (
            data[0:init_block_size] - init_mean
        ) / np.maximum(eps, init_std)
        standardized[0:init_block_size] = init_block_standardized
    return standardized


def exponential_running_demean(data, factor_new=0.001, init_block_size=None):
    """
    Perform exponential running demeanining. 
    Compute the exponental running mean :math:`m_t` at time `t` as 
    :math:`m_t=\mathrm{factornew} \cdot mean(x_t) + (1 - \mathrm{factornew}) \cdot m_{t-1}`.
    Deman the data point :math:`x_t` at time `t` as:
    :math:`x'_t=(x_t - m_t)`.
    Parameters
    ----------
    data: 2darray (time, channels)
    factor_new: float
    init_block_size: int
        Demean data before to this index with regular demeaning. 
        
    Returns
    -------
    demeaned: 2darray (time, channels)
        Demeaned data.
    """
    df = pd.DataFrame(data)
    meaned = df.ewm(alpha=factor_new).mean()
    demeaned = df - meaned
    demeaned = np.array(demeaned)
    if init_block_size is not None:
        other_axis = tuple(range(1, len(data.shape)))
        init_mean = np.mean(
            data[0:init_block_size], axis=other_axis, keepdims=True
        )
        demeaned[0:init_block_size] = data[0:init_block_size] - init_mean
    return demeaned


def highpass_cnt(data, low_cut_hz, fs, filt_order=3, axis=0):
    """
     Highpass signal applying **causal** butterworth filter of given order.
    Parameters
    ----------
    data: 2d-array
        Time x channels
    low_cut_hz: float
    fs: float
    filt_order: int
    Returns
    -------
    highpassed_data: 2d-array
        Data after applying highpass filter.
    """
    if (low_cut_hz is None) or (low_cut_hz == 0):
        log.info("Not doing any highpass, since low 0 or None")
        return data.copy()
    b, a = scipy.signal.butter(
        filt_order, low_cut_hz / (fs / 2.0), btype="highpass"
    )
    assert filter_is_stable(a)
    data_highpassed = scipy.signal.lfilter(b, a, data, axis=axis)
    return data_highpassed


def lowpass_cnt(data, high_cut_hz, fs, filt_order=3, axis=0):
    """
     Lowpass signal applying **causal** butterworth filter of given order.
    Parameters
    ----------
    data: 2d-array
        Time x channels
    high_cut_hz: float
    fs: float
    filt_order: int
    Returns
    -------
    lowpassed_data: 2d-array
        Data after applying lowpass filter.
    """
    if (high_cut_hz is None) or (high_cut_hz == fs / 2.0):
        log.info(
            "Not doing any lowpass, since high cut hz is None or nyquist freq."
        )
        return data.copy()
    b, a = scipy.signal.butter(
        filt_order, high_cut_hz / (fs / 2.0), btype="lowpass"
    )
    assert filter_is_stable(a)
    data_lowpassed = scipy.signal.lfilter(b, a, data, axis=axis)
    return data_lowpassed


def bandpass_cnt(
    data, low_cut_hz, high_cut_hz, fs, filt_order=3, axis=0, filtfilt=False
):
    """
     Bandpass signal applying **causal** butterworth filter of given order.
    Parameters
    ----------
    data: 2d-array
        Time x channels
    low_cut_hz: float
    high_cut_hz: float
    fs: float
    filt_order: int
    filtfilt: bool
        Whether to use filtfilt instead of lfilter
    Returns
    -------
    bandpassed_data: 2d-array
        Data after applying bandpass filter.
    """
    
    if (low_cut_hz == 0 or low_cut_hz is None) and (
        high_cut_hz == None or high_cut_hz == fs / 2.0
    ):
        log.info(
            "Not doing any bandpass, since low 0 or None and "
            "high None or nyquist frequency"
        )
        return data.copy()
    if low_cut_hz == 0 or low_cut_hz == None:
        log.info("Using lowpass filter since low cut hz is 0 or None")
        return lowpass_cnt(
            data, high_cut_hz, fs, filt_order=filt_order, axis=axis
        )
    if high_cut_hz == None or high_cut_hz == (fs / 2.0):
        log.info(
            "Using highpass filter since high cut hz is None or nyquist freq"
        )
        return highpass_cnt(
            data, low_cut_hz, fs, filt_order=filt_order, axis=axis
        )

    nyq_freq = 0.5 * fs
    low = low_cut_hz / nyq_freq
    high = high_cut_hz / nyq_freq
    b, a = scipy.signal.butter(filt_order, [low, high], btype="bandpass", output="ba")
    assert filter_is_stable(a), "Filter should be stable..."
    if filtfilt:
        data_bandpassed = scipy.signal.filtfilt(b, a, data, axis=axis)
    else:
        data_bandpassed = scipy.signal.lfilter(b, a, data, axis=axis)
    return data_bandpassed


def filter_is_stable(a):
    """
    Check if filter coefficients of IIR filter are stable.
    
    Parameters
    ----------
    a: list or 1darray of number
        Denominator filter coefficients a.
    Returns
    -------
    is_stable: bool
        Filter is stable or not.  
    Notes
    ----
    Filter is stable if absolute value of all  roots is smaller than 1,
    see [1]_.
    
    References
    ----------
    .. [1] HYRY, "SciPy 'lfilter' returns only NaNs" StackOverflow,
       http://stackoverflow.com/a/8812737/1469195
    """
    assert a[0] == 1.0, (
        "a[0] should normally be zero, did you accidentally supply b?\n"
        "a: {:s}".format(str(a))
    )
    # from http://stackoverflow.com/a/8812737/1469195
    return np.all(np.abs(np.roots(a)) < 1)

def pre_process_dataset(dataset_X, dataset_Y, is_train=True, val_pct=0.25, sig_filter=['BPF', 4, 38, 250], normalize=[True, 1e-3], augment_sample = [False, 276, 4, 80], req_unshuffle=False, downsample_factor=1):
    """
    Function: 
    Performs selected pre_processing on the input dataset_X. Pre-processing is performed in the order: "sig_filter" -> "normalize". Note pre-processing is performed inplace (i.e. overwrite the input data)
          
    Argument:
    dataset_X: EEG data, numpy.ndarray with dimension (subject_num, trial_num, sample_num, channel_num)
    dataset_Y: EEG label, numpy.ndarray with dimension (subject_num, trial_num, 1)
    is_train: type bool. True if input is for training+validation, False if input for inference only
        if True, pre_process will include all steps
        if False, pre_process will EXCLUDE train_test_split. 
            augment_by_sample will be simplified to truncating dataset_X to the sample length required only (for fitting to model)
    val_pct: type float of value between 0 and 1. The percentage of validation data if dataset needs to be split into train and validation set (i.e. is_train == True)
    sig_filter: Parameters of the filter operation on the input dataset_X. List type
        [0]: Filter Type - Acceptable parameters: 'None', 'LPF', 'HPF', 'BPF'
        [1]: Lower Freq (Hz) - Applicable for 'LPF' and 'BPF' Filter Type
        [2]: Upper Freq (Hz) - Applicable for 'HPF' and 'BPF' Filter Type
        [3]: Sample Rate (Hz) - Sample rate of the input dataset_X
    normalize: Parameters of the normalization operation. List type
        [0]: Requires Normalization - boolean type
        [1]: factor_new parameter of exponential_running_standardize 
    augment_sample: Parameters of the augment_sample. List type
        [0]: type bool. if True, perform augment_sample, else feed-through
        [1]: type int. num_sample. the length of each sample if augment_sample is performed
        [2]: type int. offset_res. the resolution for each offset step
        [3]: type int. final_offset. the final offset value
    req_unshuffle: type bool. Denote if need to unshuffle train set (Note: validation set not unshuffled) to classes 0,1,2,3,0,1,2,3,...
    downsample_factor: type int. Amount of downsampling. e.g. if 2, sampling rate will be halved
        
    Return:
         X_train: EEG data after preprocess, numpy.ndarray with dimension (subject_num, trial_num, sample_num, channel_num)
         Y_train: EEG label after preprocess, numpy.ndarray with dimension (subject_num, trial_num, 1)
         X_val: if is_train=True, the validation set for X, same shape as X_train; else return None
         Y_val: if is_train=True, the validation set for Y, same shape as Y_train; else return None
    """
    log.debug('running pre_process_dataset')

    
    # Check correctness of input
    assert_pre_process_dataset(dataset_X, sig_filter, normalize)
        
    print('====================================================================================================')
    print('Performing pre-process for input dataset')
    print('Input shape data.X = {}, data.Y = {}'.format(dataset_X.shape, dataset_Y.shape))
    
    # Perform filtering on dataset_X
    if sig_filter[0] == 'None':
        print('== Pre-Process == => No Software Spectrum Filtering Performed')
    elif (sig_filter[0] == 'LPF') or (sig_filter[0] == 'HPF') or (sig_filter[0] == 'BPF'):
        print('== Pre-Process == => Performing Software {} on input EEG signal'.format(sig_filter[0]))
        for i in range(dataset_X.shape[0]):       # For each subject_num
            for j in range(dataset_X.shape[1]):   # For each trial_num
                dataset_X[i][j] = bandpass_cnt(dataset_X[i][j], sig_filter[1], sig_filter[2], sig_filter[3])
    else:
        assert False, 'reached invalid state in filter stage of pre_process_dataset'
          
            
    # Perform normalization on dataset_X
    if normalize[0] is False:
        print('== Pre-Process == => No normalization performed')
    elif normalize[0] is True:
        print('== Pre-Process == => Performing normalization on input EEG signal')
        for i in range(dataset_X.shape[0]):       # For each subject_num
            for j in range(dataset_X.shape[1]):   # For each trial_num
                dataset_X[i][j] = exponential_running_standardize(dataset_X[i][j], normalize[1])
    else:
        assert False, 'reached invalid state in normalize stage of pre_process_dataset'
    
    
    # Split to train-validation set
    if is_train == True:
        print('== Pre-Process == => Performing train-val split on dataset')
        X_train, Y_train, X_val, Y_val = train_val_split(dataset_X, dataset_Y, val_pct)
        print('dataset_X.shape={}, dataset_Y.shape={}'.format(dataset_X.shape, dataset_Y.shape))
        print('X_train.shape={}, X_val.shape={}'.format(X_train.shape, X_val.shape))
    else:
        print('== Pre-Process == => No train-val split performed')
        X_train = np.copy(dataset_X)
        Y_train = np.copy(dataset_Y)
        X_val, Y_val = None, None
        
    # Perform data augmentation by sample offset (i.e. augment_by_sample)
    if is_train == True:
        print('== Pre-Process == => Performing sample augmentation on dataset')
        X_train, Y_train = augment_by_sample(X_train, Y_train, req_augment=augment_sample[0], num_sample=augment_sample[1], offset_res=augment_sample[2], final_offset=augment_sample[3])
        X_val, Y_val = augment_by_sample(X_val, Y_val, req_augment=augment_sample[0], num_sample=augment_sample[1], offset_res=augment_sample[2], final_offset=augment_sample[3])
        
    else:
        if augment_sample[0] == True:  # Augment sample True for test set. Truncate test set to num_sample only, so can fit model
            print('== Pre-Process == => No sample augmentation performed, but truncating dataset to length of augment_sample[1]')
            X_train = X_train[:,:,:augment_sample[1]]
        else:  # Otherwise no need to do anything
            print('== Pre-Process == => No sample augmentation performed')
    
    
    # Perform unshuffle if required.
    if req_unshuffle == True:
        X_train, Y_train = unshuffle(X_train, Y_train)
        print('== Pre-Process == => Input X_train and Y_train has been unshuffled to facilitate aggregate by class')

        
    # Perform down_sample
    X_train = down_sample(X_train, downsample_factor)
    if is_train == True:
        X_val = down_sample(X_val, downsample_factor)
    print('== Pre-Process == => Downsample with factor = {}'.format(downsample_factor))
    

    if is_train == True:
        print('Output shape X_train={}, Y_train={}, X_val={}, Y_val={}'.format(X_train.shape, Y_train.shape, X_val.shape, Y_val.shape))
    else:
        print('Output shape X_test={}, Y_test={}'.format(X_train.shape, Y_train.shape))
        print('====================================================================================================\n')
    
    return X_train, Y_train, X_val, Y_val


def assert_pre_process_dataset(dataset_X, sig_filter, normalize):
    # Check dataset_X
    assert type(dataset_X) is np.ndarray, 'Input dataset_X must be of type numpy.ndarry'
    assert len(dataset_X.shape) == 4, 'Input dataset_X must be of shape (subject_num, trial_num, sample_num, channel_num)'
    
    # Check sig_filter
    assert type(sig_filter) is list, 'sig_filter must be of type list'
    assert len(sig_filter) == 4, 'sig_filter must be of length 4'
    assert (sig_filter[0] == 'None') or (sig_filter[0] == 'LPF') or (sig_filter[0] == 'HPF') or (sig_filter[0] == 'BPF'), ' sig_filter[0] only accepts arguments "None", "LPF", "HPF" or "BPF"'
    assert (type(sig_filter[1]) is float) or (type(sig_filter[1]) is int), 'sig_filter[1] must be of int or float type'
    assert (type(sig_filter[2]) is float) or (type(sig_filter[2]) is int), 'sig_filter[2] must be of int or float type'
    assert (type(sig_filter[3]) is float) or (type(sig_filter[3]) is int), 'sig_filter[3] must be of int or float type'
    
    # Check normalize
    assert type(normalize) is list, 'normalize must be of type list'
    assert len(normalize) == 2, 'normalize must be of length 2'
    assert type(normalize[0]) is bool, 'normalize[0] must be of bool type'
    assert (type(normalize[1]) is float) or (type(normalize[1]) is int), 'normalize[1] must be of int or float type'
    

    
def train_val_split(X, Y, val_pct):
    for i in range(X.shape[0]): # for each subject_num
        # Use train_test_split from sklearn with stratify so that all classes are present in both train and test set
        # (i.e. not have the case where any class is not present in test set)
        X_train_temp, X_val_temp, Y_train_temp, Y_val_temp = train_test_split(X[i], Y[i], test_size=val_pct, stratify=Y[i])
        if i == 0:
            X_train = np.expand_dims(X_train_temp, axis=0)
            X_val = np.expand_dims(X_val_temp, axis=0)
            Y_train = np.expand_dims(Y_train_temp, axis=0)
            Y_val = np.expand_dims(Y_val_temp, axis=0)
        else:
            X_train = np.concatenate((X_train, np.expand_dims(X_train_temp, axis=0)), axis=0)
            X_val = np.concatenate((X_val, np.expand_dims(X_val_temp, axis=0)), axis=0)
            Y_train = np.concatenate((Y_train, np.expand_dims(Y_train_temp, axis=0)), axis=0)
            Y_val = np.concatenate((Y_val, np.expand_dims(Y_val_temp, axis=0)), axis=0)
    
    log.info('Train and validation set prepared for training. With shapes below:')
    log.info('X_train = {}, Y_train = {}, X_val = {}, Y_val = {}'.format(X_train.shape, Y_train.shape, X_val.shape, Y_val.shape))
    
    # Following code to test above code if required
    #     subj_num = 5
    #     class_num = 3
    #     start_sample, end_sample = -10, -1
    #     ch = 6

    #     print(Y.shape, Y_train.shape, Y_val.shape)
    #     print(X.shape, X_train.shape, X_val.shape, '\n')

    #     # Test 
    #     index_y_test = np.arange(40)[Y_val[subj_num, :,0]==class_num]
    #     print('index of class {} in Y_val = {}'.format(class_num, index_y_test))
    #     print(X_val[subj_num,index_y_test,start_sample:end_sample,ch],'\n')

    #     # Train 
    #     index_y_train = np.arange(120)[Y_train[subj_num,:,0]==class_num]
    #     print('index of class {} in Y_train = {}'.format(class_num, index_y_train))
    #     print(X_train[subj_num,index_y_train,start_sample:end_sample,ch],'\n')

    #     # Before Split
    #     index_y = np.arange(160)[Y[subj_num, :,0] == class_num]
    #     print('index of class {} in Y = {}'.format(class_num, index_y))
    #     print(X[subj_num, index_y, start_sample:end_sample, ch],'\n')

    return X_train, Y_train, X_val, Y_val

    
def filter_by_sample(data_X, data_Y, filter_type='none', value=None, length=276):
    """
    Function to filter the dataset by sample
    Argument:
        data_X: input data with shape (subject_num, trial_num, sample_num, channel_num)
        data_Y: input data with shape (subject_num, trial_num, 1).
        filter_type: type string with values:
            'fix_length' : Fix length of 276 samples. 
            'none': direct feed-through of data_X and data_Y to output
        value: depends on value of filter_type:
            'fix_length' : type int, where value denotes the fix sample to offset for all, keeping length of entire sample to 'length'
        length: type int. Size of number of samples of output_X
    Return:
        output_X: output data with shape (subject_num, trial_num, sample_num, channel_num), where sample_num = length
        output_Y: output data with shape (subject_num, trial_num, 1). Same as input data_Y
    """
    import numpy as np
    output_Y = np.copy(data_Y)
    
    if filter_type == 'none':
        output_X = np.copy(data_X)
        
    elif filter_type == 'fix_length':
        assert type(value) is int, 'value must be of int type when filter_type == "fix_length" '
        assert type(length) is int, 'length must be of int type'
        assert value >= 0, 'value must be positive'
        assert length <= data_X.shape[2], 'final length cannot be longer than initial length'
        output_X = data_X[:,:,value:value+length]
        
    return output_X, output_Y


def augment_by_sample(data_X, data_Y, req_augment=True, num_sample=276, offset_res=4, final_offset=80):
    """
    Function to augment the sample by recursively calling the filter_by_sample function. Offset the sample data_X to fixed sample length = num_samples. 
    Argument:
        data_X: input data with shape (subject_num, trial_num, sample_num, channel_num)
        data_Y: input data with shape (subject_num, trial_num, 1).
        req_augment: type bool. If False, relay input data_X and data_Y to output, else, perform augmentation
        num_sample: type int. Size of number of samples of output_X
        offset_res: type int. Resolution of the increase in offset sample number
        final_offset: type int. Final value in the offset sample number
    Return:
        output_X: output data with shape (subject_num, trial_num, sample_num, channel_num), where sample_num = num_sample
        output_Y: output data with shape (subject_num, trial_num, 1). Same as input data_Y
        Note that trial_num for both output_X and output_Y will be increased in multiples of the input
    """
    assert type(req_augment) is bool, 'req_augment must be of boolean type'
    assert type(num_sample) is int, 'num_sample must be of int type'
    assert type(offset_res) is int, 'offset_res must be of int type'
    assert type(final_offset) is int, 'final_offset must be of int type'
    assert final_offset <= data_X.shape[2], 'final_offset cannot be longer than initial length'
    
    if req_augment is True:
        output_X, output_Y = filter_by_sample(data_X, data_Y, 'fix_length', 0, num_sample)
        for i in range(offset_res, final_offset, offset_res):
            dataX1, dataY1 = filter_by_sample(data_X, data_Y, 'fix_length', i, num_sample)
            output_X = np.concatenate((output_X, dataX1), axis=1)
            output_Y = np.concatenate((output_Y, dataY1), axis=1)
    else:
        output_X = np.copy(data_X)
        output_Y = np.copy(data_Y)
        
    return output_X, output_Y


def apply_window(data_X, win_type='hanning', beta=14):
    """
    Function to apply window function on the data.
    Argument:
        data_X: input data of type np.ndarray with shape (subject, trial, sample, channel)
        win_type: type str. Type of window function from numpy. It can be one of following:
            'hanning': Hanning window
            'hamming': Hamming window
            'blackman': Blackman window
            'kaiser': Kaiser window with beta = 14, unless otherwise provided in beta
        beta: type int. beta value for kaiser window
    Return:
        output: output data of type np.ndarray with shape (subject, trial, sample, channel), where the shape of output = shape of input data_X
    """
    import numpy as np
    window_size = data_X.shape[2]
    if win_type == 'hanning':
        window = np.hamming(window_size)
    elif win_type == 'hamming':
        window = np.hamming(window_size)
    elif win_type == 'blackman':
        window = np.blackman(window_size)
    elif win_type == 'kaiser':
        window = np.kaiser(window_size, beta)
    else:
        assert False, 'win_type input is not valid'
    
    output = data_X.transpose(0,1,3,2)*window   # Transpose before * so broadcasting occur
    output = output.transpose(0,1,3,2)          # Then transpose back to (subject,trial,sample,channel)

    return output


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


def unshuffle(X_train, Y_train):
    """
    Function to unshuffle the shuffled data so that can perform FFT plot by class aggregation. Y_train will be used to sort. Note that X_train.shape[1] and Y_train.shape[1] must both be divisible by 4. Else cannot. 
    Argument: 
        X_train: np.ndarray of shape (subject, trial, sample, channel)
        Y_train: np.ndarray of shape (subject, trial, 1)
    Return:
        X_train: np.ndarray of shape (subject, trial, sample, channel) (sorted)
        Y_train: np.ndarray of shape (subject, trial, 1) (sorted)
            i.e. Y_train[0,:8,0] will be [0,1,2,3,0,1,2,3]
    """
    assert X_train.shape[1] % 4 == 0, 'X_train.shape[1] must be divisible by 4'
    assert Y_train.shape[1] % 4 == 0, 'Y_train.shape[1] must be divisible by 4'
    for i in range(X_train.shape[0]):   # For each subject, do unshuffle

        X_copy = np.copy(X_train[i])
        Y_copy = np.copy(Y_train[i])

        # Get index based on ascending order, then reshape to interleave classes to 0,1,2,3,0,1,...
        index = np.argsort(Y_copy, 0)
        index = np.reshape(index, (4,Y_copy.shape[0]//4))  # By Default, order='C'
        index = np.reshape(index, -1, order='F')

        # Using sorted index to sort the data
        Y_sorted = np.take(Y_copy, index, axis=0)
        X_sorted = np.take(X_copy, index, axis=0)

        X_train[i] = X_sorted
        Y_train[i] = Y_sorted
    return X_train, Y_train


def down_sample(X_train, factor):
    """
    Function to downsample the data by the factor denoted by factor
    Argument: 
        X_train: np.ndarray of shape (subject, trial, sample, channel)
    Return:
        X_train_red : np.ndarray of shape (subject, trial, sample_down, channel), where sample_down == sample//factor
    """
    X_train_red = np.take(X_train, list(range(0,X_train.shape[2],factor)), axis=2)
    return X_train_red
    
    