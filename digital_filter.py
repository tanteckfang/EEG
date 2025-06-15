from brainflow.data_filter import DataFilter, AggOperations, WindowOperations, WaveletTypes, FilterTypes, NoiseTypes
import numpy as np

def perform_filter(data_X, band_pass = [True, 5.8, 40, 4], notch_filter = True, sampling_freq = 250):
    """
    Function: Perform filtering for the EEG signal.
    Argument:
        data_X: type np.ndarry of shape (subject, trial, samples, channels). EEG Data
        band_pass: type list
            band_pass[0]: type bool. Denote if band_pass filter is on or off
            band_pass[1]: type float. low_freq of band_pass filter
            band_pass[2]: type float. high_freq of band_pass filter
            band_pass[3]: type int. Order of the band_pass filter
        notch_filter: type bool. If True, turn on 50Hz notch filter, else off
        sampling_freq: type int. Sampling rate of the ADC
    Return:
        output: type np.ndarry of shape (subject, trial, samples, channels) after filtering
    """
    total_trials = data_X.shape[1]
    total_channels = data_X.shape[3]
    
    # Copy the numpy array as new array after transpose (so that C_CONTIGUOUS = False)
    output_reshaped = data_X.copy()
    output_reshaped = np.transpose(output_reshaped, (0,1,3,2)).copy()

    # loop through the trials to perform BPF
    low_freq, high_freq, filter_order = band_pass[1], band_pass[2], band_pass[3]

    for trial_num in range(total_trials):
        for channel in range(total_channels):
            if band_pass[0]: # Bandpass filter
                DataFilter.perform_bandpass(output_reshaped[0, trial_num, channel], 
                                            sampling_rate=sampling_freq, 
                                            start_freq = low_freq, 
                                            stop_freq  = high_freq, 
                                            order = filter_order,  
                                            filter_type = FilterTypes.BESSEL.value, 
                                            ripple = 0)

            if notch_filter:  # Notch Filter
                DataFilter.remove_environmental_noise(output_reshaped[0, trial_num, channel], 
                                                      sampling_rate = sampling_freq,
                                                      noise_type = NoiseTypes.FIFTY.value)

    # Reshape output_reshaped back to shape acceptable by own FFT algorithm
    output = np.transpose(output_reshaped, (0,1,3,2))
    
    return output