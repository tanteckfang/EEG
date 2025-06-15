import os
import numpy as np

def import_text_file(file_dir_name, row_start=5, header_row=4):
    """
    Function to import the openbci text file. Function prints out the header files
    Argument:
        file_dir_name: type str. Full directory and name of the file, with .txt extension to be read
        row_start: type int. Indicate the row index to start collecting the data
        header_row: type int. Indicate the row index of the header
    Return
        text_array: type numpy.ndarray of shape (number of rows, number of columns). 
                    Entries are of str type. 
    """
    # Import text file
    with open(file_dir_name) as f:
        lines = f.readlines()

    # Condition text file into numpy array
    header = lines[header_row].split(',')
    # print('Header of files are as follow:')
    # print(header)

    text_data = []
    
    for i in range(row_start, len(lines)):  # Go through each line and split by comma, put to list
        text_data.append(lines[i].split(','))
    
    text_array = np.array(text_data)
    return text_array


def filter_eeg_columns(text_array, eeg_start_column=1, eeg_end_column=9, expand_dims_trial=True, expand_dims_subject=True):
    """
    Function to filter only the EEG columns of the text array.
    Argument:
        text_array: type numpy.ndarray of shape (number of rows, number of columns). Entries are of str type.
        eeg_start_column: type int. denote the start column for EEG data
        eeg_end_column: type int. denote the end column for EEG data
        expand_dims_trial: type bool. If True, expand dimension from (sample, channel) to (trial, sample, channel)
        expand_dims_subject: type bool. If True, expand dimension to also add subject
    Return:
        eeg_array: type numpy.ndarray of shape (subject, trial, samples, channel),
                if expand both dims. Else change accordingly
                Entries are of float type.
    """
    # Filter for the appropriate columns, and change to float type
    eeg_array = text_array[:,eeg_start_column:eeg_end_column].astype(float)
    
    # Expand dimension if required
    if expand_dims_trial:
        eeg_array = np.expand_dims(eeg_array, axis=0)
    if expand_dims_subject:
        eeg_array = np.expand_dims(eeg_array, axis=0)

    return eeg_array
    

def stack_by_axis(array_1, array_2, axis_to_stack=3):
    """
    Function to stack the 2 input along the axis indicated. Dimension of all samples axis will be the minimum of the 2 arrays.
    Argument:
        array_1, array_2: type numpy.ndarray of shape (subject, trial, samples, channels)
        axis_to_stack: type int. The axis on which to stack the 2 arrays
    Return:
        eeg_array: type numpy.ndarray after stacking along the indicated axis. (subject, trial, samples, channels)
                i.e. if axis_to_stack=3, channels will double. samples will be minimum of array_1, array_2.
    """
    # Concatenate the left and right side to shortest length
    combined_len = min(array_1.shape[2], array_2.shape[2])

    # Concatenate along axis_to_stack axis with trimming of samples axis.
    eeg_array = np.concatenate((array_1[:,:,:combined_len,:], array_2[:,:,:combined_len,:]), axis=axis_to_stack)

    return eeg_array