from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets
import time
import datetime
import numpy as np
import os


"""
Example from https://www.digitalocean.com/community/tutorials/python-multiprocessing-example But must have code in .py file, then use terminal to run, else no printout
"""

class EegInterface:
    def __init__(self, serial_port: int):
        """
        Function: Initialize a BoardShim object to control the openBCI Cyton board.
        Argument:
            serial_port: type int. Provide the COM port of openBCI Cyton board. Input value of x from: "Device Manager > Ports (COM and LPT) > USB Serial Port (COM x)""
        """
        self.serial_port = 'COM' + str(serial_port)  # Serial port in str format
        self.buffer_time = 1    # Buffer duration for each get_recording (in sec)
        
        # BoardShim params setup, and generate BoardShim object
        BoardShim.enable_dev_board_logger()
        params = BrainFlowInputParams()
        params.ip_port = 0
        params.serial_port = self.serial_port
        params.mac_address = ''
        params.other_info = ''
        params.serial_number = ''
        params.ip_address = ''
        params.ip_protocol = 0
        params.timeout = 0
        params.file = ''
        params.master_board = 0   # i.e. BoardIds.CYTON_BOARD
        params.preset = BrainFlowPresets.DEFAULT_PRESET
        self.params = params
        self.board = BoardShim(BoardIds.CYTON_BOARD, params)
        
        # Extract BoardShim object information
        board_id = BoardIds.CYTON_BOARD.value
        self.board_id = board_id
        self.sample_rate = self.board.get_sampling_rate(self.board_id)
        self.version = self.board.get_version()
        self.device_name = self.board.get_device_name(self.board_id)
        
        
    def prepare_board(self, ):
        """
        Function to prepare board for configuration and/or reading
        """
        self.board.prepare_session()
        
        
    def config_board(self, config_str : str ):
        """
        Function to configure board based on updated channel settings, if required. Else use default settings.
        Argument:
            config_str: type str. string to configure openBCI Cyton board. See https://docs.openbci.com/Cyton/CytonSDK/     
        """
        self.board.config_board(config_str)
        
    def start_stream(self, buffer_size=4096):
        """
        Function to setup a ring buffer to stream EEG data into
        Argument:
            buffer_size: type int. Set size of buffer. buffer_size / 250 = size of buffer in seconds. Default to 4096 ~ 16s
        """
        self.board.start_stream(buffer_size)   # Set buffer size to record
        
        
    def get_recording(self, rec_duration_s: float, has_accelerometer=False, include_timestamp=False):
        """
        Function to collect rec_duration_s amount of recording.
        Argument: 
            rec_duration_s: type float. Amount of EEG recording (in seconds) to capture.
            has_accelerometer: type bool. If True, include accelerometer data, else only include EEG data.
            include_timestamp: type bool. If True, include the timestamp index data at channel 0.
        Return: 
            data: type numpy.ndarray of shape (num_channels, num_recording), where num_channels = 8 if has_accelerometer = False, else 12. num_recording = self.sample_rate x rec_duration_s. Add 1 more channel is include_timestamp = True
        """
        # Get data
#         self.board.start_stream(int(self.sample_rate * rec_duration_s))   # Set buffer size to record
#         time.sleep(rec_duration_s + self.buffer_time)    # Wait for recording to complete + buffer_time
        data = self.board.get_board_data(int(self.sample_rate * rec_duration_s))  # Get all data and remove it from internal buffer
        
        # Return data
        if has_accelerometer:  
            if include_timestamp:
                return data[0:12]   # Include timestamp and accelerometer data
            else:
                return data[1:12]   # Include accelerometer data
        else: 
            if include_timestamp:
                return data[0:9]   # Include timestamp, but exclude accelerometer data
            else:
                return data[1:9]   # Exclude timestamp and accelerometer data
            
    def get_recording_no_time(self, has_accelerometer=False, include_timestamp=False):
        """
        Function to collect rec_duration_s amount of recording.
        Argument: 
            has_accelerometer: type bool. If True, include accelerometer data, else only include EEG data.
            include_timestamp: type bool. If True, include the timestamp index data at channel 0.
        Return: 
            data: type numpy.ndarray of shape (num_channels, num_recording), where num_channels = 8 if has_accelerometer = False, else 12. num_recording = self.sample_rate x rec_duration_s. Add 1 more channel is include_timestamp = True
        """
        # Get data
#         self.board.start_stream(int(self.sample_rate * rec_duration_s))   # Set buffer size to record
#         time.sleep(rec_duration_s + self.buffer_time)    # Wait for recording to complete + buffer_time
        data = self.board.get_board_data()  # Get all data and remove it from internal buffer
        
        # Return data
        if has_accelerometer:  
            if include_timestamp:
                return data[0:12]   # Include timestamp and accelerometer data
            else:
                return data[1:12]   # Include accelerometer data
        else: 
            if include_timestamp:
                return data[0:9]   # Include timestamp, but exclude accelerometer data
            else:
                return data[1:9]   # Exclude timestamp and accelerometer data
            
        
    def stop_stream(self):
        """
        Function to stop the streaming and release the session.
        """
        # Stop stream, release session
        self.board.stop_stream()
        self.board.release_session()
        
        
    def save_recording(self, data_to_save: np.ndarray):
        """
        Function to save recording in .npy format to disk.
        Argument:
            data_to_save: type np.ndarray. Recorded data to save to file
        Return:
            file_name: type str. Name of file saved to disk
        """
        # Create file name to save file
        file_date = datetime.datetime.now().date().strftime('%y%m%d')
        file_time = datetime.datetime.now().time().strftime('%H%M%S')
        file_date_time = os.path.join('records',self.serial_port + '_EEG_' + file_date + '_' + file_time + '.npy')
        np.save(file_date_time, data_to_save)      
        return file_date_time
    
    