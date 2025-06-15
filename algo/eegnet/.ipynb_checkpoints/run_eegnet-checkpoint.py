import logging
log = logging.getLogger(__name__)

import tensorflow as tf
log.debug('tensorflow version used to run = {}. EEGNet proposed is 1.12.0'.format(tf.__version__))
log.debug('Compatible version: tf=2.3.0, CUDA=10.1, CuDNN=7.6.5')

import numpy as np
from tensorflow.keras import utils as np_utils
from algo.eegnet import EEGModels
from algo.eegnet import EEGModelsAfterFFT
from algo.eegnet import scu
from algo.eegnet.cca import cca
from algo.eegnet.cca import reference_frequencies
# from algo.eegnet import EEGModels_original
# import EEGModels import EEGNet
import matplotlib.pyplot as plt

import time
import datetime
import os

## Using IPython.core.debugger.set_trace for debug
from IPython.core.debugger import set_trace


def train_eegnet(data_X, data_Y, data_X_val, data_Y_val, index_subject_to_train=0, epoch=130, learning_rate = 0.001, batch_size=4, val_pct=0.2, F1=8, D=2, F2=8*2, \
                optimizer='adam', loss='categorical_crossentropy', save_weights_while_training=[False,100], save_final_weight_n_model=[True, True], pre_trained_model=None, checkpoint_dir='', regularize=[False,'L1',0.01], model='eegnet', kernLength=256):
    """
    Function: Train input dataset with EEGNet (https://github.com/vlawhern/arl-eegmodels)
    Argument:
        data_X: input X data for training, numpy.ndarray of dimension (subject_num, trial_num, sample_num, channel_num)
        data_Y: input Y label for training, numpy.ndarry of dimension (subject_num, trial_num, 1)
        data_X_val: input X data for validation, numpy.ndarray of dimension (subject_num, trial_num, sample_num, channel_num)
        data_Y_val: input Y label for validation, numpy.ndarry of dimension (subject_num, trial_num, 1)
        index_subject_to_train: index of the subject number (subject_num) to train. If -1, collapse all subjects together and train. type = int
        epoch: number of epoch to train. type = int
        learning_rate: learning rate for adam optimizer
        batch_size: number of samples per batch of computation. type = int
        val_pct: percentage of dataset to be reserved for validation. takes value between 0 and 1. type = float.
        F1, D, F2: hyper-parameters for the EEGNet model. type = int
        optimizer: optimizer to be used for the training updates. possible optimizer are: 'adam', 
        loss: loss function to be used for the training updates. possible loss functions are: 'categorical_crossentropy', 
        save_weights_while_training: list of configuration while training the tensorflow model
            [0]: type bool: True - To save weights while training model; False - Not to save weights while training
            [1]: type int: number of epochs before saving the weights while training
        save_final_weight_n_model: list of configuration to save weights / full model
            [0]: type bool: True - save weights of final model; False - Dont save
            [1]: type bool: True - save full final model; False - Dont save
        pre_trained_model: a pretrained EEGNet Model of type tensorflow.python.keras.engine.functional.Functional if provided. parameters F1,F2,D,optimizer,loss will be disregarded if pre_trained model is provided. Default = None implies start training using EEGNet with hyperparameters provided.
        checkpoint_dir: type string. Full directory used as checkpoint to save weights through training
        regularize: type list. Parameter for regularization if required
            regularize[0]: type bool, True if require regularization, False if not
            regularize[1]: type string. 'L1' or 'L2' regularization
            regularize[2]: type float. weight of regularization
        model: type str. Acceptable models are 'eegnet', 'eegnet_after_fft', 'scu', 'cca'
        kernLength: type int. Size of kernel length, default is 256
    Return:
        model: Trained tensorflow model, including the history of training in model.history
    """
    # Check validity of input signal
    assert_train_eegnet(data_X, data_Y, data_X_val, data_Y_val, index_subject_to_train, epoch, batch_size, val_pct, F1, D, F2,
                optimizer, loss, save_weights_while_training, save_final_weight_n_model, pre_trained_model,checkpoint_dir, model)
    
    # Retrieve relevant subject and shape to that for EEGNet for processing
    if model == 'eegnet':
        X_train, Y_train = reshape_on_subject(data_X, data_Y, index_subject_to_train)
        X_val, Y_val = reshape_on_subject(data_X_val, data_Y_val, index_subject_to_train)
    elif model == 'eegnet_after_fft':
        X_train, Y_train = reshape_on_subject(data_X, data_Y, index_subject_to_train,model=model)
        X_val, Y_val = reshape_on_subject(data_X_val, data_Y_val, index_subject_to_train,model=model)
    elif model == 'scu':
        X_train, Y_train = reshape_on_subject(data_X, data_Y, index_subject_to_train)
        X_val, Y_val = reshape_on_subject(data_X_val, data_Y_val, index_subject_to_train)
    elif model == 'cca':
        X_train, Y_train = reshape_on_subject(data_X, data_Y, index_subject_to_train)
        X_val, Y_val = reshape_on_subject(data_X_val, data_Y_val, index_subject_to_train)
    else:
        assert False, 'wrong model for reshape_on_subject'

    # Split dataset to train and validation set using val_pct
    # X_train, Y_train, X_val, Y_val = train_val_split(X, Y, val_pct)

       
    
    # Setup EEGNet model for training
    if optimizer == 'adam':
        # optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,clipnorm=0.5,clipvalue=0.5)
        
    if pre_trained_model is None:   # Start EEGNet training from scratch
        if model == 'eegnet':
            model = EEGModels.EEGNet(nb_classes=Y_train.shape[1], Chans=X_train.shape[2], kernLength=kernLength, F1=F1, D=D, F2=F2, Samples=X_train.shape[3], regularize=regularize)
            model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        elif model == 'eegnet_after_fft':
            model = EEGModelsAfterFFT.EEGNet(nb_classes=Y_train.shape[1], Chans=X_train.shape[2], F1=F1, D=D, F2=F2,  Samples=X_train.shape[3], regularize=regularize)
            model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        elif model == 'scu':
            model = scu.scu_model(nb_classes=Y_train.shape[1], Chans=X_train.shape[2], F1=F1, D=D, F2=F2,  Samples=X_train.shape[3], regularize=regularize)
            model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        elif model == 'cca':
            target_frequencies = [6.0, 6.666666, 7.5, 8.57] 
            sampling_rate = 250
            ref_freq = reference_frequencies.generate_reference_signals(target_frequencies, size=X_train.shape[3], sampling_rate=sampling_rate, num_harmonics=3)
            print(X_train[:,0].shape, ref_freq.shape, Y_train.shape)
            predicted_class, accuracy, predicted_probabilities, _, _ = cca.perform_cca(X_train[:,0], ref_freq, labels=np.argmax(Y_train, axis=1))
            print(accuracy)
            return predicted_class, predicted_probabilities
        else:
            assert False, 'wrong model for training'
    else:     # Use pre-trained model
        log.info('Using pre-trained EEGNet Model')
        model = pre_trained_model

    
    # Prepare directory for saving checkpoint
#     CURRENT_DIR= os.path.dirname(__file__)    # Use os.getcwd() for running directly on .ipynb
#     CHECKPOINT_DIR = os.path.join(CURRENT_DIR,'..','..','checkpoints')
#     CURRENT_CKPT_REL = str(datetime.datetime.now()).replace('-','').replace(':','').replace('.','_').replace(' ','_')
#     CURRENT_CKPT_DIR = os.path.join(CHECKPOINT_DIR, CURRENT_CKPT_REL)
#     os.mkdir(CURRENT_CKPT_DIR)
#     log.info('Created directory {} for checkpoint'.format(CURRENT_CKPT_DIR))
#     NAME_MODEL_WEIGHT = 'tf_model_weights'
#     NAME_MODEL_FULL = 'tf_model_full'
    
    
    # Prepare callback to save weights while training
    CKPT_PATH = os.path.join(checkpoint_dir, 'cp-{epoch:04d}.ckpt')
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=CKPT_PATH, 
        verbose=1, 
        save_weights_only=True,
        save_freq='epoch',
        period= save_weights_while_training[1])
    
    
    # Actual training
    print('====================================================================================================')
    print('Start Training: EEGNet setup with parameters nb_classes={}, Chans={}, F1={}, D={}, F2={}, Trials={}, Samples={}'.format(Y_train.shape[1], X_train.shape[2], F1, D, F2, X_train.shape[0], X_train.shape[3]))
    print('Regularization = {}, Params = {}, {}'.format(regularize[0], regularize[1], regularize[2]))
    log.info('====================================================================================================')
    log.info('Start Training: EEGNet setup with parameters nb_classes={}, Chans={}, F1={}, D={}, F2={}, Trials={}, Samples={}'.format(Y_train.shape[1], X_train.shape[2], F1, D, F2, X_train.shape[0], X_train.shape[3]))
    
    t = time.time()
    if save_weights_while_training[0] == True:   # To save weights while training
        history=model.fit(X_train, Y_train, epochs=epoch, verbose=0, validation_data=(X_val, Y_val), callbacks=[cp_callback], batch_size=batch_size)
    else:
        history=model.fit(X_train, Y_train, epochs=epoch, verbose=0, validation_data=(X_val, Y_val), batch_size=batch_size)
    elapsed = time.time() - t
    print('Completed Training of {} epoch in {}min {}s'.format(epoch, elapsed//60, elapsed%60))
    print('====================================================================================================')
    
    log.info('Completed Training of {} epoch in {}min {}s'.format(epoch, elapsed//60, elapsed%60))
    log.info('====================================================================================================')
    
           
        
    # Plot performance of training
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    if index_subject_to_train == -1:
        plt.title('Accuracy for all subjects and trials')
    else:
        plt.title('Accuracy for Subject {}'.format(index_subject_to_train))
    plt.xlabel('epoch')
    plt.ylabel('validation accuracy (%)')
    plt.legend(['Train Accuracy','Validation Accuracy'])
    plt.show()
    
    return model, history



    
def assert_train_eegnet(data_X, data_Y, data_X_val, data_Y_val, index_subject_to_train, epoch, batch_size, val_pct, F1, D, F2,
                optimizer, loss, save_weights_while_training, save_final_weight_n_model, pre_trained_model, checkpoint_dir, model):
    """
    data_X: input X data for training, numpy.ndarray of dimension (subject_num, trial_num, sample_num, channel_num)
    data_Y: input Y label for training, numpy.ndarry of dimension (subject_num, trial_num, 1)
    data_X_val: input X data for validation, numpy.ndarray of dimension (subject_num, trial_num, sample_num, channel_num)
    data_Y_val: input Y label for validation, numpy.ndarry of dimension (subject_num, trial_num, 1)
    index_subject_to_train: index of the subject number (subject_num) to train. If -1, collapse all subjects together and train. type = int
    epoch: number of epoch to train. type = int
    batch_size: number of samples per batch of computation. type = int
    val_pct: percentage of dataset to be reserved for validation. takes value between 0 and 1. type = float.
    F1, D, F2: hyper-parameters for the EEGNet model. type = int
    optimizer: optimizer to be used for the training updates. possible optimizer are: 'adam', 
    loss: loss function to be used for the training updates. possible loss functions are: 'categorical_crossentropy', 
    save_weights_while_training: list of configuration while training the tensorflow model
        [0]: type bool: True - To save weights while training model; False - Not to save weights while training
        [1]: type int: number of epochs before saving the weights while training
    save_final_weight_n_model: list of configuration to save weights / full model
        [0]: type bool: True - save weights of final model; False - Dont save
        [1]: type bool: True - save full final model; False - Dont save
    checkpoint_dir: type string. Full directory used as checkpoint to save weights through training
    model: type str. Acceptable models are 'eegnet', 'eegnet_after_fft'
    """
    import numpy as np
    import tensorflow as tf
    
    # Check data_X and data_Y input, together with index_subject_to_train
    assert type(data_X) == np.ndarray, 'Input data_X must be of np.ndarray type'
    assert len(data_X.shape) == 4, 'Input data_X must be of shape (subject_num, trial_num, sample_num. channel_num)'
    assert type(data_Y) == np.ndarray, 'Input data_Y must be of np.ndarray type'
    assert len(data_Y.shape) == 3, 'Input data_Y must be of shape (subject_num, trial_num, 1)'
    assert data_Y.shape[2] == 1, 'Input data_Y must be of shape (subject_num, trial_num, 1), where .shape[2] = 1'
    assert data_X.shape[0] == data_Y.shape[0], 'data_X and data_Y must have the same number of subjects'
    assert data_X.shape[1] == data_Y.shape[1], 'data_X and data_Y must have the same number of trials'
    
    assert type(data_X_val) == np.ndarray, 'Input data_X_val must be of np.ndarray type'
    assert len(data_X_val.shape) == 4, 'Input data_X_val must be of shape (subject_num, trial_num, sample_num. channel_num)'
    assert type(data_Y_val) == np.ndarray, 'Input data_Y_val must be of np.ndarray type'
    assert len(data_Y_val.shape) == 3, 'Input data_Y_val must be of shape (subject_num, trial_num, 1)'
    assert data_Y_val.shape[2] == 1, 'Input data_Y must be of shape (subject_num, trial_num, 1), where .shape[2] = 1'
    assert data_X_val.shape[0] == data_Y_val.shape[0], 'data_X_val and data_Y_val must have the same number of subjects'
    assert data_X_val.shape[1] == data_Y_val.shape[1], 'data_X_val and data_Y_val must have the same number of trials'
    
    assert type(index_subject_to_train) == int, 'index_subject_to_train must be of integer type'
    if index_subject_to_train != -1:
        assert index_subject_to_train <= data_X.shape[0]-1, 'Subject to train not found in input dataset'
        assert index_subject_to_train >= 0, 'index_subject_to_train must be a positive integer'
    
    assert type(epoch) == int, 'epoch must be of integer type'
    assert epoch > 0, 'epoch must be a positive integer'
    
    assert type(batch_size) == int, 'batch_size must be of integer type'
    assert batch_size > 0, 'batch_size must be a positive integer'
    
    assert type(val_pct) == float, 'val_pct must be of float type'
    assert val_pct <= 1 and val_pct >= 0, 'val_pct must be of value between 0 and 1'
    
    assert type(F1) == int, 'F1 must be of int type'
    assert F1 > 0, 'F1 must be a positive integer'
    
    assert type(D) == int, 'D must be of int type'
    assert D > 0, 'D must be a positive integer'
    
    assert type(F2) == int, 'F2 must be of int type'
    assert F2 > 0, 'F2 must be a positive integer'
    
    assert type(save_weights_while_training) == list, 'save_weights_while_training must be of list type'
    assert len(save_weights_while_training) == 2, 'save_weights_while_training must be list of size 2'
    assert type(save_weights_while_training[0]) == bool, 'save_weights_while_training[0] must be bool type'
    assert type(save_weights_while_training[1]) == int, 'save_weights_while_training[1] must be int type'
    
    assert type(save_final_weight_n_model) == list, 'save_final_weight_n_model must be of list type'
    assert len(save_final_weight_n_model) == 2, 'save_final_weight_n_model must be list of size 2'
    assert type(save_final_weight_n_model[0]) == bool, 'save_final_weight_n_model[0] must be bool type'
    assert type(save_final_weight_n_model[1]) == bool, 'save_final_weight_n_model[1] must be int type'
    
    # if pre_trained_model is None:
    #     pass
    # else:
    #     assert type(pre_trained_model) == tf.python.keras.engine.functional.Functional, 'pre_trained model must be of type tensorflow.python.keras.engine.functional.Functional'
        
    assert type(checkpoint_dir) == str, 'checkpoint_dir must be of type string'
    
    assert type(model) == str, 'model must be of type string'
    assert model == 'eegnet' or model == 'eegnet_after_fft' or model == 'scu' or model == 'cca', 'model must be either "eegnet", "eegnet_after_fft", "scu" or "cca" only'
    
    

def reshape_on_subject(data_X, data_Y, index_subject_to_train, model='eegnet'):
    """
    Function to reshape the input to that to be accepted by eegnet
    Argument: 
        data_X: input X data for training, numpy.ndarray of dimension (subject_num, trial_num, sample_num, channel_num)
        data_Y: input Y label for training, numpy.ndarry of dimension (subject_num, trial_num, 1)
        index_subject_to_train: index of the subject number (subject_num) to train. If -1, collapse all subjects together and train. type = int
        model: type str. acceptable input are 'eegnet' and 'eegnet_after_fft'. For purpose of rejecting trials. 
    Return:
        X : np.ndarray of shape (trial, 1, channel, samples). Note that subject is removed
        Y : np.ndarray of shape (trial, class)
    """
    # Retrieve relevant subject and shape to that for EEGNet for processing
    # For prediction, data_Y can be of None type
    if index_subject_to_train == -1:    # To collapse all subjects to train together
        X=np.reshape(data_X, (-1, 1, data_X.shape[2], data_X.shape[3]), order='C')  # X current dimension = (subject+trial_num, 1, samples, channels)
        X=np.transpose(X, (0,1,3,2))                    # X dim to EEGNet = (subject+trial_num, 1, channels, samples)
        if data_Y is None:
            Y = None
        else:
            Y = np.reshape(data_Y, (-1, 1), order='C')        # Y current dimension = (subject+trial_num, 1)
            Y = np_utils.to_categorical(Y[:,0])             # Y dim to EEGNet = (subject+trial_num, 5), where 5 is 4 class+class0
        
    else:       # To train only the specified subject
        X=np.transpose(data_X[[index_subject_to_train],:,:,:], (1,0,3,2))  # EEGNet accepts dim(trial_num,1,channels,samples)
        # Alternatively can use np.expand_dims(data_X[index_subject_to_train,:,:,:], axis=0) before doing np.transpose
        if data_Y is None:
            Y = None
        else:
            Y = data_Y[index_subject_to_train,:,:]            # Y current dimension = (trial_num, 1)
            Y = np_utils.to_categorical(Y[:,0])             # Y dim to EEGNet = (trial_num, 5)
    
    ###########################################
    # GTL: Newly added on 08 Mar 2022
    # Algorithm to remove trials that are flagged to be rejected (see function clean_data.clean_noisy_channel)
    print('Shape before trimming = {}'.format(X.shape))
    if model == 'eegnet':
        all_samples_reject = np.sum(X, axis=3) == 1000*X.shape[3]
        all_samples_channel_reject = np.sum(all_samples_reject, axis=2) == X.shape[2]
    elif model == 'eegnet_after_fft':
        print('Is eegnet_after_fft')
        all_samples_reject = np.round(X[:,:,:,0]) == 2e3   # 1000 folded to 0.5 thus 2000
        all_samples_channel_reject = np.sum(all_samples_reject, axis=2) == X.shape[2]
        
    X_trim = X[np.logical_not(all_samples_channel_reject[:,0])]
    Y_trim = Y[np.logical_not(all_samples_channel_reject[:,0])]
    print('Shape after trimming = {}'.format(X_trim.shape))

        # Check to ensure that after trimming, have trials of each class (i.e. for none of the class do we trim off all trials)
    has_all_classes = np.sum(np.sum(Y_trim, axis=0) > 0) == Y.shape[1]
    each_class =             np.sum(Y_trim, axis=0) > 0
    if not(has_all_classes):
        print('After trimming the rejects, some class(es) {} is/are fully removed'.format(np.arange(Y.shape[1])[np.logical_not(each_class)]))
        return None, None
#     assert has_all_classes, 'After trimming the rejects, some class(es) {} is/are fully removed'.format(np.arange(Y.shape[1])[np.logical_not(each_class)])
    ###########################################


#     return X, Y
    return X_trim, Y_trim
    

    


def predict_evaluate(data_X, trained_model, index_subject_to_predict=0, data_Y=None):
    """
    Function: Predict and evaluate (if app) the output label using the trained model 
    Argument:
        data_X: input X data, numpy.ndarray of dimension (subject_num, trial_num, sample_num, channel_num)
        trained_model: the trained model of type tensorflow.python.keras.engine.functional.Functional
        index_subject_to_predict: index of the subject number (subject_num) to predict. If -1, collapse all subjects together and predict. type = int
        data_Y: input Y label, numpy.ndarry of dimension (subject_num, trial_num, 1). Default None. If None, no evaluation
        
    Return:
        pred_Y: predicted label Y with dimension (trial_num, number of classes)
        truth_Y: ground truth label Y with dimension (trial_num, number of classes)
        accuracy: accuracy of the prediction, compared to data_Y ground truth. type float of value between 0 and 1
    """
    # Check correctness of input
    assert_predict_evaluate(data_X, trained_model, index_subject_to_predict, data_Y)
    
    import tensorflow as tf
    import numpy as np
    
    # Retrieve relevant subject and shape to that for EEGNet for processing
    # Shape of X after reshape_on_subject = (trial, 1, channel, samples)
    X, Y = reshape_on_subject(data_X, data_Y, index_subject_to_predict)
    
    if X is None: # Some classes have no more data
        pred_Y = np.array([999])  # To symbolise invalid data
        truth_Y = np.array([999])
        accuracy = 999
        
    elif Y is None:   # No ground truth available, perform prediction only, no evaluation
        pred_Y = trained_model.predict(X)
        pred_Y = np.argmax(pred_Y, axis=1)
        truth_Y = None
        accuracy = None
        
    else:           # Ground truth provided. Perform prediction and evaluation
        pred_Y = trained_model.predict(X)
        pred_Y = np.argmax(pred_Y, axis=1)
        truth_Y = np.argmax(Y, axis=1)
        _, accuracy = trained_model.evaluate(X, Y, verbose=0)
    
    return pred_Y, truth_Y, accuracy
        
    

def assert_predict_evaluate(data_X, trained_model, index_subject_to_predict, data_Y):
    import tensorflow as tf
    
    # Check data_X and data_Y input, together with index_subject_to_train
    assert type(data_X) == np.ndarray, 'Input data_X must be of np.ndarray type'
    assert len(data_X.shape) == 4, 'Input data_X must be of shape (subject_num, trial_num, sample_num. channel_num)'
    
    if data_Y is None:
        pass
    else:
        assert type(data_Y) == np.ndarray, 'Input data_Y must be of np.ndarray type, else None type'
        assert len(data_Y.shape) == 3, 'Input data_Y must be of shape (subject_num, trial_num, 1)'
        assert data_Y.shape[2] == 1, 'Input data_Y must be of shape (subject_num, trial_num, 1), where .shape[2] = 1'
        assert data_X.shape[0] == data_Y.shape[0], 'data_X and data_Y must have the same number of subjects'
        assert data_X.shape[1] == data_Y.shape[1], 'data_X and data_Y must have the same number of trials'

    assert type(index_subject_to_predict) == int, 'index_subject_to_train must be of integer type'
    if index_subject_to_predict != -1:
        assert index_subject_to_predict <= data_X.shape[0]-1, 'Subject to train not found in input dataset'
        assert index_subject_to_predict >= 0, 'index_subject_to_train must be a positive integer'
        
    # assert type(trained_model) == tf.python.keras.engine.functional.Functional, \
        # 'trained_model must be of type tensorflow.python.keras.engine.functional.Functional'
    
    
def save_model(directory, model, save_final_weight_n_model=[False, True], model_name_weights='tf_model_weights', model_name_full='tf_model_full'):
    """
    Save the final model, weights only or full model
    save_final_weight_n_model: list of configuration to save weights / full model
        [0]: type bool: True - save weights of final model; False - Dont save
        [1]: type bool: True - save full final model; False - Dont save
    model: trained model of type tensorflow.python.keras.engine.functional.Functional that is to be saved in the directory provided
    """
     # Save model weights/full 
    if save_final_weight_n_model[0] == True:    # Save trained model weights 
        model.save_weights(os.path.join(directory,model_name_weights))
        print('Created and saved model weight {} to {}'.format(model_name_weights, directory))
        log.info('Created and saved model weight {} to {}'.format(model_name_weights, directory))
    if save_final_weight_n_model[1] == True:    # Save full trained model
        model.save(os.path.join(directory,model_name_full))
        print('Created and saved full model {} to {}'.format(model_name_full, directory))
        log.info('Created and saved full model {} to {}'.format(model_name_full, directory))
    
    
def load_full_model(directory, model_name='tf_model_full'):
    """
    Load the tensorflow model using just the provided directory that model resides in
    Note may need to replace some \ with // on directory path
    """
    import os
    import tensorflow as tf
    
    full_directory = os.path.join(directory, model_name)
    print('Loading Model from ', full_directory)
    trained_model = tf.keras.models.load_model(full_directory)
    
    return trained_model




def track_train_accuracy(train_history=None, checkpoint_dir='', folder='test', run_number=1, pre_trained = False, pct_acc=0.0, F1=1, F2=1, D=1, index_subject_to_train=1, num_epoch=1, num_electrode=64, f_class_type='NA', num_class=40, cr_subj_acc = -1, offset_sample = 0, elapsed_min=0, elapsed_sec=0):
    """
    Function that write to train_results.out file of each training checkpoint directory with the performance of the current training cycle, and the train cycles prior. Results are arranged in order of the accuracy (%) denoted by pct_acc. This allows offline analysis and comparison of the hyperparameters and their effect on the accuracy results.
    
    Arguments:
        train_history: numpy.ndarray type of shape (num_of_train_cycle, num_of_args_tracked) from output of previous track_train_accuracy. Default = None
        checkpoint_dir: directory to save the train_results.out file
        folder: string type indicating the folder name where the model is evaluated and saved
        run_number: a running number, starting from 1 (to be handled by function that calls track_train_accuracy). gives insight on order in which training occured. type int
        pre_trained: type bool indicating if pre_trained model was used in the training
        pct_acc: percentage accuracy of the model. type float of value between 0 and 1
        F1, F2, D: hyperparameters for EEGNet. type int
        index_subject_to_train: The index of the subject trained on
        num_epoch: number epoch that was trained on
        num_electrode: number of electrodes (i.e. channel) trained on
        f_class_type: type string with value: 'NA', 'freq', 'spatial', 'manual'
        cr_subj_acc: type int. Denotes the cross_subject accuracy. Default to -1 if NA
        offset_sample: type int. Sample offset to shift time sample filtering parameter
        num_class: type int, denoting number of classes 
        elapsed_min, elapsed_sec: amount of time for the run
        
        
    Return:
        history: history of the previous training runs till now. numpy.ndarray type of shape (num_of_train_cycle, num_of_args_tracked)
            num_of_train_cycle: number of training runs, which should equal to run_number if run_number is incremented correctly by parent function
            num_of_args_tracked: number of hyperparameters tracked. E.g. if only F1, F2 and num_epoch, will be 3
    """
    import numpy as np
    import os

    # header and format string definition
    header_string = 'folder_name \t\t\t\t\t\t run_number \t\t is_pre_trained \t % accuracy \t F1 \t\t F2 \t D \t\t subject_num \t\t num_epoch \t\t num_electrode \t\t class_type \t\t num_class \t\t cr_subj_acc \t\t offset_sample \t\t time_taken'
    format_string = '  %s \t\t\t %s \t\t\t\t %s \t\t\t\t %s \t\t\t %s \t\t %s \t %s \t %s \t\t\t\t %s \t\t\t %s \t\t\t\t %s \t\t\t %s \t\t\t %s \t\t\t %s \t\t\t\t %smin %ssec'
    
    
    # Reformat string to same length
    run_number = '{:4.0f}'.format(run_number)
    pct_acc = '{:.3f}'.format(pct_acc)    # Set pct_acc to string with 3 decimal place
    F1 = '{:4.0f}'.format(F1)
    F2 = '{:4.0f}'.format(F2)
    D = '{:4.0f}'.format(D)
    index_subject_to_train = '{:4.0f}'.format(index_subject_to_train)
    num_epoch = '{:5.0f}'.format(num_epoch)
    num_electrode = '{:5.0f}'.format(num_electrode)
    
    if f_class_type == 'freq' or f_class_type == 'spatial':
        num_class = '{:5.0f}'.format(num_class)
    else:
        num_class = '{:4s}'.format('NA') 
    f_class_type = '{:6s}'.format(f_class_type)
    
    cr_subj_acc = '{:.3f}'.format(cr_subj_acc)
    offset_sample = '{:5.0f}'.format(offset_sample)
    elapsed_min = '{:.0f}'.format(elapsed_min)
    elapsed_sec = '{:.0f}'.format(elapsed_sec)
    
    
    
    
    
    # capturing information in array
    if int(run_number) == 1:    
        history =  np.array((folder, run_number, pre_trained, pct_acc, F1, F2, D, index_subject_to_train, num_epoch, num_electrode, f_class_type, num_class, cr_subj_acc, offset_sample, elapsed_min, elapsed_sec))
        history = np.reshape(history, (1,-1))
        
    else: 
        history = np.append(train_history, \
                  [np.array((folder, run_number, pre_trained, pct_acc, F1, F2, D, index_subject_to_train, num_epoch, num_electrode, f_class_type, num_class, cr_subj_acc, offset_sample, elapsed_min, elapsed_sec))\
                  ], axis=0)

    
    # Sort the numpy array according to column 3 (pct_acc)
    if run_number == 1:   # No need to sort if only 1 run
        history_to_print = history
    else: 
        history_to_print = history[history[:,3].argsort()]
    
    # Save to train_results.out
    np.savetxt(os.path.join(checkpoint_dir,'train_results.out'), history_to_print, header=header_string, fmt=format_string)
    
    return history_to_print






def create_checkpoint_dir():
    import os
    import datetime
    CURRENT_DIR= os.path.dirname(__file__)    # Use os.getcwd() on .ipynb; os.path.dirname(__file__) on .py
    CHECKPOINT_DIR = os.path.join(CURRENT_DIR,'..','..','checkpoints')
    CURRENT_CKPT_REL = str(datetime.datetime.now()).replace('-','').replace(':','').replace('.','_').replace(' ','_')
    CURRENT_CKPT_DIR = os.path.join(CHECKPOINT_DIR, CURRENT_CKPT_REL)
    os.mkdir(CURRENT_CKPT_DIR)
    log.info('Created directory {} for checkpoint'.format(CURRENT_CKPT_DIR))
    print('Created directory {} for checkpoint'.format(CURRENT_CKPT_DIR))
    return CURRENT_CKPT_REL, CURRENT_CKPT_DIR


def filter_by_channel(dataset, sel_channel, invert_ch=False):
    """
    Function to filter the dataset to use only the respective channels
    Augument:
        dataset: input dataset of type numpy.ndarray with shape (subject_num, trial_num, sample_num, channel_num)
        sel_channels: selected channels of type list, containing the channels in dataset to be selected 
        invert_ch: type bool. If True, the output is that which channels are not in sel_channel
    Return:
        output: output dataset of type numpy.ndarry with shape (subject_num, trial_num, sample_num, channel_num)
    """
    import numpy as np
    if invert_ch is False:
        return dataset[:,:,:,sel_channel]
    else: 
        all_ch_index = np.array(range(dataset.shape[3]))
        filtered_ch_index = all_ch_index[np.isin(all_ch_index, sel_channel, invert=True)]
        return dataset[:,:,:,filtered_ch_index]


def filter_by_subject_n_trial(dataset_X, dataset_Y, subj_trial_index):
    """
    Function to produce output dataset_X, dataset_Y after filtering for subj_trial_index
    Argument:
        dataset_X: type np.ndarray w shape (subject_num, trial_num, sample_num, channel_num);
        dataset_Y: corresponding type np.ndarray w shape (subject_num, trial_num, 1)
        subj_trial_index: np.ndarray type of shape (2, num), where 
            num: denotes the number of selected subject trial to be selected (if invert=False)
            output[0]: denotes the subject_num for each of the num
            output[0]: denotes the trial_num for each of the num
    Return:
        output_X: type np.ndarray w shape (subject_num, trial_num, sample_num, channel_num);
        output_Y: corresponding type np.ndarray w shape (subject_num, trial_num, 1)
    """
    import numpy as np
    temp_out_X = dataset_X[subj_trial_index[0],subj_trial_index[1]]
    temp_out_Y = dataset_Y[subj_trial_index[0],subj_trial_index[1]]
    
    # Find number of unique subject index in subj_trial_index
    unique = len(np.unique(subj_trial_index[0]))

    # Reshape output to required shape
    output_X = np.reshape(temp_out_X, (unique,-1,dataset_X.shape[2],dataset_X.shape[3]))
    output_Y = np.reshape(temp_out_Y, (unique, -1, 1))
    
    # Change labels of output_Y s.t. output labels are from 0 to num_class (without missing numbers between)
    _, unique_inverse = np.unique(output_Y, return_inverse=True)
    output_Y = np.reshape(unique_inverse, output_Y.shape)
    
    return output_X, output_Y    


def filter_by_sample(data_X, data_Y, filter_type='none', value=None):
    """
    Function to filter the dataset by sample
    Argument:
        data_X: input data with shape (subject_num, trial_num, sample_num, channel_num)
        data_Y: input data with shape (subject_num, trial_num, 1). Can be None if filter_type='random'
        filter_type: type string with values:
            'random': randomly offset the samples by value 
            'fixed' : offset all samples by value
            'fix_length' : Fix length of 276 samples. 
            'manual':
            'none': direct feed-through of data_X and data_Y to output
        value: depends on value of filter_type:
            'random': type int where value denotes the maximum random number of samples to offset from original data_X
            'fixed' : type int, where value denotes the fix sample to offset for all
            'fix_length' : type int, where value denotes the fix sample to offset for all, keeping length of entire sample to 276
            'manual': type list of size 2, where
                value[0]: class_num to change. All other classes no offset but truncated
                value[1]: number of samples to offset by (only positive integer)
    Return:
        output_X: output data with shape (subject_num, trial_num, sample_num, channel_num), where sample_num may be truncated based on the amount of offset
        output_Y: output data with shape (subject_num, trial_num, 1). Same as input data_Y
    """
    output_Y = np.copy(data_Y)
    
    if filter_type == 'none':
        output_X = np.copy(data_X)
    
    elif filter_type == 'fixed':
        assert type(value) is int, 'value must be of int type when filter_type == "fixed" '
        assert value >= 0, 'value must be positive'
        output_X = data_X[:,:,value:]
        
    elif filter_type == 'fix_length':
        assert type(value) is int, 'value must be of int type when filter_type == "fix_length" '
        assert value >= 0, 'value must be positive'
        output_X = data_X[:,:,value:value+275]
    
    elif filter_type == 'random':
        assert type(value) is int, 'value must be of int type when filter_type == "random" '
        assert value >= 0, 'value must be positive' 
        random_sample = np.random.randint(value, size=(data_X.shape[0]*data_X.shape[1]))
        max_random = max(random_sample)
        
        output_X = np.zeros((data_X.shape[0], data_X.shape[1], data_X.shape[2]-max_random, data_X.shape[3]))

        counter=0
        for i in range(data_X.shape[0]):
            for j in range(data_X.shape[1]):
                output_X[i,j] = data_X[i,j,random_sample[counter]:data_X.shape[2]-(max_random-random_sample[counter])]
                counter = counter+1
                
    elif filter_type == 'manual':
        assert type(value) is list, 'value must be of list type when filter_type == "manual" '
        assert len(value) == 2, 'value must be list of length 2 when filter_type == "manual" '
        indexes = np.where(np.isin(output_Y, value[0]))
        indexes = np.array(indexes[:2])

        mat = data_X[indexes[0],indexes[1]]

        offset = value[1]
        changed = mat[:,offset:]

        output_X = data_X
        output_X = output_X[:,:,:-offset]
        output_X[indexes[0],indexes[1]] = changed
        
    return output_X, output_Y
    
    
def cross_subject_acc(checkpoint_folder_str, X_test, Y_test, num_electrode=64, num_class=4, class_type='freq', offset_sample_type='none', offset_sample=0, subject=list(range(10)), filter_f=None, filter_g=None):
    """
    Function to check cross subject accuracy using a specific model and print results out
    Argument
        checkpoint_folder_str: type string. Sub-folder in checkpoint folder to get model (e.g. 20210325_212739_006840)
        X_test: type np.ndarray with shape (subj, trial, sample, ch)
        Y_test: type np.ndarray with shape (subj, trial, 1)
        num_electrode: type int. number of electrodes to use
        num_class: type int. number of classes filtered to
        class_type: type string of value, 'freq', 'spatial' or 'manual'
        offset_sample_type: type string of value, 'fixed', 'random', 'none', 'manual'
        offset_sample: type int to offset the sample
        subject: type list. list of subject to test model on
        filter_f: filter function provided as input to function. Could be the beta_dataprep.select_electrode_position to select the electrodes (thus filter by channel) to run
    Return:
        average_acc: average accuracy of the cross subject evaluation
    """
    import os
    import tensorflow as tf
    
    # Load model
    print('=================================================')
    model = load_full_model(os.path.join('E:\GitHub\eeg_Deep_Learning\checkpoints',checkpoint_folder_str))
    
          
    # To filter dataset if required
    data_X, data_Y = filter_by_sample(X_test, Y_test, filter_type=offset_sample_type, value=offset_sample)
        
    if filter_f is None:
        pass
    else:
        channel_index = filter_f(num_electrode)
        data_X = filter_by_channel(data_X, channel_index)

    if filter_g is None:
        pass
    else:
        filtered = filter_g(data_X, data_Y, criteria=class_type, quantity=num_class)
        data_X, data_Y = filter_by_subject_n_trial(data_X, data_Y, filtered)
                        

    # Check accuracy for respective subjects
    total, count = 0, 0
    print('=================================================')
    print('Checking accuracy performance for {}'.format(checkpoint_folder_str))
    for i in subject:
        pred,truth,acc = predict_evaluate(data_X, model, i, data_Y)
        print('Subject {}, Accuracy {}'.format(i,acc))
        total, count = total+acc, count+1
    print('Average accuracy for {} subjects = {}'.format(count, total/count))
    print('=================================================\n')
    average_acc = total/count
    return  average_acc
    
def get_top_pct_accuracy(accuracy, pct=0.1):
    """
    Function to return the top percentile accuracy. E.g. if accuracy = [0.2, 0.1, 0.5, 0.8], pct=0.25 will return 0.5, while pct=0.5 will return 0.2.
    Argument: 
        accuracy: type list. List of accuracy value
        pct: type float. Of value between 0 and 1
    """
    import numpy as np
    sorted_acc = np.sort(accuracy)
    return sorted_acc[int(np.floor((1-pct)*len(sorted_acc)))]

    
def find_optimum_eegnet(data_X, data_Y, data_X_val, data_Y_val, filter_f=None, filter_g=None, hyperparameters=[], get_test_accuracy=False, X_test=None, Y_test=None, test_subject=list(range(10))):
    """
    Function to train EEGNet to search for the optimum configuration of EEGNet, given the range of input hyperparameters.
    Arguments:
        data_X: input X data for training, numpy.ndarray of dimension (subject_num, trial_num, sample_num, channel_num)
        data_Y: input Y label for training, numpy.ndarry of dimension (subject_num, trial_num, 1)
        data_X_val: input X data for validation, numpy.ndarray of dimension (subject_num, trial_num, sample_num, channel_num)
        data_Y_val: input Y label for validation, numpy.ndarry of dimension (subject_num, trial_num, 1)
        filter_f: filter function provided as input to function. Could be the beta_dataprep.select_electrode_position to select the electrodes (thus filter by channel) to run
        filter_g: filter function to beta_dataprep.select_label
        hyperparameters: list of hyperparameters and start_index, end_index, interval. E.g.
            [['F1', 2, 12, 4], ['D', 4, 10, 1]] 
            will tune F1 from 2 to 12 in steps of 4, and D from 4 to 10 in steps 1
            for 'num_electrode' and 'num_class'. Contains index 4 and 5, see code below
        get_test_accuracy: type bool. If True, perform cross_subject_acc on the test_subject (see next Argument)
        X_test: type np.ndarray. Test X data with shape (subject_num, trial_num, sample_num, channel_num)
        Y_test: type np.ndarray. Test Y label with shape (subject_num, trial_num, 1)
        test_subject: type list. List containing index of the test subject to be used to check accuracy
    """
    import time
    # Initialize variables
    pre_trained_model = None
            
    # For tracking purpose
    train_history = None
    run_number = 1
    
    # Unpack hyperparameters input
    num_electrode_range = [8]
    num_class_range = [4]
    offset_sample_range = [0]
    index_subject_to_train_range = [1]
    F1_range = [8]
    D_range = [2]
    epoch_range = [100]
    
    for i in hyperparameters:
        assert len(range(i[1], i[2], i[3])) > 0, 'Hyperparameters for {} is set such that no run will occur'.format(i[0])
        if i[0] == 'num_electrode':
            num_electrode_manual = i[5]
            if num_electrode_manual == 'manual':
                num_electrode_range = i[6]
            else:
                num_electrode_range = range(i[1], i[2], i[3])
            num_electrode_invert = i[4]
            
        elif i[0] == 'num_class':
            num_class_manual = i[6]
            if num_class_manual == 'manual':
                num_class_range = i[7]
            else:
                num_class_range = range(i[1], i[2], i[3])
            num_class_type = i[4]
            num_class_invert = i[5]
            
        elif i[0] == 'offset_sample':
            offset_sample_range = range(i[1], i[2], i[3])
            offset_sample_type = i[4]
            if offset_sample_type == 'manual':
                offset_manually = i[5]
            
        elif i[0] == 'index_subject_to_train':
            index_subject_to_train_range = range(i[1], i[2], i[3])
        elif i[0] == 'F1':
            F1_range = range(i[1], i[2], i[3])
        elif i[0] == 'D':
            D_range = range(i[1], i[2], i[3])
        elif i[0] == 'epoch':
            epoch_range = range(i[1], i[2], i[3])
        
    
    data_X_org = np.copy(data_X)
    data_Y_org = np.copy(data_Y)
    data_X_val_org = np.copy(data_X_val)
    data_Y_val_org = np.copy(data_Y_val)
    
    # Training with hyperparameter
    total_iter = \
        len(num_electrode_range)*\
        len(num_class_range)*\
        len(offset_sample_range)*\
        len(index_subject_to_train_range)*\
        len(F1_range)*\
        len(D_range)*\
        len(epoch_range)
    current_iter = 0
    for num_electrode in num_electrode_range:
        for offset_sample in offset_sample_range:
            for index_subject_to_train in index_subject_to_train_range:
                for F1 in F1_range: # range(80, 115, 8):   # range(start, end, interval)
                    for D in D_range: # range (4, 12, 4):
                        for epoch in epoch_range:
                            for num_class in num_class_range:

                                t = time.time()
                                F2 = F1*D
                                current_iter = current_iter+1
                                
                                print('====================================================================================================')
                                print('Performing training for iteration {} / {}'.format(current_iter, total_iter))
                                print('Hyperparameter: num_electrode={}, offset_sample={}, subject={}, F1={}, D={}, epoch={}, num_class={}'.format(num_electrode, offset_sample, index_subject_to_train, F1, D, epoch, num_class))

                                # Get checkpoint directory to save training results and full model
                                CURRENT_CKPT_REL, CURRENT_CKPT_DIR = create_checkpoint_dir()

                                # To filter dataset if required
                                if offset_sample_type == 'manual':
                                    data_X, data_Y = filter_by_sample(data_X_org, data_Y_org, 
                                                                  filter_type=offset_sample_type, value=offset_manually)
                                    data_X_val, data_Y_val = filter_by_sample(data_X_val_org, data_Y_val_org, 
                                                                  filter_type=offset_sample_type, value=offset_manually)
                                else:
                                    data_X, data_Y = filter_by_sample(data_X_org, data_Y_org, 
                                                                  filter_type=offset_sample_type, value=offset_sample)
                                    data_X_val, data_Y_val = filter_by_sample(data_X_val_org, data_Y_val_org,                                                                                             filter_type=offset_sample_type, value=offset_sample)
#                                     data_X, data_Y = filter_by_sample(data_X_org, data_Y_org, 
#                                                                   filter_type='manual', value=[1, 100])  # Hardcode for offset sample test
                                
                                if filter_f is None:
                                    data_X = np.copy(data_X)
                                    data_X_val = np.copy(data_X_val)
                                else:
                                    channel_index = filter_f(num_electrode)
                                    data_X = filter_by_channel(data_X, channel_index, invert_ch=num_electrode_invert)
                                    data_X_val = filter_by_channel(data_X_val, channel_index, invert_ch=num_electrode_invert)

                                if filter_g is None:
                                    data_Y = np.copy(data_Y)
                                    data_Y_val = np.copy(data_Y_val)
                                else:
                                    filtered = filter_g(data_X, data_Y, criteria=num_class_type, quantity=[num_class], invert=num_class_invert)
                                    data_X, data_Y = filter_by_subject_n_trial(data_X, data_Y, filtered)
                                    
                                    filtered = filter_g(data_X_val, data_Y_val, criteria=num_class_type, quantity=[num_class], invert=num_class_invert)
                                    data_X_val, data_Y_val = filter_by_subject_n_trial(data_X_val, data_Y_val, filtered)

                                ###############################
                                # Start regularization testing
                                if num_electrode == 0:
                                    regularize = [False, 0, 0]
                                elif num_electrode == 1:
                                    regularize = [True, 'L1', offset_sample/1000]
                                elif num_electrode == 2:
                                    regularize = [True, 'L2', offset_sample/1000]
                                trained_model = train_eegnet(data_X, data_Y, data_X_val, data_Y_val, index_subject_to_train=index_subject_to_train, epoch=epoch, batch_size=1024, F1=F1, D=D, F2=F2,save_weights_while_training=[False,10], pre_trained_model=pre_trained_model, checkpoint_dir=CURRENT_CKPT_DIR, regularize = regularize)
                                # End regularization testing
                                ##############################
                                
                                # Train EEGNet
#                                 trained_model = train_eegnet(data_X, data_Y, data_X_val, data_Y_val, index_subject_to_train=index_subject_to_train, epoch=epoch, batch_size=1024, F1=F1, D=D, F2=F2,save_weights_while_training=[False,10], pre_trained_model=pre_trained_model, checkpoint_dir=CURRENT_CKPT_DIR)
                #                     pre_trained_model = trained_model    # Copy trained model to be pre_trained model for next training

                                # Save model
                                save_model(CURRENT_CKPT_DIR, trained_model)
                        
                                # Get Validation Accuracy by top 10 percentile
                                validation_accuracy = get_top_pct_accuracy(trained_model.history.history['val_accuracy'], pct=0.1)
                                
                                # Check test accuracy
                                if get_test_accuracy == True:
                                    average_accuracy = cross_subject_acc(CURRENT_CKPT_REL, X_test, Y_test, num_electrode=num_electrode, num_class=[num_class], class_type=num_class_type, offset_sample_type=offset_sample_type, offset_sample=offset_sample, subject=test_subject, filter_f=filter_f, filter_g=filter_g)
#                                       average_accuracy = cross_subject_acc(CURRENT_CKPT_REL, X_test, Y_test, num_electrode=num_electrode, num_class=[num_class], class_type=num_class_type, offset_sample_type='fix_length', offset_sample=offset_sample, subject=test_subject, filter_f=filter_f, filter_g=filter_g)   # Hardcode for offset sample test
                                else:
                                    average_accuracy = -1
                                    

                                # Track train history    
                                elapsed = time.time() - t
                                elapsed_min = elapsed // 60
                                elapsed_sec = elapsed %60

                                if pre_trained_model is None:
                                    pre_trained = False
                                else:
                                    pre_trained = True

                                train_history = track_train_accuracy(train_history=train_history, checkpoint_dir=CURRENT_CKPT_DIR, folder=CURRENT_CKPT_REL, run_number=run_number, pre_trained = pre_trained, pct_acc=validation_accuracy, F1=F1, F2=F2, D=D, index_subject_to_train=index_subject_to_train, num_epoch=epoch, num_electrode=num_electrode, f_class_type = num_class_type, num_class = num_class, cr_subj_acc = average_accuracy, offset_sample = offset_sample, elapsed_min=elapsed_min, elapsed_sec=elapsed_sec)
                                run_number = run_number + 1
    
    print('==================================================================')
    print(' = = ======================================================== = = ')
    print('= = = = = ============= Completed Training ============= = = = = =')
    print(' = = ======================================================== = = ')
    print('==================================================================')
    return train_history, trained_model



def weight_name_shape(model_weight):
    """
    Function to display the respective name and shape of each layer of model (tensorflow). So that can set respective parameters in plot_weight function to visualize the model.
    Input:
        model_weight: type list. List of each weight layer (i.e. trained_model.weight), where model_weight[index] is of tf.Variable type with model_weight[index].numpy() giving the weights of that model layer
    Return:
        None. Only printout
    """
    assert type(model_weight) is list, 'Input model_weight must be of type list'
    for i in range(len(model_weight)):
        assert type(model_weight[i]) is tf.python.ops.resource_variable_ops.ResourceVariable, 'Input model_weight[{}] must be of type tensorflow.python.ops.resource_variable_ops.ResourceVariable'.format(i)
        
    for i in range(len(model_weight)):
        print('Layer {}: {}'.format(i, model_weight[i].name))
        print('Shape = {}; dtype = {} \n'.format(model_weight[i].shape, model_weight[i].dtype))

def plot_weight(model_weight,layer, offset=0.1):
    """
    Function to plot and visualize the weights on each layer.
    Input:
        model_weight: type list. List of each weight layer (i.e. trained_model.weight), where model_weight[index] is of tf.Variable type with model_weight[index].numpy() giving the weights of that model layer
        layer: type int. Layer to plot and visualize
        offset: type float. Offset from plot to plot
    """
    if layer == 0:
        for i in range(model_weight[layer].shape[-1]):
            plt.plot(model_weight[layer].numpy()[0,:,0,i]+i*offset)
    elif layer == 5:
        for i in range(model_weight[layer].shape[-1]):
            plt.plot(model_weight[layer].numpy()[0,0,:,i]+i*offset)
    plt.show()