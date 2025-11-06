#To run with GPU don't forget to activate the virtual env: conda activate keras_gpu_env 
### --------- load modules -------------------#
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers, activations, optimizers, regularizers, mixed_precision
from tensorflow.keras.utils import plot_model,Sequence
from tensorflow.keras.models import load_model,Model,model_from_json
from tensorflow.keras.layers import Input,Conv2D,Dense,Flatten,Dropout,MaxPool2D,Layer,BatchNormalization,Activation #,Conv1D,MaxPool1D
from tensorflow.keras.optimizers import Adam #RMSprop,
from tensorflow.keras.callbacks import  Callback, ModelCheckpoint #EarlyStopping
from tensorflow.keras.losses import binary_crossentropy,categorical_crossentropy
import tensorflow.keras.backend as K
from tensorflow.keras.mixed_precision import LossScaleOptimizer
#from tensorflow.data.experimental import AutoShardPolicy

import gc #garbage collector
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix
import random


### --------- check GPU -------------------#
print(f"Tensorflow version: {tf.__version__}")
print(f"Keras Version: {tf.keras.__version__}")
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))



### --------- Mixed Precision Training -------------------#


# Set the policy to mixed float16
#mixed_precision.set_global_policy('mixed_float16')


### ----------- Data generator --------------------#

class CustomDataGen(Sequence):
   
   def __init__(self,gene_sim_neu,gene_sim_hs,gene_sim_tar,batch_size):#,src_neutral_train_idx,src_sweep_train_idx): 
      '''
      Initialize data generator
      
      Arguements:
        gene_sim = ImaGene object (data of images to train model)
        batch size
      '''
      self.offsetHS=10**6 #constant to separate neutral from hard sweep simulations

      #source data (comnstant Ne model) -- set so that same number of simulations in source and target domains
      ##############################################
      self.tar_data =gene_sim_tar #.data
      print("tar shape",self.tar_data.shape)
      ############################################    

      self.src_neutral =gene_sim_neu 
      self.src_HS =gene_sim_hs #[:96000,:,:,:] ##[:,:99,:,:]

      src_neu_size = self.src_neutral.shape[0]
      src_HS_size = self.src_HS.shape[0]

      print("TOTAL SIMULATIONS: ",(src_neu_size+src_HS_size))
      print("TOTAL DATA: ",(self.tar_data.shape[0]))
  
      #define neutral and sweep indices
      src_neutral_idx = np.arange(self.src_neutral.shape[0])
      src_HS_idx = np.arange(self.src_HS.shape[0])
      tar_data_idx = np.arange(self.tar_data.shape[0])

      self.src_idx_map = np.concatenate((src_neutral_idx,src_HS_idx+self.offsetHS))
      self.tar_idx_map = tar_data_idx 

      #define batch size     
      src_size=len(self.src_idx_map) #number of samples 
      tar_size=len(self.tar_idx_map)

      #undersample
      self.batch_size = batch_size
      self.no_batch = np.minimum(src_size, tar_size) // self.batch_size # model sees training sample at most once per epoch

      # get index for source (classifier and discriminator) and target (classifier) data
      self.src_classifier_idx = np.arange(src_size)
      self.src_discr_idx = np.arange(src_size)
      self.tar_discr_idx = np.arange(tar_size)

      #shuffle idx
      np.random.shuffle(self.src_classifier_idx)
      np.random.shuffle(self.src_discr_idx)
      np.random.shuffle(self.tar_discr_idx)

   
   def __len__(self):
      return self.no_batch #returns number of batches in data
   
   def __getitem__(self,idx):
      'generate batch'

      # CLASSIFIER
      # indices for source classifier 
      # All classifier data comes from source (constant Ne model)
      classifier_batch_idx = self.src_classifier_idx[idx*self.batch_size:(idx+1)*self.batch_size]
      classifier_batch_data = self.src_idx_map[classifier_batch_idx] # useful to separate data into neutral and sweep

      classifier_neutral_idx = classifier_batch_data[classifier_batch_data < self.offsetHS]
      discr_src_HS_idx = discr_src_batch_data[discr_src_batch_data >= self.offsetHS]-self.offsetHS
      # DISCRIMINATOR
      # half of data from discriminator comes from source (constant Ne) and half from target (bottleneck)
      # indices for source discriminator
      discr_src_batch_idx = self.src_discr_idx[idx*(self.batch_size//2):(idx+1)*(self.batch_size//2)]
      discr_src_batch_data = self.src_idx_map[discr_src_batch_idx]

      discr_src_neutral_idx = discr_src_batch_data[discr_src_batch_data < self.offsetHS]
      discr_src_HS_idx = discr_src_batch_data[discr_src_batch_data >= self.offsetHS]-self.offsetHS
  
      #indices for target data
      discr_tar_batch_idx = self.tar_discr_idx[idx*(self.batch_size//2):(idx+1)*(self.batch_size//2)] 
      discr_tar_batch_data_idx = self.tar_idx_map[discr_tar_batch_idx]


      # create batch X
      empty_arr=np.empty((0, *self.src_neutral.shape[1:]))
      #print(self.src_neutral.shape[1:])
      X_class_neutral=self.src_neutral[classifier_neutral_idx] if len(classifier_neutral_idx) > 0 else empty_arr
      X_disrc_src_neutral=self.src_neutral[discr_src_neutral_idx] if len(discr_src_neutral_idx) > 0 else empty_arr

      X_class_HS=self.src_HS[classifier_HS_idx] if len(classifier_HS_idx) > 0 else empty_arr
      X_disrc_src_HS= self.src_HS[discr_src_HS_idx] if len(discr_src_HS_idx) > 0 else empty_arr

      X_disrc_tar = self.tar_data[discr_tar_batch_data_idx]

        
      batch_X=np.concatenate((X_class_neutral,X_class_HS,
                              X_disrc_src_neutral, X_disrc_src_HS,
                              X_disrc_tar))
      
      
      #create output Y for batch
      neutral_label = np.array([0.0])
      HS_label = np.array([1.0])

      #It is possible that for some categories (neutral/hard/soft) I have an empty list -- so double check and account for empty lists
      neutral_class_vstack= np.vstack([neutral_label]*len(classifier_neutral_idx)) if len(classifier_neutral_idx) > 0 else np.empty((0, neutral_label.shape[0]))
      neutral_discr_src_vstack= -1 * np.vstack([neutral_label] * len(discr_src_neutral_idx)) if len(discr_src_neutral_idx) > 0 else np.empty((0, neutral_label.shape[0]))
      
      hs_class_vstack = np.vstack([HS_label]*len(classifier_HS_idx)) if len(classifier_HS_idx) > 0 else np.empty((0, HS_label.shape[0]))
      hs_discr_src_vstack = -1 * np.vstack([HS_label] * len(discr_src_HS_idx)) if len(discr_src_HS_idx) > 0 else np.empty((0, HS_label.shape[0]))
      
      discr_tar_vstack = -1 * np.vstack([neutral_label] * len(discr_tar_batch_data_idx))

      batch_Y_classifier = np.concatenate((neutral_class_vstack,hs_class_vstack,
                                           neutral_discr_src_vstack,hs_discr_src_vstack,
                                           discr_tar_vstack))
      
      batch_Y_discr = np.concatenate((-1*np.ones(len(classifier_neutral_idx)),-1*np.ones(len(classifier_HS_idx)),
                                            np.zeros(len(discr_src_neutral_idx)),np.zeros(len(discr_src_HS_idx)),
                                            np.ones(len(discr_tar_batch_data_idx))))


      assert batch_X.shape[0] == self.batch_size*2, batch_X.shape[0]
      assert batch_Y_classifier.shape[0] == batch_Y_discr.shape[0], (batch_Y_classifier, batch_Y_discr)

      return batch_X,{"classifier": batch_Y_classifier, "discriminator": batch_Y_discr}

   
   def on_epoch_end(self):
      'Updates (shuffles) indexes after each epoch --useful so that batches between epochs do no look alike'

      np.random.shuffle(self.src_classifier_idx)
      np.random.shuffle(self.src_discr_idx)
      np.random.shuffle(self.tar_discr_idx)
      gc.collect()


##############################################
### ------- custom functions ---------------##
##############################################

### ------ Gradient reversal layer --------#

@tf.custom_gradient
def grad_reverse(x):
    y = tf.identity(x)
    def custom_grad(dy):
        return -dy
    return y, custom_grad

class GradReverse(Layer):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, x):
        return grad_reverse(x)

##### ---------- Loss weights ----------------
class LossWeightsScheduler(Callback):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
    
    def on_epoch_end(self, epoch, logs={}):
        gamma=10 #10
        p=epoch/30
        lambda_new=2/(1+math.exp(-gamma*p))-1
        K.set_value(self.beta,lambda_new)
        #K.set_value(self.alpha,lambda_alpha)
        '''
        if epoch >= 25:
            K.set_value(self.alpha,0.5)
            K.set_value(self.beta,1.0)
        '''
class LossWeightsLogger(Callback):
    def __init__(self, loss_weights):
        super(LossWeightsLogger, self).__init__()
        self.loss_weights = loss_weights

    def on_epoch_end(self, epoch, logs=None):
        print("Loss Weights:", self.loss_weights)

####------------Loss and accuracy ------------
def custom_bce(y_true,y_pred):
    """
    Custom binary crossentropy loss. When label is -1 (target domain--bottleneck) the observation is 
    masked out for the task (hidden from classifier/discriminator).
    This way some observations (source-constant Ne) impact loss backpropagation for one of the two tasks (classification).

    Keyword Arguments:
    y_true = true value
    y_pred = predicted value

    Return:
    binary crossentropy
    """

    #extracts elements from y_pred where the corresponding element in y_true is not equal to 1
    #tf.print("y_true:", y_true)

    y_pred = tf.boolean_mask(y_pred,tf.not_equal(y_true,-1))  #-1 will be masked/ y_true or y_pred?
    y_true = tf.boolean_mask(y_true, tf.not_equal(y_true,-1))

    #tf.print("y_true:", y_true)
   
    return binary_crossentropy(y_true,y_pred)

def custom_categorical_ce(y_true,y_pred):
    """
    Custom categorical crossentropy loss. When label array contains -1 (ex: [-1,0,0]) the observation is 
    masked out for the task (hidden from classifier). 
    If any element of the one-hot encoded label is -1, the entire label is masked out.
    This way some observations (source-constant Ne) impact loss backpropagation for one of the two tasks (classification).

    Keyword Arguments:
    y_true = true value
    y_pred = predicted value

    Return:
    binary crossentropy
    """
    #extracts elements from y_pred where the corresponding element in y_true is not equal to -1
    y_pred = tf.boolean_mask(y_pred,tf.reduce_all(tf.not_equal(y_true, -1), axis=-1))  #-1 will be masked
    y_true = tf.boolean_mask(y_true, tf.reduce_all(tf.not_equal(y_true, -1), axis=-1))
   
    return categorical_crossentropy(y_true,y_pred)


def custom_binary_accuracy(y_true,y_pred):
     y_pred = tf.boolean_mask(y_pred,tf.not_equal(y_true,-1))  #-1 will be masked/ y_true or y_pred?
     y_true = tf.boolean_mask(y_true, tf.not_equal(y_true,-1))
     return keras.metrics.binary_accuracy(y_true, y_pred)


def custom_categorical_accuracy(y_true,y_pred):
     """
     if any element of the one-hot encoded label is -1, the entire label is masked out. 
     """
     y_pred = tf.boolean_mask(y_pred, tf.reduce_all(tf.not_equal(y_true, -1), axis=-1)) #-1 will be masked/ y_true or y_pred?
     y_true =  tf.boolean_mask(y_true, tf.reduce_all(tf.not_equal(y_true, -1), axis=-1))
     return keras.metrics.categorical_accuracy(y_true, y_pred)


### -------------- model ------------------------#
def create_model(mmap_sim,loss_weights):
    """
    create CNN model 

    Keyword Arguments:
        gene_sim (object) -- ImaGene object with simulations (images)

    Return:
        Model object 
    """
    #with strategy.scope():
    input_shape = mmap_sim.shape[1:] 
    print("input shape:", input_shape)

    # Input layer
    input_tensor = Input(shape=input_shape)

    # Convolutional layers
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(input_tensor) #kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.0001), kernel_regularizer=regularizers.l1_l2(l1=0.005, l2=0.005)
    pool1 = MaxPool2D(pool_size=(2, 2), padding='valid')(conv1)
    #pool1 = Dropout(0.15)(pool1)

    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(pool1)  #128
    pool2 = MaxPool2D(pool_size=(2, 2), padding='valid')(conv2)
    #pool2 = Dropout(0.15)(pool2)

    # Flatten the output of the last convolutional layer
    flatten = Flatten(name="feature_extractor")(pool2)

    # Dense layer #1
    dense_branch1_layer1 = Dense(128)(flatten) #,use_bias=False
    #dense_branch1_layer1 = BatchNormalization()(dense_branch1_layer1)
    dense_branch1_layer1 = Activation('relu')(dense_branch1_layer1)
    dense_branch1_layer1 = Dropout(0.5)(dense_branch1_layer1)

    # Dense layer #2
    dense_branch1_layer2 = Dense(128)(dense_branch1_layer1)
    dense_branch1_layer2 = Activation('relu')(dense_branch1_layer2)
    dense_branch1_layer2 = Dropout(0.5, name="last_hidden_layer_classifier")(dense_branch1_layer2)

    # Output layer
    output_branch1 = Dense(1, activation='sigmoid', name='classifier',dtype='float32')(dense_branch1_layer2) #sigmoid activation for binary classifucation
    # Second branch / GRL layer to discriminate domains
    GRL_branch2 = GradReverse()(flatten)
    dense_branch2_layer1 = Dense(128)(GRL_branch2)
    dense_branch2_layer1 = Activation('relu')(dense_branch2_layer1)
    dense_branch2_layer1 = Dropout(0.25)(dense_branch2_layer1)

    dense_branch2_layer2 = Dense(128)(dense_branch2_layer1)
    dense_branch2_layer2 = Activation('relu',name="last_hidden_layer_discriminator")(dense_branch2_layer2)
    dense_branch2_layer2 = Dropout(0.25)(dense_branch2_layer2)

    output_branch2 = Dense(1, activation='sigmoid', name='discriminator',dtype='float32')(dense_branch2_layer2) #sigmoid activation for binary ourput from discriminator branch

    # Create the model

    model = Model(inputs=input_tensor, outputs=[output_branch1, output_branch2])

    optimizer =  Adam(learning_rate=1e-5) #0.0005, 0.001 clipnorm=1.0
    #optimizer = mixed_precision.LossScaleOptimizer(optimizer)  
    #optimizer = RMSprop(learning_rate=0.001)# Compile the model

    alpha = K.variable(loss_weights[0])
    beta = K.variable(loss_weights[1])

    #change loss finction according to problem (binary or multiclass classification)


    model.compile(optimizer=optimizer,loss={'classifier':custom_bce,'discriminator':custom_bce }, #custom_bce
                  loss_weights = [alpha,beta],#equal loss weights I have to change this if losses are not on same scale so that they have equal weights (i.e binary versus multiclas loss)
                  metrics={'classifier': custom_binary_accuracy,'discriminator': custom_binary_accuracy}) 


    return model

def intermediate_layer_model(model,layer_name):
    return  Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

def train_model(model,gene_sim_neu,gene_sim_hs,gene_sim_traintar,val_split,batch_size,loss_weights,path):
    """
    train CNN model 

    Keyword Arguments:
        gene_sim (object) -- ImaGene object with simulations (images)
        model (object) -- Keras CNN model

    Return:
        History object from training process
    """

    '''
    gene_sim_neu.data =gene_sim_neu.data#[:,:99,:,:]
    gene_sim_hs.data =gene_sim_hs.data#[:,:99,:,:]
    gene_sim_ss.data =gene_sim_ss.data#[:,:99,:,:]
    gene_sim_traintar.data =gene_sim_traintar.data#[:,:99,:,:]


    gene_sim_neu.data[gene_sim_neu.data==0] = -1
    gene_sim_hs.data[gene_sim_hs.data==0] = -1
    gene_sim_ss.data[gene_sim_ss.data==0] = -1
    gene_sim_traintar.data[gene_sim_traintar.data==0] = -1
    
    # substitute nan's to 0's
    gene_sim_neu.data[np.isnan(gene_sim_neu.data)] = 0
    gene_sim_hs.data[np.isnan(gene_sim_hs.data)] = 0
    gene_sim_ss.data[np.isnan(gene_sim_ss.data)] = 0
    gene_sim_traintar.data[np.isnan(gene_sim_traintar.data)] = 0

    print(gene_sim_neu.data[1][20:30,30:40,0])
    print(gene_sim_traintar.data[1][20:30,30:40,0])
    '''
    #no validation data
    data_gen=CustomDataGen(gene_sim_neu=gene_sim_neu,gene_sim_hs=gene_sim_hs,gene_sim_tar = gene_sim_traintar,batch_size=batch_size)
    

    # ----------------------------------------------------------------
    
    checkpoint_filepath = path+'.{epoch:02d}.weights.h5' #hdf5

    checkpoint_weights = ModelCheckpoint(checkpoint_filepath,save_weights_only=True,save_freq='epoch')

    alpha = K.variable(loss_weights[0])
    beta = K.variable(loss_weights[1])

    save_model_architecture(model,path)
    print("START TRAINING")
    with tf.device('/GPU:0'):
        score = model.fit(data_gen, \
                      batch_size=batch_size, epochs=30, verbose=1,#use_multiprocessing=False,#steps_per_epoch=steps_per_epoch,
                      callbacks=[checkpoint_weights,LossWeightsScheduler(alpha,beta)])

    #save_model_architecture(model,path)
    save_trainedModel(model,path)

    # save preformance figures
    plot_accuracy(score)
    plot_loss(score)
    
    #save training data

    loss = score.history['loss']
    accuracy_class = score.history['classifier_custom_binary_accuracy']
    accuracy_discr = score.history['discriminator_custom_binary_accuracy']

    loss_class = score.history['classifier_loss']
    loss_discr = score.history['discriminator_loss']

    # Save to a text file
    with open('training_multiclass_results.txt', 'w') as f:
        f.write('Epoch,Loss,class_accuracy,discr_accuracy,class_loss,discr_loss\n')
        for epoch in range(len(loss)):
            f.write(f"{epoch + 1},{loss[epoch]},{accuracy_class[epoch]},{accuracy_discr[epoch]},{loss_class[epoch]},{loss_discr[epoch]}\n")

    return model,score


def save_model_architecture(model,path):
    model_json = model.to_json()
    json_file_name=path+"_model.json"
    with open(json_file_name, "w") as json_file:
        json_file.write(model_json)
    
    return 0

def save_trainedModel(model,path):
    model_json = model.to_json()
    json_file_name=path+"_model.json"
    weights_file_name=path+".weights.h5"
    with open(json_file_name, "w") as json_file:
        json_file.write(model_json)
    
    model.save_weights(weights_file_name)
    
    return 0


def load_grl_model_lambdaScheduler(path):
    optimizer =  Adam(learning_rate=1e-5) #0.0005
    # Register GradReverse for serialization  
    tf.keras.utils.get_custom_objects()['GradReverse'] = GradReverse
    tf.keras.utils.get_custom_objects()['custom_bce'] = custom_bce
    tf.keras.utils.get_custom_objects()['custom_categorical_ce'] = custom_categorical_ce
    tf.keras.utils.get_custom_objects()['custom_categorical_accuracy'] = custom_categorical_accuracy
    tf.keras.utils.get_custom_objects()['custom_binary_accuracy'] = custom_binary_accuracy
    
    json_file_name=path+"_model.json"
    weights_file_name=path+".weights.h5"
    print(json_file_name)
    print(weights_file_name)
    with open(json_file_name, "r") as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(weights_file_name)

    loaded_model.compile(optimizer=optimizer,loss={'classifier':custom_bce,'discriminator':custom_bce }, #custom_bce
                        metrics={'classifier': custom_binary_accuracy,'discriminator': custom_binary_accuracy})

    return loaded_model

def load_grl_model(path):
    # Register GradReverse for serialization  
    tf.keras.utils.get_custom_objects()['GradReverse'] = GradReverse
    tf.keras.utils.get_custom_objects()['custom_bce'] = custom_bce
    tf.keras.utils.get_custom_objects()['custom_categorical_ce'] = custom_categorical_ce
    tf.keras.utils.get_custom_objects()['custom_categorical_accuracy'] = custom_categorical_accuracy
    tf.keras.utils.get_custom_objects()['custom_binary_accuracy'] = custom_binary_accuracy
    #print(tf.keras.utils.get_custom_objects())
    return load_model(path,custom_objects={'GradReverse': GradReverse,'custom_bce': custom_bce,'custom_categorical_ce':custom_categorical_ce ,'custom_accuracy':custom_categorical_accuracy,'custom_binary_accuracy': custom_binary_accuracy})

def load_cnn_model_weights(path_model,path_weights):
    # Register GradReverse for serialization  
    tf.keras.utils.get_custom_objects()['GradReverse'] = GradReverse
    tf.keras.utils.get_custom_objects()['custom_categorical_ce'] = custom_categorical_ce
    tf.keras.utils.get_custom_objects()['custom_categorical_accuracy'] = custom_categorical_accuracy
    # Load model architecture from JSON file
    with open(path_model, 'r') as f:
      model = model_from_json(f.read(), custom_objects={'GradReverse': GradReverse,'custom_categorical_ce':custom_categorical_ce,'custom_accuracy':custom_categorical_accuracy})
    # Load model weights from HDF5 file
    model.load_weights(path_weights)
    #print(tf.keras.utils.get_custom_objects())
    return model

def plot_accuracy(score):
    # summarize history for accuracy
    #classifier accuracy
    print(score.history['classifier_custom_binary_accuracy'])
    plt.plot(score.history['classifier_custom_binary_accuracy'])
    plt.title('classifier accuracy')
    plt.ylabel('classifier accuracy')
    plt.xlabel('epoch')
    #plt.legend(['train', 'validation'], loc='center right')
    plt.savefig('classifier_accuracy_hist.png')
    plt.close()

    #discriminator accuracy
    plt.plot(score.history['discriminator_custom_binary_accuracy'])
    #plt.plot(score.history['val_discriminator_custom_binary_accuracy'])
    plt.ylim(min(score.history['discriminator_custom_binary_accuracy']), max(score.history['discriminator_custom_binary_accuracy']))
    plt.title('discriminator accuracy')
    plt.ylabel('discriminator accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='center right')
    plt.savefig('discriminator_accuracy_hist.png')
    plt.close()


def plot_loss(score):
    # classifier loss
    plt.plot(score.history['classifier_loss'])
    #plt.plot(score.history['val_classifier_loss'])
    plt.ylim(min(score.history['classifier_loss']), max(score.history['classifier_loss']))
    plt.title('classifier loss')
    plt.ylabel('classifier loss')
    plt.xlabel('epoch')
    #plt.legend(['train', 'validation'], loc='center right')
    plt.savefig('classifier_loss_hist.png')
    plt.close()

    # discriminator loss
    plt.plot(score.history['discriminator_loss'])
    #plt.plot(score.history['val_discriminator_loss'])
    plt.ylim(min(score.history['discriminator_loss']), max(score.history['discriminator_loss']))
    plt.title('discriminator  loss')
    plt.ylabel('discriminator  loss')
    plt.xlabel('epoch')
    #plt.legend(['train', 'validation'], loc='center right')
    plt.savefig('discriminator_loss_hist.png')
    plt.close()

def confusionMatrix(gene_sim_test,y_pred,output_names): 
    #print(gene_sim_test.targets_classifier)   
    output_predictions = {}
    for i in range(len(output_names)):
      output_predictions[output_names[i]] = y_pred[i]

    y_test_labels = np.argmax(gene_sim_test.targets_classifier, axis=1)

    print(np.unique(y_test_labels))
    print(np.unique(np.argmax(output_predictions['classifier'],axis=1)))

    cm = confusion_matrix(y_test_labels, np.argmax(output_predictions['classifier'],axis=1))
    

    accuracy = np.trace(cm) / float(np.sum(cm))
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(gene_sim_test.classes_classifier))
    plt.xticks(tick_marks, ["neu", "hard", "soft"], fontsize=8)
    plt.yticks(tick_marks, ["neu", "hard", "soft"], fontsize=8)
    thresh = cm.max() / 1.5
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}'.format(accuracy))
    plt.tight_layout()
    # Add labels to the confusion matrix plot
    for i in range(len(gene_sim_test.classes_classifier)):
        for j in range(len(gene_sim_test.classes_classifier)):
            plt.text(j, i, format(cm[i, j], '.2f'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=8)
            
    plt.savefig('ConfusionMatrix.png')
    plt.close()


def precision_recall(gene_sim_test,y_pred,output_names):
    # add name to each oput branch
    output_predictions = {}
    for i in range(len(output_names)):
      output_predictions[output_names[i]] = y_pred[i]

    #print(gene_sim_test.targets_classifier)
    #print(output_predictions['classifier'])
    #compute precision and recall values

    precision, recall, _ = precision_recall_curve(gene_sim_test.targets_classifier, output_predictions['classifier'])
    # Calculate AUC-PRC
    #print(precision)
    #print(recall)
    auc_prc = auc(recall, precision)
    print("AUPRC:",auc_prc)
    # Plot the precision-recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'AUC-PRC = {auc_prc:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Classifier)')
    plt.legend()
    plt.savefig('auprc.png')
    plt.close()

