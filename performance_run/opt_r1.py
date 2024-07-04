print("_"*100)
'''
# # # imports:
# # # tensorflow(ML framework)
# # # matplotlib(graph plot)
# # # time, datetime
# # # tqdm(progressbar)

# # # nvidia smi
# # # >gpu profiling and statistics(memory usage, core frequency, temperature ....)

# # # gc garbage collection
'''
import tensorflow as tf
import matplotlib.pyplot as plt
#pip install nvidia-ml-py3
#import nvidia_smi
import time
from tqdm.auto import tqdm

from tensorflow.python.framework.ops import disable_eager_execution
import gc

import datetime

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  print("Invalid device or cannot modify virtual devices once initialized.")

# # #disable_eager_execution() # read git issue https://github.com/tensorflow/tensorflow/issues/27472

from tensorflow.python.framework import ops#tensor operations
from tensorflow.python.keras import backend_config#configure runtime

from tensorflow.python.keras.optimizer_v2 import optimizer_v2#optimizer interface

from tensorflow.python.ops import array_ops#tensorflow array operations
from tensorflow.python.ops import control_flow_ops#tensorflow graph control flow operatinos
from tensorflow.python.ops import math_ops#tf math operations
from tensorflow.python.ops import state_ops#tf state change operations

from tensorflow.python.util.tf_export import keras_export
import_from_colab="from keras.optimizer_v2 import optimizer_v2"#use if trying on colab
'''
# # # import model layers and necessary libraries to create and train it
'''
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D,\
     Flatten, BatchNormalization, AveragePooling2D, Dense, Activation, Add, Lambda, Dropout, LeakyReLU 
from tensorflow.keras.models import Model
from tensorflow.keras import activations
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

'''
# # # argument parser to run on comandline
# # # parser.add_argument("--optimizer", "-p", type=int, help="set optimizer [0-5]")
# # # comandline call "--optimizer" or "-p" and type in arguement 
'''
import getopt, sys
import argparse
from ast import literal_eval as make_tuple
# # #commandline options
parser = argparse.ArgumentParser(prefix_chars='-+')
# # #i_opt=4,batch_s=128,epoch_s=100,test_diffgrads=False
# # #Dset_norm=True,Dset_aug=True
# # # Add long and short argument
parser.add_argument("--optimizer", "-p", type=int, help="set optimizer [0-5]")
# # #select optimizer 0 through 5 from list of optimizers or version of diffgrad based on selection of use diffgrad true or false
parser.add_argument("--batch_size", "-b", type=int, help="set batch size (32,64,128)",default=128)
# # #set batchsize 32, 64, 128
parser.add_argument("--actiavtion", "-a", help="set activation one of(relu,lrelu,selu,elu)",default="relu")
# # #set activation function from relu, lrelu, selu, elu
parser.add_argument("--epochs", "-e", type=int, help="set number of epochs",default=50)
# # #set number of epochs used 100
parser.add_argument("--dataset", "-d", type=int, choices=[10,100], help="set dataset  [10,100] cifar")
# # #set dataset 10 for cifar10 and 100 for cifar100
parser.add_argument("--validation_target", "-v", type=float, help="set validation target",default=0.955)
# # #stop training early if validation already reached
parser.add_argument("--test_diffgrads","-i", action="store_false", help="use other optimizers",default=False)
# # #use other optimizers set 0 to 5 for optimizer list
parser.add_argument("++test_diffgrads","+i", action="store_true", help="use diffgrad optimizers",default=False)
# # #use diffgrads set 0 to 5 for diffgrad version
parser.add_argument("--normalize_dataset","-N", action="store_false", help="normalize dataset",default=True)
# # #dont apply normalization in images  
parser.add_argument("++normalize_dataset","+N", action="store_true", help="normalize dataset",default=True)
# # #apply normalization in images
parser.add_argument("--augument_dataset","-G", action="store_false", help="augument dataset",default=True)
# # #dont augument images
parser.add_argument("++augument_dataset","+G", action="store_true", help="augument dataset",default=True)
# # #augument images
parser.add_argument("--remake_dataset","-r", action="store_false", help="remake dataset",default=False)
# # #dont force preprocess dataset again or overwrite previous save
parser.add_argument("++remake_dataset","+r", action="store_true", help="remake dataset",default=False)
# # #force preprocess dataset again and overwrite previous save
# # #commandline options
# # # Read arguments from the command line
args = parser.parse_args()
# # #<variables to get args>
i_opt=4
batch_s=128
epoch_s=100
test_diffgrads=args.test_diffgrads
Dset_norm=args.normalize_dataset
Dset_aug=args.augument_dataset
act="relu"
remake_dset=args.remake_dataset
Dset=100
val_t=0.755
# # #</variables to get args>
# # #<check args>
try:
    i_opt=int(args.optimizer)
except:
    i_opt=0
if args.optimizer:
    aw=int(args.optimizer)
    i_opt=aw
    print("Set input shape to",aw )
if args.batch_size:
    aw=int(args.batch_size)
    if (aw<1):
        print("invalid number of epochs",aw)
        sys.exit()
    if aw not in [32,64,128]:
        print("non standard batch size",aw)
    batch_s=aw
    print("Set input shape to",aw )
if args.actiavtion:
    aw=args.actiavtion
    act=aw
    print("Set input shape to",aw )
if args.epochs :
    aw=int(args.epochs)
    #print("aw")
    if (aw<1):
        print("invalid number of epochs",aw)
        sys.exit()
    epoch_s=aw
    print("Set number of epochs to",aw )
if args.dataset:
    aw=int(args.dataset)
    Dset=aw
    print("Set dataset to cifar",aw )
if args.validation_target:
    aw=float(args.validation_target)
    val_t=aw
    print("Set validation target",aw )
# # #</check args>
print("\t\t -------------inputs------------- ")
print("\t","[i_opt",i_opt,"][batch_s",batch_s,"][epoch_s",epoch_s,"][test_diffgrads",test_diffgrads,"][act",act,"][Dset",Dset,"][Dset_norm",Dset_norm,"][Dset_aug",Dset_aug,"][validation_target",val_t,"]")
print("\t\t -------------inputs------------- ")
#print(i_opt,batch_s,epoch_s,test_diffgrads,act,Dset,Dset_norm,Dset_aug)#testing print
# # #tensorboard log dir
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#tf.debugging.experimental.enable_dump_debug_info(log_dir, tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)

# # #resnet implementation without any imorivements given by keras and others from paper in tf
# # #values for stride and layer size implemented as suggested and other parameters if not present taken from resnet50 suggestions
# # #uncomment code to use layer or comment to remove from block maintain block naming scheme
def res_bottleneck(g1, filters,act="relu",stri=1,block_id="0_0"): 
    # # #dimension does not change.
    # # #The skip connection is an identity conncection

    g1_skip = g1 # this will be used for addition with the residual block 
    f1, f2 = filters

    # # #first block 
    g1 = Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001), name=block_id+"_C2D_1")(g1)
    g1 = BatchNormalization(name=block_id+"_BN_1")(g1)
    if act=="relu" :
        g1 = Activation(activations.relu, name=block_id+"_Act_1_relu")(g1)
    if act=="lrelu" :
        g1 = LeakyReLU( name=block_id+"_Act_1_lrelu")(g1)
    if act=="selu" :
        g1 = Activation(activations.selu, name=block_id+"_Act_1_selu")(g1)
    if act=="elu" :
        g1 = Activation(activations.elu, name=block_id+"_Act_1_elu")(g1)

    # # #second block #bottleneck (size kept same with padding)
    g1 = Conv2D(f1, kernel_size=(3, 3), strides=(stri, stri), padding='same', kernel_regularizer=l2(0.001), name=block_id+"_C2D_2")(g1)
    g1 = BatchNormalization(name=block_id+"_BN_2")(g1)
    if act=="relu" :
        g1 = Activation(activations.relu, name=block_id+"_Act_2_relu")(g1)
    if act=="lrelu" :
        g1 = LeakyReLU( name=block_id+"_Act_2_lrelu")(g1)
    if act=="selu" :
        g1 = Activation(activations.selu, name=block_id+"_Act_2_selu")(g1)
    if act=="elu" :
        g1 = Activation(activations.elu, name=block_id+"_Act_2_elu")(g1)

    # # # third block activation used after adding the input
    g1 = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001), name=block_id+"_C2D_3")(g1)
    g1 = BatchNormalization(name=block_id+"_BN_3")(g1)
    
    g1_skip = Conv2D(f2, kernel_size=(1, 1), strides=(stri, stri), padding='valid', kernel_regularizer=l2(0.001), name=block_id+"_C2D_skip")(g1_skip)
    g1_skip = BatchNormalization(name=block_id+"_BatchNorm_skip")(g1_skip)
    
    # # # add the input 
    g1 = Add(name=block_id+"_Add")([g1, g1_skip])
    if act=="relu" :
        g1 = Activation(activations.relu, name=block_id+"_Activation_Add_relu")(g1)
    if act=="lrelu" :
        g1 = LeakyReLU( name=block_id+"_Activation_Add_lrelu")(g1)
    if act=="selu" :
        g1 = Activation(activations.selu, name=block_id+"_Activation_Add_selu")(g1)
    if act=="elu" :
        g1 = Activation(activations.elu, name=block_id+"_Activation_Add_elu")(g1)
    
    return g1

def res_conv(g1, s, filters):
    g1_skip = g1
    f1, f2 = filters

    # # # first block
    g1 = Conv2D(f1, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=l2(0.001))(g1)
    g1 = BatchNormalization()(g1)
    g1 = Activation(activations.relu)(g1)

    # # # second block
    g1 = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001))(g1)
    g1 = BatchNormalization()(g1)
    g1 = Activation(activations.relu)(g1)

    # # # third block
    g1 = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(g1)
    g1 = BatchNormalization()(g1)

    # # # shortcut 
    g1_skip = Conv2D(f2, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=l2(0.001))(g1_skip)
    g1_skip = BatchNormalization()(g1_skip)

    # # # add 
    g1 = Add()([g1, g1_skip])
    g1 = Activation(activations.relu)(g1)

    return 
# # # resnet50 model 
def c_ResNet50(in_sh=(32,32,3),num_classes=10,act="relu"):

    #input_im = Input(shape=(train_im.shape[1], train_im.shape[2], train_im.shape[3])) # cifar 10 images size
    input_im = Input(shape=in_sh) # cifar 10 images size
    #g1 = ZeroPadding2D(padding=(3, 3))(input_im) # zeropad performed in augemnt 
    #g1 = ZeroPadding2D(padding=(4, 4), name="ZPad_4x4")(input_im) # zeropad performed in augemnt 

    # # # 1st stage
    # # # here we perform maxpooling, see the figure above

    #g1 = Conv2D(64, kernel_size=(3, 3), strides=(1,1), name="CD_0")(g1)#different initial stride
    
    g1 = Conv2D(64, kernel_size=(4, 4), strides=(1,1), name="CD_0")(input_im)
    g1 = BatchNormalization(name="BN_0")(g1)
    
    if act=="relu":
        g1 = Activation(activations.relu, name="Act_0_relu")(g1)
    g1 = MaxPooling2D((3, 3), strides=(2, 2))(g1)
    
    # g1 = Conv2D(64, kernel_size=(3, 3), strides=(1,1), name="CD_1")(g1)
    # g1 = BatchNormalization(name="BN_1")(g1)
    # if act=="relu":
        # g1 = Activation(activations.relu, name="Act_1_relu")(g1)
    # if act=="lrelu" :
        # g1 = LeakyReLU( name=block_id+"_Act_1_lrelu")(g1)
    # if act=="selu" :
        # g1 = Activation(activations.selu, name=block_id+"_Act_1_selu")(g1)
    # if act=="elu" :
        # g1 = Activation(activations.elu, name=block_id+"_Act_1_elu")(g1)
    # g1 = MaxPooling2D((3, 3), strides=(2, 2))(g1)

    # # #2nd stage 
    # # # frm here on only conv block and identity block, no pooling

    g1 = res_bottleneck(g1, filters=(64, 256), act=act, block_id="S1_B1")
    g1 = res_bottleneck(g1, filters=(64, 256), act=act, block_id="S1_B2")
    g1 = res_bottleneck(g1, filters=(64, 256), act=act, block_id="S1_B3")

    # # # 3rd stage

    g1 = res_bottleneck(g1, filters=(128, 512), act=act,stri=2, block_id="S2_B1")
    g1 = res_bottleneck(g1, filters=(128, 512), act=act, block_id="S2_B2")
    g1 = res_bottleneck(g1, filters=(128, 512), act=act, block_id="S2_B3")
    g1 = res_bottleneck(g1, filters=(128, 512), act=act, block_id="S2_B4")

    # # # 4th stage

    g1 = res_bottleneck(g1, filters=(256, 1024), act=act,stri=2, block_id="S3_B1")
    g1 = res_bottleneck(g1, filters=(256, 1024), act=act, block_id="S3_B2")
    g1 = res_bottleneck(g1, filters=(256, 1024), act=act, block_id="S3_B3")
    g1 = res_bottleneck(g1, filters=(256, 1024), act=act, block_id="S3_B4")
    g1 = res_bottleneck(g1, filters=(256, 1024), act=act, block_id="S3_B5")
    g1 = res_bottleneck(g1, filters=(256, 1024), act=act, block_id="S3_B6")

    # # # 5th stage

    g1 = res_bottleneck(g1, filters=(512, 2048), act=act,stri=2, block_id="S4_B1")
    g1 = res_bottleneck(g1, filters=(512, 2048), act=act, block_id="S4_B2")
    g1 = res_bottleneck(g1, filters=(512, 2048), act=act, block_id="S4_B3")

    # # # ends with average pooling and dense connection

    g1 = AveragePooling2D((4, 4), padding='same', name="AvgPool2D_4x4")(g1)

    g1 = Flatten(name="Flatten_1")(g1)
    #g1 = Dense(len(class_types), activation='softmax', kernel_initializer='he_normal')(g1) #multi-class
    #g1 = Dense(512, activation='softmax', kernel_initializer='he_normal', name="Dense_pred1")(g1)
    g1 = Dense(num_classes, activation='softmax', kernel_initializer='he_normal', name="Dense_pred")(g1) #multi-class
    g1=tf.keras.layers.Activation(activation='sigmoid')(g1)
    # # # define the model 

    model = Model(inputs=input_im, outputs=g1, name='Resnet50')

    return model


#----------------------diffgrad----------------------#
@keras_export('keras.optimizers.Diffgrad')
class DiffGrad(optimizer_v2.OptimizerV2):
    def __init__(self,
                 learning_rate=0.1,
                 beta_1=0.95,
                 beta_2=0.999,
                 epsilon=1e-7,
                 name='DiffGrad',
                 diff_version=0,
                 **kwargs):
        super(DiffGrad, self).__init__(name, **kwargs)
        # # # constructor
        self.d_version = diff_version
        if self.d_version<0 or self.d_version>5:
          raise RuntimeError("No diffGrad version ",self.d_version)
        # # # set hyper parameters
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._set_hyper('decay', self._initial_decay)
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self.epsilon = epsilon or backend_config.epsilon()

    def _create_slots(self, var_list):
        # # # Create slots for the first and second moments.
        # # # Create slots for previous g (gradient)
        # # # Create slots for sum of g and sum of g square
        # # # Separate for-loops to respect the ordering of slot variables from v1.
        # # # See if slot can be controlled by version for sig_1 and sig_2
        for var in var_list:
            self.add_slot(var,'m', initializer='zeros')
        for var in var_list:
            self.add_slot(var,'v', initializer='zeros')
        for var in var_list:
            self.add_slot(var,'prev_g')
        if self.d_version>=3:
          for var in var_list:
              self.add_slot(var,'sig_1', initializer='zeros')
          for var in var_list:
              self.add_slot(var,'sig_2', initializer='zeros')

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(DiffGrad, self)._prepare_local(var_device, var_dtype, apply_state)
        # # # prepare local varialbes for iteration 
        local_step = math_ops.cast(self.iterations + 1, var_dtype)#increase iterator
        B_1_t = array_ops.identity(self._get_hyper('beta_1', var_dtype))#get beta_1
        B_2_t = array_ops.identity(self._get_hyper('beta_2', var_dtype))#get beta_2
        B_1_pt = math_ops.pow(B_1_t, local_step)#calculate beta_1^ local_step
        B_2_pt = math_ops.pow(B_2_t, local_step)#calculate beta_2^ local_step
        lr = (apply_state[(var_device, var_dtype)]['lr_t'] *(math_ops.sqrt(1 - B_2_pt) / (1 - B_1_pt)))#scale learning rate
        apply_state[(var_device, var_dtype)].update(dict(
            lr=lr,#learning rate
            epsilon=ops.convert_to_tensor(self.epsilon,var_dtype),#make small value epsi and make it tensor
            B_1_t=B_1_t,#beta_1 t update
            B_1_pt=B_1_pt,#beta_1^ t update
            om_B_1_t=1 - B_1_t,#1 - beta_1 t update
            B_2_t=B_2_t,#beta_2 t update
            B_2_pt=B_2_pt,#beta_2^ t update
            om_B_2_t=1 - B_2_t#1 - beta_1 t update
        ))#update changes 

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))

        # # # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, 'm')
        m_scaled_g_v = grad * coefficients['om_B_1_t']
        m_t = state_ops.assign(m, m * coefficients['B_1_t'] + m_scaled_g_v,
                               use_locking=self._use_locking)

        # # # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, 'v')
        v_scaled_g_v = (grad * grad) * coefficients['om_B_2_t']
        v_t = state_ops.assign(v, v * coefficients['B_2_t'] + v_scaled_g_v,
                               use_locking=self._use_locking)

        # # # diffgrad
        prev_g = self.get_slot(var, 'prev_g')
        
        # # # calculate DFC for the version
        dfc = 1.0
        g_mu=0.0
        g_sig=1.0
        if(self.d_version>=3):
          t=math_ops.cast(self.iterations + 1, var_dtype)
          sig_1=self.get_slot(var,'sig_1')
          sig_2=self.get_slot(var,'sig_2')
          g_mu=(sig_1/t)
          g_sig=(sig_2/t)-g_mu

        if(self.d_version==0):
          dfc = 1.0 / (1.0 + math_ops.exp(-math_ops.abs(prev_g - grad)))
        if(self.d_version==1):
          dfc = 1.0 / (1.0 + math_ops.exp(-(prev_g - grad)))
        if(self.d_version==2):
          dfc = (9.0 / (1.0 + math_ops.exp((-math_ops.abs(prev_g - grad))*0.5))) +0.4
        if(self.d_version==3):
          dfc = 1.0 / (1.0 + math_ops.exp(-(g_sig*math_ops.abs(prev_g - grad)-g_mu)))
        if(self.d_version==4):
          dfc = 1.0 / (1.0 + math_ops.exp(-(g_sig*g_sig*math_ops.abs(prev_g - grad)-g_mu)))
        if(self.d_version==5):
          dfc = 1.0 / (1.0 + math_ops.exp(-(math_ops.sqrt(abs(g_sig))*math_ops.abs(prev_g - grad)-g_mu)))
        



        # # #calculate root v_t
        v_sqrt = math_ops.sqrt(v_t)
        # # #calculate the update 
        var_update = state_ops.assign_sub(
            var, coefficients['lr'] * m_t * dfc / (v_sqrt + coefficients['epsilon']),#learning_rate*mt*DFC/(root_v+epsi)
            use_locking=self._use_locking)

        new_prev_g = state_ops.assign(prev_g, grad, use_locking=self._use_locking)
        
        # # #graph flow management to finish all computations and then proceed
        if(self.d_version>=3):
          new_sig_1=state_ops.assign(sig_1, sig_1+grad, use_locking=self._use_locking)
          new_sig_2=state_ops.assign(sig_2, sig_2+grad*grad, use_locking=self._use_locking)
          return control_flow_ops.group(*[var_update, m_t, v_t, new_prev_g,new_sig_1,new_sig_2])
        
        else:
          return control_flow_ops.group(*[var_update, m_t, v_t, new_prev_g])
    def get_config(self):
        # # #get hyper parameters
        config = super(DiffGrad, self).get_config()
        config.update({
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'decay': self._serialize_hyperparameter('decay'),
            'beta_1': self._serialize_hyperparameter('beta_1'),
            'beta_2': self._serialize_hyperparameter('beta_2'),
            'epsilon': self.epsilon,
            'diff_version': self.d_version
        })
        return config
#----------------------diffgrad----------------------#
'''
general imports for preprocessing data, models and training
'''

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.datasets import cifar10, cifar100

from tensorflow.keras.models import load_model
import np_utils
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np





#Dset=10#10,100
(x_train, y_train), (x_test, y_test)=((0,0),(0,0))




#----------------------dataset----------------------#
import cifar_preprocess as cifdset
from pathlib import Path
s_dir=cifdset.s_dir#get path for dataset
def dset_init(Dset=10,Dset_norm=True,Dset_aug=True):
    global x_train
    global y_train
    global x_test
    global y_test
    # # #get full path for dataset
    s_dir=cifdset.s_dir
    s_dir+=("_"+str(Dset))
    if(Dset_norm):
        s_dir+="_Nrm"
    if(Dset_aug):
        s_dir+="_Aug"
    
    # # #make dataset if not there or if remakearg
    
    if(remake_dset):
        print("\t-- cifar preprocess --")
        cifdset.cifar_preprocess(dset=Dset,showres=False,dset_norm=Dset_norm,dset_aug=Dset_aug)
        print("\t-- cifar preprocess --")



    if not(Path(s_dir).is_dir() and Path(s_dir+"\\x_train.npy").exists() and Path(s_dir+"\\y_train.npy").exists() and Path(s_dir+"\\x_test.npy").exists() and Path(s_dir+"\\y_test.npy").exists()):
        print("\t-- cifar preprocess --")
        cifdset.cifar_preprocess(dset=Dset,showres=False,dset_norm=Dset_norm,dset_aug=Dset_aug)
        print("\t-- cifar preprocess --")
        

    # # #load dataset 
    with open(s_dir+"\\"+"x_train.npy","rb") as f:
        x_train=np.load(f)
    with open(s_dir+"\\"+"y_train.npy","rb") as f:
        y_train=np.load(f)
    with open(s_dir+"\\"+"x_test.npy","rb") as f:
        x_test=np.load(f)
    with open(s_dir+"\\"+"y_test.npy","rb") as f:
        y_test=np.load(f)

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)


#----------------------dataset----------------------#

# # #learning rate scheduler
def scheduler(epoch, lr):
  if epoch < 80:
    return 9.21e-4
  else:
    return 1e-4
callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
# # #learning rate scheduler
class MyThresholdCallback(tf.keras.callbacks.Callback):
    def __init__(self, threshold):
        super(MyThresholdCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None): 
        val_acc = logs["val_acc"]
        if val_acc >= self.threshold:
            self.model.stop_training = True


'''
# # # nvidia gpu profiler callbackuse only if correct library with correct dll is present 
# # # memory usage statistics
'''
'''
class timecallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.times = []
        self.mems = []
        self.e_st = 0.0
        # use this value as reference to calculate cummulative time taken
        self.timetaken = 0.0
        nvidia_smi.nvmlInit()
        self.handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        self.info = nvidia_smi.nvmlDeviceGetMemoryInfo(self.handle)
        self.time_st = time.time()
    def on_train_begin(self,logs = {}):
        self.mems.append(self.info.used)
        print("\rmem:",self.info.used)
        self.timetaken = time.time()
    def on_epoch_begin(self,epoch,logs = {}):
        self.mems.append(self.info.used)
        self.e_st=time.time()
    def on_epoch_end(self,epoch,logs = {}):
        self.times.append((epoch,time.time() - self.e_st))
        self.mems.append(self.info.used)
        print("\r time:",self.times[-1])
    def on_train_end(self,logs = {}):
        self.mems.append(self.info.used)
        nvidia_smi.nvmlShutdown()
''' 
# # #resnet50 with imnet weights frozen and appended inference dense layers
def ResNet50imnet(in_sh=(32,32,3),num_classes=10,act="relu",num_freeze=143,upscale_im=False):


    input_im = Input(shape=in_sh)

    res_model=ResNet50(weights='imagenet', input_tensor=input_im, classes=16,include_top=False)
    for layer in res_model.layers[:num_freeze]:
        layer.trainable = False

    to_res = (224, 224)
    model = tf.keras.models.Sequential()
    if(upscale_im):
        model.add(Lambda(lambda image: tf.image.resize(image, to_res))) 
    model.add(res_model)
    model.add(Flatten())
    model.add(BatchNormalization())
    # if(act=="relu"):
        # model.add(Dense(256, activation='relu'))
    # if act=="lrelu" :
        # model.add(Dense(64, activation=LeakyReLU()))
    # if act=="selu" :
        # model.add(Dense(64, activation='selu'))
    # if act=="elu" :
        # model.add(Dense(64, activation='elu'))
    # model.add(Dropout(0.5))
    # model.add(BatchNormalization())
    
    # if(act=="relu"):
        # model.add(Dense(128, activation='relu'))
    # if act=="lrelu" :
        # model.add(Dense(64, activation=LeakyReLU()))
    # if act=="selu" :
        # model.add(Dense(64, activation='selu'))
    # if act=="elu" :
        # model.add(Dense(64, activation='elu'))
    # model.add(Dropout(0.1))
    # model.add(BatchNormalization())
    
    if(act=="relu"):
        model.add(Dense(64, activation='relu'))
    if act=="lrelu" :
        model.add(Dense(64, activation=LeakyReLU()))
    if act=="selu" :
        model.add(Dense(64, activation='selu'))
    if act=="elu" :
        model.add(Dense(64, activation='elu'))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    
    model.add(Dense(num_classes, activation='softmax'))

    return model


checkpoint_path="cp_cr50"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

   
opts=[]#optimizer list
opts.append(tf.keras.optimizers.SGD(learning_rate=1e-3, momentum=0.0, nesterov=False, name='SGD'))
opts.append(tf.keras.optimizers.Adagrad(learning_rate=1e-3, initial_accumulator_value=0.1, epsilon=1e-07,name='Adagrad'))
opts.append(tf.keras.optimizers.Adadelta(learning_rate=1e-3, rho=0.4, epsilon=1e-05, name='Adadelta'))
opts.append(tf.keras.optimizers.RMSprop(learning_rate=1e-3, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False,name='RMSprop'))
opts.append(tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,name='Adam'))
opts.append(tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True,name='AMSGrad'))


m_history={}#hostory of trainang 

resnet50=0

tf.keras.backend.clear_session()
# # #'imagenet' can be given for weight in resnet50
resnet50=ResNet50(weights=None, input_shape=(32, 32, 3), classes=Dset)#resnet50
#resnet50=ResNet50imnet(in_sh=(32,32,3),num_classes=10,act=act,num_freeze=143,upscale_im=False)#resnet50 with imnet weigths
#resnet50=c_ResNet50(in_sh=(32,32,3),num_classes=10,act=act)#unoptimized resnet50
#resnet50.summary()

def run_train(i_opt=4,batch_s=128,epoch_s=100,test_diffgrads=False,val_t=0.945): #32,64,128 #100
    print("\t\t ---------------train--------------- ")
    print("\t","[i_opt=",i_opt,"][batch_s=",batch_s,"][epoch_s=",epoch_s,"][test_diffgrads=",test_diffgrads,"]")
    #print("\t\t ---------------train--------------- ")
    
    #checkpoint callback to save trainig state
    # cp_callback = tf.keras.callbacks.ModelCheckpoint(
        # filepath=checkpoint_path, 
        # verbose=1, 
        # save_weights_only=True,
        # save_freq=20*batch_s)
    
    Threshold_Callback=MyThresholdCallback(threshold=val_t)
    #print("_"*100)
    if not test_diffgrads:
        print("\t\t\t ----",opts[i_opt]._name,batch_s,"----")
        print("\t\t ---------------train--------------- ")

        opt=opts[i_opt]
        tf.keras.backend.clear_session()
        
        resnet50.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])
        #callback_time=timecallback()
        print("_"*100)
        m_history = resnet50.fit(x_train, y_train, shuffle=True, epochs=epoch_s, batch_size=batch_s, validation_data=(x_test, y_test), callbacks=[callback,tensorboard_callback,Threshold_Callback])#callback_time

        #gc.collect()#garbage collection
        #del resnet50

        #print((m_history,callback_time.times,callback_time.mems))
        print("_"*100)
        print(m_history.history)
        # # #save history
        with open("hist_"+str(opts[i_opt]._name)+str(batch_s)+".csv","w") as f_hist:
            f_hist.write("loss,acc,val_loss,val_acc,lr\n")
            for i in range(len(m_history.history["loss"])):
                f_hist.write(str(m_history.history["loss"][i])+","+str(m_history.history["acc"][i])+","+str(m_history.history["val_loss"][i])+","+str(m_history.history["val_acc"][i])+","+str(m_history.history["lr"][i])+"\n")

    else:
        ver=i_opt
        print("\t\t\t ---- DiffGrad",ver,batch_s,"----")
        print("\t\t ---------------train--------------- ")

        opt=DiffGrad(diff_version=ver)
        tf.keras.backend.clear_session()
        
        resnet50.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])
        #callback_time=timecallback()
        print("_"*100)
        m_history = resnet50.fit(x_train, y_train, shuffle=True, epochs=epoch_s, batch_size=batch_s, validation_data=(x_test, y_test), callbacks=[callback,tensorboard_callback,Threshold_Callback])
        
        #gc.collect()#garbage collection
        #del resnet50
        
        #print((m_history,callback_time.times,callback_time.mems))
        print("_"*100)
        print(m_history.history)
        # # #save history
        with open("hist_"+str(opt._name)+str(ver)+str(batch_s)+".csv","w") as f_hist:
            f_hist.write("loss,acc,val_loss,val_acc,lr\n")
            for i in range(len(m_history.history["loss"])):
                f_hist.write(str(m_history.history["loss"][i])+","+str(m_history.history["acc"][i])+","+str(m_history.history["val_loss"][i])+","+str(m_history.history["val_loss"][i])+","+str(m_history.history["lr"][i])+"\n")


print("_"*100)
dset_init(Dset=Dset,Dset_norm=Dset_norm,Dset_aug=Dset_aug)
print("_"*100)
run_train(i_opt=i_opt,batch_s=batch_s,epoch_s=epoch_s,test_diffgrads=test_diffgrads,val_t=val_t)



















