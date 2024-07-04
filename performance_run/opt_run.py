import tensorflow as tf
import matplotlib.pyplot as plt
#import nvidia_smi
import time
from tqdm.auto import tqdm

from tensorflow.python.framework.ops import disable_eager_execution
import gc


physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

#disable_eager_execution()

from tensorflow.python.framework import ops
from tensorflow.python.keras import backend_config

from tensorflow.python.keras.optimizer_v2 import optimizer_v2

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops

from tensorflow.python.util.tf_export import keras_export
import_from_colab="from keras.optimizer_v2 import optimizer_v2"

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
        #self.version = version
        self.d_version = diff_version
        if self.d_version<0 or self.d_version>5:
          raise RuntimeError("No diffGrad version ",self.d_version)
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._set_hyper('decay', self._initial_decay)
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self.epsilon = epsilon or backend_config.epsilon()

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        # Create slots for previous g (gradient)
        # Create slots for sum of g and sum of g square
        # Separate for-loops to respect the ordering of slot variables from v1.
        # See if slot can be controlled by version for sig_1 and sig_2
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

        local_step = math_ops.cast(self.iterations + 1, var_dtype)
        beta_1_t = array_ops.identity(self._get_hyper('beta_1', var_dtype))
        beta_2_t = array_ops.identity(self._get_hyper('beta_2', var_dtype))
        beta_1_power = math_ops.pow(beta_1_t, local_step)
        beta_2_power = math_ops.pow(beta_2_t, local_step)
        lr = (apply_state[(var_device, var_dtype)]['lr_t'] *
              (math_ops.sqrt(1 - beta_2_power) / (1 - beta_1_power)))
        apply_state[(var_device, var_dtype)].update(dict(
            lr=lr,
            epsilon=ops.convert_to_tensor(self.epsilon, var_dtype),
            beta_1_t=beta_1_t,
            beta_1_power=beta_1_power,
            one_minus_beta_1_t=1 - beta_1_t,
            beta_2_t=beta_2_t,
            beta_2_power=beta_2_power,
            one_minus_beta_2_t=1 - beta_2_t
        ))

    def set_weights(self, weights):
        return #modify values
        params = self.weights
        num_vars = int((len(params) - 1) / 2)
        if len(weights) == 3 * num_vars + 1:
            weights = weights[:len(params)]
        super(DiffGrad, self).set_weights(weights)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))

        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, 'm')
        m_scaled_g_values = grad * coefficients['one_minus_beta_1_t']
        m_t = state_ops.assign(m, m * coefficients['beta_1_t'] + m_scaled_g_values,
                               use_locking=self._use_locking)

        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, 'v')
        v_scaled_g_values = (grad * grad) * coefficients['one_minus_beta_2_t']
        v_t = state_ops.assign(v, v * coefficients['beta_2_t'] + v_scaled_g_values,
                               use_locking=self._use_locking)

        # diffgrad
        prev_g = self.get_slot(var, 'prev_g')

        #dfc = 1.0 / (1.0 + math_ops.exp(-math_ops.abs(prev_g - grad)))
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
          dfc = 1.0 / (1.0 + math_ops.exp(-(math_ops.sqrt(g_sig)*math_ops.abs(prev_g - grad)-g_mu)))
        




        v_sqrt = math_ops.sqrt(v_t)
        var_update = state_ops.assign_sub(
            var, coefficients['lr'] * m_t * dfc / (v_sqrt + coefficients['epsilon']),
            use_locking=self._use_locking)

        new_prev_g = state_ops.assign(prev_g, grad, use_locking=self._use_locking)

        if(self.d_version>=3):
          new_sig_1=state_ops.assign(sig_1, sig_1+grad, use_locking=self._use_locking)
          new_sig_2=state_ops.assign(sig_2, sig_2+grad*grad, use_locking=self._use_locking)
          return control_flow_ops.group(*[var_update, m_t, v_t, new_prev_g,new_sig_1,new_sig_2])
        
        else:
          return control_flow_ops.group(*[var_update, m_t, v_t, new_prev_g])
    def get_config(self):
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

#----------------------dataset----------------------#
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
from tensorflow.keras.utils import to_categorical

x_train = x_train / 255.0
x_test = x_test / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

#x_train=x_train[:1000]
#y_train=y_train[:1000]

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
for j in tqdm(range(len(x_train))):
    for i in range(3):
        x_train[j][:,:,i:i+1]=(x_train[j][:,:,i:i+1]-np.mean(x_train[j][:,:,i:i+1]))/np.std(x_train[j][:,:,i:i+1])
        #print(round(x_train[j][:,:,i:i+1].mean(),5),round(x_train[j][:,:,i:i+1].std(),5))
        #print(x_train[j][:,:,i:i+1].mean(),x_train[j][:,:,i:i+1].std())
        #print(".",end="")

for j in tqdm(range(len(x_test))):
    for i in range(3):
        x_test[j][:,:,i:i+1]=(x_test[j][:,:,i:i+1]-np.mean(x_test[j][:,:,i:i+1]))/np.std(x_test[j][:,:,i:i+1])
        #print(round(x_test[j][:,:,i:i+1].mean(),5),round(x_test[j][:,:,i:i+1].std(),5))
        #print(x_test[j][:,:,i:i+1].mean(),x_test[j][:,:,i:i+1].std())
        #print(".",end="")
#----------------------dataset----------------------#

#learning rate scheduler
def scheduler(epoch, lr):
  if epoch < 80:
    return 1e-3
  else:
    return 1e-4
callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
#learning rate scheduler

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
opts=[]
opts.append(tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False, name='SGD'))
opts.append(tf.keras.optimizers.Adagrad(learning_rate=0.001, initial_accumulator_value=0.1, epsilon=1e-07,name='Adagrad'))
opts.append(tf.keras.optimizers.Adadelta(learning_rate=0.001, rho=0.95, epsilon=1e-07, name='Adadelta'))
opts.append(tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False,name='RMSprop'))
opts.append(tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,name='Adam'))
opts.append(tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True,name='AMSGrad'))


m_history={}


def run_train(i_opt=4,batch_s=32,epoch_s=100,test_diffgrads=False): #32,64,128 #100
    print("i_opt=",i_opt,"batch_s=",batch_s,"epoch_s=",epoch_s,"test_diffgrads=",test_diffgrads)
    if not test_diffgrads:
        print(opts[i_opt],batch_s)

        opt=opts[i_opt]
        tf.keras.backend.clear_session()

        resnet50=ResNet50(weights=None, input_shape=(32, 32, 3), classes=10)

        resnet50.compile(optimizer=opt, loss='binary_crossentropy', metrics=['acc'])
        #callback_time=timecallback()
        m_history = resnet50.fit(x_train, y_train, epochs=epoch_s, batch_size=batch_s, validation_data=(x_test, y_test), callbacks=[callback])#callback_time

        gc.collect()
        del resnet50

        #print((m_history,callback_time.times,callback_time.mems))
        
        print(m_history.history)
        with open("hist_"+str(opts[i_opt]._name)+str(batch_s)+".csv","w") as f_hist:
            f_hist.write("loss,acc,val_loss,val_acc,lr\n")
            for i in range(len(m_history.history["loss"])):
                f_hist.write(str(m_history.history["loss"][i])+","+str(m_history.history["acc"][i])+","+str(m_history.history["val_loss"][i])+","+str(m_history.history["val_acc"][i])+","+str(m_history.history["lr"][i])+"\n")

    else:
        ver=i_opt
        print(ver,batch_s)

        opt=DiffGrad(diff_version=ver)
        tf.keras.backend.clear_session()

        resnet50=ResNet50(weights=None, input_shape=(32, 32, 3), classes=10)

        resnet50.compile(optimizer=opt, loss='binary_crossentropy', metrics=['acc'])
        #callback_time=timecallback()
        m_history = resnet50.fit(x_train, y_train, epochs=epoch_s, batch_size=batch_s, validation_data=(x_test, y_test), callbacks=[callback])
        
        gc.collect()
        del resnet50
        
        #print((m_history,callback_time.times,callback_time.mems))
        
        print(m_history.history)
        with open("hist_"+str(opt._name)+str(ver)+str(batch_s)+".csv","w") as f_hist:
            f_hist.write("loss,acc,val_loss,val_acc,lr\n")
            for i in range(len(m_history.history["loss"])):
                f_hist.write(str(m_history.history["loss"][i])+","+str(m_history.history["acc"][i])+","+str(m_history.history["val_loss"][i])+","+str(m_history.history["val_loss"][i])+","+str(m_history.history["lr"][i])+"\n")

