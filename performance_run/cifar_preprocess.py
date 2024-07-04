import tensorflow as tf
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.models import load_model
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import numpy as np

from pathlib import Path

from sklearn.utils import shuffle

kap=40000
kapc=1000
kapd=kap//4
imsla=[6,78,39,4976,789,536,3739,89,924,96,34,59,805,653,275,516,165,450,1580,676]
(x_train, y_train), (x_test, y_test)=((0,0),(0,0))
s_dir="cifar"
def cifar_preprocess (dset=10,showres=False,dset_norm=False,dset_aug=True): #10,100 #True,False
    global s_dir#="cifar"
    s_dir+=("_"+str(dset))
    if(dset_norm):
        s_dir+="_Nrm"
    if(dset_aug):
        s_dir+="_Aug"
    Path(s_dir).mkdir(parents=True, exist_ok=True)
    Path(s_dir+"\\x_train.npy").touch(exist_ok=True)
    Path(s_dir+"\\y_train.npy").touch(exist_ok=True)
    Path(s_dir+"\\x_test.npy").touch(exist_ok=True)
    Path(s_dir+"\\y_test.npy").touch(exist_ok=True)
    if(dset==10):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    if(dset==100):
        (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode="fine")
    from tensorflow.keras.utils import to_categorical

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    y_train = to_categorical(y_train, dset)
    y_test = to_categorical(y_test, dset)

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
    print("merge")
    x=np.append(x_train,x_test,axis=0)
    y=np.append(y_train,y_test,axis=0)
    
    print(x.shape)
    print(y.shape)
    print("merge")
    print("shuffle")
    x,y = shuffle(x,y, random_state=3)
    print("shuffle")
    
    print(x.shape)
    print(y.shape)
    print("split")
    
    x_train = x[:kap]
    x_test = x[kap-kapd+kapc:(kap+kapc)]
    y_train = y[:kap]
    y_test = y[kap-kapd+kapc:(kap+kapc)]
    print("split")
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)



    xtb=[]
    x_train1=[]
    y_train1=[]
    for i in imsla:
        xtb.append(np.copy(x_train[i]))
    print(len(xtb))
    if(dset_norm):
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
    
    #for j in tqdm(range(len(x_train))):
        #x_train[j]=tf.image.random_flip_left_right(tf.image.random_crop(tf.image.pad_to_bounding_box(x_train[j],4,4,40,40),size=(32,32,3)))
        #x_train[j]=tf.image.random_flip_left_right(x_train[j])

    #for j in tqdm(range(len(x_test))):
        #x_test[j]=tf.image.random_flip_left_right(tf.image.random_crop(tf.image.pad_to_bounding_box(x_test[j],4,4,40,40),size=(32,32,3)))
        #x_test[j]=tf.image.random_flip_left_right(x_test[j])
    trn_in=len(x_train)
    trn_in=25000
    if(dset_aug):
        for j in tqdm(range(trn_in)):
            x_train1.append(np.copy(tf.image.random_flip_left_right(tf.image.random_crop(tf.image.pad_to_bounding_box(x_train[j],4,4,40,40),size=(32,32,3)))))
        for j in tqdm(range(trn_in)):
            y_train1.append(np.copy(y_train[j]))
        
        x_train=np.append(x_train,x_train1,axis=0)
        y_train=np.append(y_train,y_train1,axis=0)
    
    print("shuffle")
    x_train,y_train = shuffle(x_train,y_train, random_state=3)
    print("shuffle")
    
    print("image increase")
    print(x_train.shape)
    print(y_train.shape)
    print("image increase")

    with open(s_dir+"\\"+"x_train.npy","wb+") as f:
        np.save(f,x_train)
    with open(s_dir+"\\"+"y_train.npy","wb+") as f:
        np.save(f,y_train)
    with open(s_dir+"\\"+"x_test.npy","wb+") as f:
        np.save(f,x_test)
    with open(s_dir+"\\"+"y_test.npy","wb+") as f:
        np.save(f,y_test)
    xr=[]
    yr=[]
    xt=[]
    yt=[]
    with open(s_dir+"\\"+"x_train.npy","rb") as f:
        xr=np.load(f)
    with open(s_dir+"\\"+"y_train.npy","rb") as f:
        yr=np.load(f)
    with open(s_dir+"\\"+"x_test.npy","rb") as f:
        xt=np.load(f)
    with open(s_dir+"\\"+"y_test.npy","rb") as f:
        yt=np.load(f)

    if((x_train==xr).all()  and (y_train==yr).all()  and (x_test==xt).all()  and (y_test==yt).all()):
        print("saved verified")
        print("x_train y_train x_test y_test")
        print( (x_train==xr).all()  , (y_train==yr).all()  , (x_test==xt).all()  , (y_test==yt).all() )
        print("saved verified")
    else :
        print("x_train y_train x_test y_test")
        print( (x_train==xr).all()  , (y_train==yr).all()  , (x_test==xt).all()  , (y_test==yt).all() )


    #plt.imshow(np.interp(xr[imsl],(x_train[imsl].min(),x_train[imsl].max()),(0,1)))
    #plt.show()
    if(showres):
        fig = plt.figure()
        columns = 4
        rows = 5
        ax = []
        for i in range(0,int(columns*rows),2):
            print(i//2,i+1,i+2)
            imsl=imsla[i//2]
            #img = np.random.randint(10, size=(h,w))
            # create subplot and append to ax
            ax.append( fig.add_subplot(rows, columns, i+1) )
            #ax[-1].set_title("ax:"+str(i))  # set title
            plt.imshow(np.interp(xtb[i//2],(xtb[i//2].min(),xtb[i//2].max()),(0,1)))
            plt.axis('off')
            ax.append( fig.add_subplot(rows, columns, i+2) )
            #ax[-1].set_title("ax:"+str(i))  # set title
            plt.imshow(np.interp(xr[imsl],(xr[imsl].min(),xr[imsl].max()),(0,1)))
            plt.axis('off')


        plt.show()
    