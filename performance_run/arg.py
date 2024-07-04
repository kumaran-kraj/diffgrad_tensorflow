import getopt, sys
import argparse
from ast import literal_eval as make_tuple
# Initiate the parser
parser = argparse.ArgumentParser(prefix_chars='-+')
#i_opt=4,batch_s=128,epoch_s=100,test_diffgrads=False
#Dset_norm=True,Dset_aug=True
# Add long and short argument
parser.add_argument("--optimizer", "-p", type=int, help="set optimizer [0-5]")
parser.add_argument("--batch_size", "-b", type=int, help="set batch size (32,64,128)",default=128)
parser.add_argument("--actiavtion", "-a", help="set activation one of(relu)",default="relu")
parser.add_argument("--epochs", "-e", type=int, help="set number of epochs",default=50)
parser.add_argument("--dataset", "-d", type=int, choices=[10,100], help="set dataset  [10,100] cifar")
parser.add_argument("--validation_target", "-v", type=float, help="set validation target",default=0.94)
parser.add_argument("--test_diffgrads","-i", action="store_false", help="use other optimizers",default=False)
parser.add_argument("++test_diffgrads","+i", action="store_true", help="use diffgrad optimizers",default=False)
parser.add_argument("--normalize_dataset","-N", action="store_false", help="normalize dataset",default=True)
parser.add_argument("++normalize_dataset","+N", action="store_true", help="normalize dataset",default=True)
parser.add_argument("--augument_dataset","-G", action="store_false", help="augument dataset",default=True)
parser.add_argument("++augument_dataset","+G", action="store_true", help="augument dataset",default=True)
parser.add_argument("--remake_dataset","-r", action="store_false", help="remake dataset",default=False)
parser.add_argument("++remake_dataset","+r", action="store_true", help="remake dataset",default=False)

#choices=
# Read arguments from the command line
args = parser.parse_args()

# Check for --width


i_opt=4
batch_s=128
epoch_s=100
test_diffgrads=args.test_diffgrads
Dset_norm=args.normalize_dataset
Dset_aug=args.augument_dataset
act="relu"
remake_dset=args.remake_dataset
Dset=10
val_t=0.94
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

print("i_opt,batch_s,epoch_s,test_diffgrads,act,Dset,test_diffgrads,Dset_norm,Dset_aug,remake_dset,val_t")
print(i_opt,batch_s,epoch_s,test_diffgrads,act,Dset,test_diffgrads,Dset_norm,Dset_aug,remake_dset,val_t)

# # Get full command-line arguments
# full_cmd_arguments = sys.argv

# # Keep all but the first
# argument_list = full_cmd_arguments[1:]

# print (argument_list)

# short_options = "ho:v"
# long_options = ["help", "output=", "verbose"]

# try:
    # arguments, values = getopt.getopt(argument_list, short_options, long_options)
# except getopt.error as err:
    # # Output error, and return with an error code
    # print (str(err))
    # sys.exit(2)

# for current_argument, current_value in arguments:
    # print(current_argument, current_value)
  
# for current_argument, current_value in arguments:
    # if current_argument in ("-h", "--help"):
        # print ("Displaying help")
        # sys.exit()
    # elif current_argument in ("-v", "--verbose"):
        # print ("Enabling verbose mode")
    # elif current_argument in ("-o", "--output"):
        # print (("Enabling special output mode (%s)") % (current_value),int(current_value))