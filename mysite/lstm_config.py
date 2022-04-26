from datetime import datetime, date
import argparse


arg_lists = []
parser = argparse.ArgumentParser(description = "a3cRAM")

def str2bool(v):
    return v.lower() in ("true", "1")


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# with open("/sciclone/home20/hmbaier/tm/lstm/date.txt", "r") as f:
#     now = f.read().strip("\n")


training_args = add_argument_group("Training Arguments")
training_args.add_argument("--tv_split", 
                           type = float, 
                           default = 0.75, 
                           help = "Train/Val split percentage - given as a float i.e. .75")
training_args.add_argument("--lr", 
                           type = float, 
                           default = .0001, 
                           help = "Learning Rate")
training_args.add_argument("--batch_size", 
                           type = float, 
                           default = 4,
                           help = "Batch Size")
training_args.add_argument("--epochs", 
                           type = float, 
                           default = 2,
                           help = "Batch Size")
training_args.add_argument("--time_steps", 
                           type = float, 
                           default = 6,
                           help = "Batch Size")


data_args = add_argument_group("Data Arguments")
data_args.add_argument("--features_dir",
                      type = str,
                      default = "/sciclone/scr-mlt/hmbaier/temporal_features/",
                      help = "Full path to directory containing extracted temporal imagery features")


# misc_args = add_argument_group("Miscellaneous Arguments")
# misc_args.add_argument("--log_name",
#                       type = str,
#                       default = "/sciclone/home20/hmbaier/tm/lstm/records/records (" + now + ")/log (" + now + ").txt",
#                       help = "Full path to directory containing imagery")
# misc_args.add_argument("--use_rpc",
#                       type = bool,
#                       default = False,
#                       help = "Whether to use RPC to communicate training statistics.")
# misc_args.add_argument("--find_unused_parameters",
#                       type = bool,
#                       default = False,
#                       help = "Whether to use find unused parameters in DDP model initialization.")
# misc_args.add_argument("--records_dir",
#                       type = str,
#                       default = "/sciclone/home20/hmbaier/tm/lstm/records/records (" + now + ")/",
#                       help = "Path to save epoch records too")
# misc_args.add_argument("--epochs_dir",
#                       type = str,
#                       default = "/sciclone/home20/hmbaier/tm/lstm/records/records (" + now + ")/epochs/",
#                       help = "Path to save epoch records too")
# misc_args.add_argument("--models_dir",
#                       type = str,
#                       default = "/sciclone/home20/hmbaier/tm/lstm/records/records (" + now + ")/models/",
#                       help = "Path to save trained_models too")


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed