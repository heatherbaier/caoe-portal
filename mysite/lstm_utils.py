from collections import OrderedDict
from torchvision import models
from threading import Lock
import torch
import heapq
import os
import gc

from lstm import *

from lstm_config import get_config
config, _ = get_config()


def load_ddp_state(state_dict):

    r18 = LSTM(input_size = 512,
                 hidden_size = 128,
                 output_size = config.time_steps)

    key_transformation = {k:v for k,v in zip(state_dict.keys(), r18.state_dict().keys())}

    new_state_dict = OrderedDict()

    for key, value in state_dict.items():
        new_key = key_transformation[key]
        new_state_dict[new_key] = value

    del r18, key_transformation, state_dict
    gc.collect()

    return new_state_dict
    

def sort_by_size(munis, imagery_dir):
    """
    Sort images by size, highest to lowest, but keep the name of the images
    """
    munis = [imagery_dir + i for i in munis]
    munis = sorted(munis, key =  lambda x: os.stat(x).st_size)
    munis.reverse()
    return munis


def sublist_creator(lst, n):
    """
    Split the sorted file list into n lists of equal sums
    """
    lists = [[] for _ in range(n)]
    totals = [(0, i) for i in range(n)]
    heapq.heapify(totals)
    for value in lst:
        total, index = heapq.heappop(totals)
        lists[index].append(value)
        heapq.heappush(totals, (total + value, index))
    return lists


def make_worker_list(files_lists, ppn):
    """
    Get the list of workers based on the number of files to each node
    """
    workers = []
    for c, (i) in enumerate(files_lists):
        for j in range(0, len(i)):
            workers.append(j + (ppn * c))
    return workers


def reverse_size(files_lists, size_dict):
    """
    Make a new imagery list 
    """
    image_list = []
    for j in files_lists:
        for i in j:
            image_list.append(size_dict[i])
    return image_list


def organize_data(base_dir, ppn, nodes):
    
    # Get a list of the municipalities
    munis = os.listdir(base_dir)
    munis = [i for i in munis if i.startswith("484")]
        
    # Sort the municipalities from biggest to smallest size
    munis = sorted(munis, key =  lambda x: os.stat(base_dir + x).st_size)
    
    # Make a dictionary with the image sizes as keys and image names as values
    size_dict = {}
    for x in munis:
        size_dict[os.stat(base_dir + x).st_size] = base_dir + x
                    
    # Change the munis list to be image sizes then reverse it
    munis = [os.stat(base_dir + x).st_size for x in munis]
    munis.reverse()
        
    files_lists = sublist_creator(munis, nodes)
    workers = make_worker_list(files_lists, ppn)
    image_list = reverse_size(files_lists, size_dict)
        
    return image_list, workers


def load_extracter_state(state_dict):

    r18 = LSTM(input_size = 512,
                 hidden_size = 128,
                 output_size = 12)

    key_transformation = {k:v for k,v in zip(state_dict.keys(), r18.state_dict().keys())}

    new_state_dict = OrderedDict()

    for key, value in state_dict.items():
        if "fc." not in key:
            new_key = key_transformation[key]
            new_state_dict[new_key] = value

    del r18, key_transformation, state_dict
    gc.collect()

    return new_state_dict
