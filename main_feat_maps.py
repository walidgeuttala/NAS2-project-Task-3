import argparse
import logging
import pickle
import os
import math
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from cka import CKA, CudaCKA
from utils import *
from model_utils import ModelUtils
import itertools

# !python3 main_feat_maps.py --models_names resnet18 resnet34 --layers_depth -1 --compare_all 1 --conv_only 1 --kernel_size 3 --remove_output_layer 0 --batch_size 50 --dataloader_size 2
# pnasnet5large, nasnetalarge, senet154, polynet, inceptionresnetv2 


def parse_args():

    parser = argparse.ArgumentParser(description="helps for extracting CKA feature maps")
    parser.add_argument("--dataset", type=str, default="ImageNet", help="choose dataset")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--dataloader_size", type=int, default=1, help="dataloader size")
    parser.add_argument("--torchvision", type=int, default=0, help="use torchvision library or pretrainedmodels")
    parser.add_argument("--pretrained", type=int, nargs='+', default=[1], help="pretrained 1 or 0")
    parser.add_argument("--device", type=int, default=0, help="device id, -1 for cpu")
    parser.add_argument("--models_names",type=str, nargs='+', help="model architectures you can enter one name model or list of models")
    parser.add_argument("--compare_models_names",type=str, default=None, nargs='+', help="model architectures vs who to compare with you can enter one name model or list of models",)
    parser.add_argument("--output_path", type=str, default="./output")
    parser.add_argument("--CKA_type", type=str, nargs='+', default=["kernel_CKA"], choices = ["kernel_CKA", "linear_CKA"], help="kernel_CKA, linear_CKA")
    parser.add_argument("--layers_depth", type=int, nargs='+', help="you can enter -1 for max depth or specifie depth you want")
    parser.add_argument("--conv_only", type=int, default=0, help="use conv_only for the weights return")
    parser.add_argument("--input_shape", type=int, default=[224], nargs='+', help="enter input shape as list of models for torchvision models default is 224")
    parser.add_argument("--remove_output_layer", type=int, default=0, help="remove last layer you 0 or 1")
    parser.add_argument("--compare_all", type=int, default=0, help="compare_all generation of experiments")
    parser.add_argument("--kernel_size", type=int, default=[3, 100], nargs='+', help="range of conv_only kernel size you can enter one kernel size or range")
    parser.add_argument("--flatten", type=int, default=1, help="use flatten with no losing info about CKA or 0 means use kernel average")

    args = parser.parse_args()

    # device
    args.device = "cpu" if args.device == -1 else "cuda:0"
    if not torch.cuda.is_available():
        logging.warning("CUDA is not available, use CPU for training.")
        args.device = "cpu"

    if args.compare_all == 1:
        pairs = list(itertools.combinations(args.models_names, 2))
        for model_name in args.models_names:
            pairs.append((model_name, model_name))
        args.models_names = []
        args.compare_models_names = []
        for pair in pairs:
            args.models_names.append(pair[0])
            args.compare_models_names.append(pair[1])
        args.models_names = [item for _ in range(len(args.layers_depth)) for item in args.models_names] 
        args.compare_models_names = [item for _ in range(len(args.compare_models_names)) for item in args.compare_models_names] 
        args.length = len(args.models_names)
        args.layers_depth = [item for _ in range(args.length//len(args.layers_depth)) for item in args.layers_depth] 
        args.layers_depth.sort()
    else:
        args.length = len(args.models_names)
        if args.length != len(args.layers_depth):
            args.layers_depth = [args.layers_depth[0]] * args.length
    
    if args.conv_only == 1 and len(args.kernel_size) == 1:
        args.kernel_size.append(args.kernel_size[0])
    if args.length != len(args.pretrained):
        args.pretrained = [args.pretrained[0]] * args.length
    if args.length != len(args.input_shape):
        args.input_shape = [args.input_shape[0]] * args.length
    if args.length != len(args.CKA_type):
        args.CKA_type = [args.CKA_type[0]] * args.length
    if args.compare_models_names == None:
        args.compare_models_names = args.models_names


    # paths
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    name = "Data_{}_Batch_Size{}_Dataloader_Size_{}.txt".format(
        args.dataset,
        args.batch_size,
        args.dataloader_size,
    ) 
    args.output = os.path.join(args.output_path, name)
    
    return args

def main():
    
    set_random_seed()
    if args.device == 'cuda:0':
        cka = CudaCKA(args.device)
    else:
        cka = CKA()
    
    if args.dataset == "ImageNet" and not os.path.exists('./data/valid'):
        download_validation_ImagenNet(args)
    for i in range(args.length):
        
        if args.compare_models_names[i] == args.models_names[i]:
            dataloader = import_dataloader(args, args.models_names[i], i)
            output = find_feature_maps_for_model(args, args.models_names[i], i, dataloader)
            CKA_matrix = run_CKA(args, cka, output, args.remove_output_layer)
            del output
        
        else:
            dataloader = import_dataloader(args, args.models_names[i], i)
            output1 = find_feature_maps_for_model(args, args.models_names[i], i, dataloader)
            del dataloader        
            dataloader = import_dataloader(args, args.compare_models_names[i], i)
            output2 = find_feature_maps_for_model(args, args.compare_models_names[i], i, dataloader)
    
            CKA_matrix = run_CKA_diff(args, cka, output1, output2, args.remove_output_layer)  
            del output1, output2
        
        heatmap_plot(args, CKA_matrix, i)

        del dataloader, CKA_matrix
        torch.cuda.empty_cache()

if __name__ == "__main__":
    
    args = parse_args()
    
    ssl._create_default_https_context = ssl._create_unverified_context
    response = urllib.request.urlopen("https://www.example.com")

    with open(args.output, 'w') as f:
        f.write(str(args))
    main()
