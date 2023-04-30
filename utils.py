import logging
import random
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset, DataLoader
from model_utils import ModelUtils
import matplotlib.pyplot as plt
import seaborn as sns
import pretrainedmodels
import subprocess
import ssl
import urllib.request
import os
import tarfile
import zipfile
import os
import libtorrent as lt
import time
import datetime
import shutil

def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def import_dataloader(args, model_name, i):
    if args.torchvision == 0:
        X = pretrainedmodels.pretrained_settings[model_name]['imagenet']['input_size'][-2]
        Y = pretrainedmodels.pretrained_settings[model_name]['imagenet']['input_size'][-1]
    else:
        X = Y = args.input_shape[i]
    
    transform = transforms.Compose([
        transforms.Resize(size=(X, Y)),
        transforms.ToTensor(),
        transforms.Normalize( 
        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010) 
        )
    ])

    # Load the full dataset
    dataset = getattr(torchvision.datasets, args.dataset)
    if args.dataset == "ImageNet":
        install_ImageNet_libraries()
        download_validation_ImagenNet(args)
        train_dataset = torchvision.datasets.ImageFolder(root='./data', transform=transform)
    else:
        train_dataset = dataset(root='./data', train=True, download=True, transform=transform)
    logging.info(len(train_dataset))

    # Select a subset of the data
    subset_indices = list(range(0, args.batch_size*2*args.dataloader_size, 2))
    subset = Subset(train_dataset, subset_indices)

    # Create a dataloader for the subset
    dataloader = DataLoader(subset, batch_size=args.batch_size, shuffle=False)
    logging.info(len(dataloader))

    return dataloader

def avarage_output_feat_maps_dataloader(output):
    
    # Count the number of tensors in each list
    num_tensors = len(output[0])

    # Initialize a list to store the mean of tensors
    mean_output = []

    # Compute the mean of tensors with the same index
    for i in range(num_tensors):
        # Initialize a variable to store the sum of tensors with the same index
        tensor_sum = torch.zeros_like(output[0][i])
        # Iterate through each list of tensors
        for j in range(len(output)):
            # Add the tensor at the current index to tensor_sum
            tensor_sum += output[j][i]
        # Compute the mean of tensors with the same index
        tensor_mean = tensor_sum / len(output)
        # Add the tensor_mean to the mean_output
        mean_output.append(tensor_mean)

    logging.info("len(mean_output) : ",len(mean_output))
    logging.info("mean_output[0].shape : ",mean_output[0].shape)

    return mean_output


def run_CKA(args, cka, mean_output, remove_output_layer=1): 
    
    # if args.conv_only == 1:
    #     mean_output = list(filter(lambda x: len(list(x.shape)) == 4, mean_output))

    CKA_matrix = torch.zeros((len(mean_output)-remove_output_layer, len(mean_output)-remove_output_layer), dtype=float)
    logging.info(CKA_matrix.shape)

    for i in range(CKA_matrix.shape[0]):
        if len(list(mean_output[i].shape)) == 4:
            mean_output[i] = torch.mean(mean_output[i], axis=(1, 2))
        elif len(list(mean_output[i].shape)) == 3:
            mean_output[i] = torch.mean(mean_output[i], axis=(1))
    
    for i in range(CKA_matrix.shape[0]):
        for j in  range(CKA_matrix.shape[0]):
            if args.CKA_type == 'kernel_CKA':
                CKA_matrix[i][j] = CKA_matrix[j][i] = cka.kernel_CKA(mean_output[i], mean_output[j])
            else:
                CKA_matrix[i][j] = CKA_matrix[j][i] = cka.linear_CKA(mean_output[i], mean_output[j])

    return CKA_matrix

def run_CKA_diff(args, cka, mean_output, mean_output2, remove_output_layer=1): 

    # if args.conv_only == 1:
    #     mean_output = list(filter(lambda x: len(list(x.shape)) == 4, mean_output))
    #     mean_output2 = list(filter(lambda x: len(list(x.shape)) == 4, mean_output2))

    CKA_matrix = torch.zeros((len(mean_output)-remove_output_layer, len(mean_output2)-remove_output_layer), dtype=float)
    logging.info(CKA_matrix.shape)

    for i in range(CKA_matrix.shape[0]):
        if len(list(mean_output[i].shape)) == 4:
            mean_output[i] = torch.mean(mean_output[i], axis=(1, 2))
        elif len(list(mean_output[i].shape)) == 3:
            mean_output[i] = torch.mean(mean_output[i], axis=(1))

    for i in range(CKA_matrix.shape[1]):
        if len(list(mean_output2[i].shape)) == 4:
            mean_output2[i] = torch.mean(mean_output2[i], axis=(1, 2))
        elif len(list(mean_output2[i].shape)) == 3:
            mean_output2[i] = torch.mean(mean_output2[i], axis=(1))

    for i in range(CKA_matrix.shape[0]):
        for j in  range(CKA_matrix.shape[1]):
            if args.CKA_type == 'kernel_CKA':
                CKA_matrix[i][j] = cka.kernel_CKA(mean_output[i], mean_output2[j])
            else:
                CKA_matrix[i][j] = cka.linear_CKA(mean_output[i], mean_output2[j])
            
            # if j<CKA_matrix.shape[0] and i<CKA_matrix.shape[1]:
            #     CKA_matrix[j][i] = CKA_matrix[i][j]

    return CKA_matrix

def find_feature_maps_for_model(args, model_name, i, dataloader):
    
    if args.torchvision == 1:
        model_path = getattr(torchvision.models, model_name)
        model = model_path(pretrained=args.pretrained[i]).to(args.device)
    else:
        model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet').to(args.device)

    model_utils = ModelUtils(model, args.device, args.layers_depth[i], args.conv_only, args.kernel_size)
    if model_utils.max_layer_depth < args.layers_depth[i]:
        logging.warning(f'out of range layer depth for model with index {i} named : {model_name}')

    output = model_utils.get_feature_maps_dataloader_for_all_layers(dataloader)
    logging.info(len(output))
    output = avarage_output_feat_maps_dataloader(output)

    del model
    
    return output

def heatmap_plot(args, CKA_matrix, i):

    # Create a heatmap using seaborn
    ax = sns.heatmap(CKA_matrix) #cbar=False
    ax.invert_yaxis()
    string = f"depth {args.layers_depth[i]} "
    if args.conv_only == 1:
        string = "conv_only "
    title = string+f"{args.compare_models_names[i]} vs {args.models_names[i]}.png"    
    plt.title(title)
    plt.savefig(args.output_path+"/"+title, bbox_inches='tight')
    # Display the plot
    plt.show()
    
def install_pretrainedmodels():

    subprocess.check_call(['pip', 'install', 'pretrainedmodels'])
    ssl._create_default_https_context = ssl._create_unverified_context
    response = urllib.request.urlopen("https://www.example.com")


def install_ImageNet_libraries():

    subprocess.run(["pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    subprocess.run(["pip", "install", "lbry-libtorrent", "wget", "torf"])
    subprocess.run(["apt", "install", "python3-libtorrent"])


def download_validation_ImagenNet(args):

    if os.path.exists('./data/valid'):
        return

    params = {
        'save_path': './Torrent/',
        'storage_mode': lt.storage_mode_t(2),
    }

    ses = lt.session()
    ses.listen_on(6881, 6891)
    link = "magnet:?xt=urn:btih:5d6d0df7ed81efd49ca99ea4737e0ae5e3a5f2e5&tr=https%3A%2F%2Facademictorrents.com%2Fannounce.php&tr=udp%3A%2F%2Ftracker.coppersurfer.tk%3A6969&tr=udp%3A%2F%2Ftracker.opentrackr.org%3A1337%2Fannounce"

    if link.endswith('.torrent'):
        import wget
        from torf import Torrent

        if os.path.exists('torrent.torrent'):
            os.remove('torrent.torrent')

        wget.download(link, 'torrent.torrent')
        t = Torrent.read('torrent.torrent')
        link = str(t.magnet(name=True, size=False, trackers=False, tracker=False))


    logging.info(link)
    handle = lt.add_magnet_uri(ses, link, params)
    # change the 0 to a 1 to download sequentially
    handle.set_sequential_download(0)
    ses.start_dht()
    begin = time.time()

    logging.info(datetime.datetime.now())
    logging.info('Downloading Metadata...')

    while not handle.has_metadata():
        time.sleep(1)

    logging.info('Got Metadata, Starting Torrent Download...')
    logging.info("Starting", handle.name())

    while handle.status().state != lt.torrent_status.seeding:
        s = handle.status()
        state_str = ['queued', 'checking', 'downloading metadata',
                    'downloading', 'finished', 'seeding', 'allocating']
        logging.info('%.2f%% complete (down: %.1f kb/s up: %.1f kB/s peers: %d) %s ' %
            (s.progress * 100, s.download_rate / 1000, s.upload_rate / 1000,
            s.num_peers, state_str[s.state]))
        time.sleep(5)

    end = time.time()
    logging.info(handle.name(), "COMPLETE")
    logging.info("Elapsed Time: ", int((end - begin) // 60), "min :", int((end - begin) % 60), "sec")
    logging.info(datetime.datetime.now())


    # Create the data directory if it does not exist
    if not os.path.exists('data'):
        os.mkdir('data')

    if not os.path.exists('./data/valid'):
        os.mkdir('./data/valid')

    # Extract the contents of the tar file to the data directory
    with tarfile.open('./Torrent/ILSVRC2012_img_val.tar', 'r') as tar:
        tar.extractall('./data/')
    
    os.remove('./Torrent/ILSVRC2012_img_val.tar')

    for filename in sorted(os.listdir("/data/valid"))[:-(args.dataloader_size*args.batch_size+10)]:
        filename_relPath = os.path.join("/data/valid",filename)
        os.remove(filename_relPath)


   
