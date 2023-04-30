# Feature map similarity for model layers using CKA
# =================================================

Key words : 
> NAS, hidden feature maps, CKA, heatmap

### General idea about the project

I have created dynamic code that allows you to control the depth of the block. I named it 'layer' as I prefer the name. You can set any depth you desire. I observed that there are some architectures with a depth of 5 blocks inside an extra block, which is more general, in my opinion.

I have written the code with only one line, and you can compare multiple models with the same or different models. You can also use either the kernel or linear CKA. Furthermore, there are other parameters in the code that control the input, such as data, batch size, data loader size, pre-trained models, conv_only, etc.

I display for each pair of models the heatmap and save the information about the command.

### requirement 

You must install pretrainedmodels library and fix ssl unverified context

```python
!pip install pretrainedmodels
import ssl
import urllib.request

# Disable SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

# Access the website
response = urllib.request.urlopen("https://www.example.com")
```

### command line paramters 

Note : the main file called : main_feat_maps.py


- `--dataset` type=str, default="CIFAR10", help="specfie the name of the dataset to use example: CIFAR10"

- `--batch_size` type=int, default=5, help="number of images in the batch"

- `--dataloader_size` type=int, default=2, help="number of batchs in the dataloader"

- `--torchvision` type=int, default=0, help="use torchvision library or pretrainedmodels library for importing models"

- `--pretrained` type=int, nargs='+', default=[1], help="pretrained weights is 1 or not pretrained 0", Note="works only when using `--torchvision 1`"

- `--device` type=int, default=0, help="device id -1 for cpu other then -1 it uses cuda:0, Note="it is slow as uses CKA class for cpu"

- `--models_names` type=str, nargs='+', help="model architectures names you can enter one name model or list of models"
```python
# You can use one model
!python3 main_feat_maps.py --models_names vgg16
# Or use more then 1 model
!python3 main_feat_maps.py --models_names vgg16 resnet18
```
- `--compare_models_names` type=str, default=None, nargs='+', help="model architectures to compare with `--models_names`. You can enter one name model or list of models", Note="if you did not spesfie the list the models of `--models_names` each model comapred to it self"

- `--output_path` type=str, default="./output"

- `--input_shape` type=int, default=[224], nargs='+', help="used only for torchvision when it it not compatble with input model shape, default is 224. for pretrainedmodels library it is seted automatcily"

- `--CKA_type` type=str, nargs='+', default=["kernel_CKA"], choices = ["kernel_CKA", "linear_CKA"], help="use kernel_CKA or linear_CKA", Note="if one item in the list it will be used for all the `--models_names`"

- `--layers_depth` type=int, nargs='+', help="you can enter -1 for max depth meaning layers if used 1 or more then we spsfie the the depth of block for dfs in the hierachical in model architucture", Note="you can enter 1 item in the list and will be used for all models or you give same size list as the models_names, only in one case whene `--compare_all 1` then you can use any number of items"

- `--conv_only` type=int, default=0, help="use conv_only 1 or use all layers 0", Note="used only when the `--layers_depth -1`"

- `--remove_output_layer` type=int, default=1, help="remove last layer from representation of matrix of CKA used only when `--conv_only 0`, meaning we have softmax or any type last layer that we don't want it in CKA presentation"

- `--compare_all` type=int, default=0, help="compare_all generation of nCr where r = 2, used to make comparsing of all choises of 2 models between the list"
> if we have list of (`--models_names` and `--layers_depth`) the we generaete all cobimantion of `--models_names` (`--models_names vgg16 resnet18`) will be [vgg16 vgg16 resnet18] and this generated list will be combined with all possible choices with `--output_path` list

- `--kernel_size` type=int, default=[3, 100], nargs='+', help="range of conv_only kernel size you can enter one kernel size or range", Note="used only when `--conv_only 1`"


### example
 
```python
!python3 main_feat_maps.py --models_names resnet18 resnet34 --layers_depth -1 --compare_all 1 --conv_only 1 --remove_output_layer 0 --batch_size 150 --dataloader_size 4
```

![Alt text](./images/"conv_only resnet18 vs resnet18.png" "conv_only resnet18 vs resnet18")




