import numpy as np
import re
import torch

class ModelUtils(object):

    def __init__(self, model, device, layer_depth=-1, conv_only=0, kernel_size=[3, 100]):

        if layer_depth == -1:
          layer_depth = 1000000
        self.model = model
        self.state_dict_keys = list(model.state_dict().keys())
        self.max_layer_depth = self.find_max_layer_depth()
        self.layer_depth = min(max(1, layer_depth), self.max_layer_depth)
        self.conv_only = conv_only
        self.kernel_size = kernel_size
        self.layers_names = self.list_of_layers_names(self.layer_depth)
        self.num_layers = len(self.layers_names)
        self.device = device
        
    def find_max_layer_depth(self):

        ans = 1
        for name in self.state_dict_keys:
          name = name.split('.')
          ans = max(ans, len(name))

        return ans-1

    def list_of_layers_names(self, layer_depth):

        names = np.array(list(map(lambda x: '.'.join( x.split('.')[:-1] if layer_depth >= len(x.split('.')) else x.split('.')[:layer_depth]), self.state_dict_keys)))
        unique_arr, indices = np.unique(names, return_index=True)
        indices = np.sort(indices)
        names = names[indices]
        
        if self.conv_only == 1:
          results = []
          pattern = r"kernel_size=\((\d+), (\d+)\)"
          for module_str in names:
            module = self.model
            for layer_name in module_str.split("."):
              module = getattr(module, layer_name)

            string = module
            match = re.search(pattern, str(string))
            if match:
              x1 = int(match.group(1))
              x2 = int(match.group(2))
              if self.kernel_size[0]<=x1<=self.kernel_size[1] and self.kernel_size[0]<=x2<=self.kernel_size[1]:
                results.append(module_str)
          names = results

        return names

    def get_layer_name(self, index):

        if index < 0 or index >= self.num_layers:
            raise ValueError(f"Invalid index: {index}. Expected an integer between 0 and {self.num_layers - 1}.")

        return self.layers_names[index]

    def find_matching_strings(self, string1, strs):

        matching_strings = [s for s in strs if s.startswith(string1)]
        
        return matching_strings

    def get_layer_weights(self, index):

        layer_name = self.get_layer_name(index)
        layers_names = self.find_matching_strings(layer_name, self.state_dict_keys)
        
        ans = []
        for name in layers_names:
            if self.conv_only == 0 or len(list(self.model.state_dict()[name].shape)) == 4:
                ans.append(self.model.state_dict()[name])

        return ans
    
    def get_feature_map_batch(self, index, X):

        # a dict to store the activations
        activation = {}
        def getActivation(name):
          # the hook signature
          def hook(model, input, output):
            activation[name] = output.detach()
          return hook
        
        module_str = self.get_layer_name(index)
        module = self.model
        for layer_name in module_str.split("."):
          module = getattr(module, layer_name)
        
        h = module.register_forward_hook(getActivation(module_str))

        # forward pass -- getting the outputs
        out = self.model(X)
        # detach the hooks
        h.remove()

        return activation[module_str]

    def get_feature_maps_dataloader(self, index, dataloader):

        # a dict to store the activations
        activation = {}
        def getActivation(name):
          # the hook signature
          def hook(model, input, output):
            activation[name] = output.detach()
          return hook

        module_str = self.get_layer_name(index)
        module = self.model
        for layer_name in module_str.split("."):
          module = getattr(module, layer_name)
        
        h = module.register_forward_hook(getActivation(module_str))

        # forward pass -- getting the outputs
        feat_maps_list = []
        # forward pass -- getting the outputs
        for X, y in dataloader: 
          out = self.model(X.to(self.device))
          feat_maps_list.append(activation[module_str])
          
        # detach the hooks
        h.remove()

        return feat_maps_list

    
    def get_feature_maps_dataloader_for_all_layers(self, dataloader):

        # a dict to store the activations
        activation = {}
        def getActivation(name):
          # the hook signature
          def hook(model, input, output):
            activation[name] = output.detach()
          return hook

        layers_hooks = []
        names = self.layers_names

        for module_str in names:
          module = self.model
          for layer_name in module_str.split("."):
            module = getattr(module, layer_name)
          
          layers_hooks.append(module.register_forward_hook(getActivation(module_str)))

        feat_maps_list = None
        # forward pass -- getting the outputs
        for X, y in dataloader: 
          _ = self.model(X.to(self.device))
          if feat_maps_list == None:
            feat_maps_list = list(activation.values())
          else:
            values = list(activation.values())
            for i in range(len(feat_maps_list)):
              feat_maps_list[i] += values[i]
        
        for i in range(len(feat_maps_list)):
          feat_maps_list[i] /= len(dataloader)
        # detach the hooks
        for h in layers_hooks:
          h.remove()

        return feat_maps_list