"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).
"""

import numpy as np
import matplotlib.pyplot as plt

from torchvision import utils as vu

import torch
from torch.autograd import Variable
from collections import OrderedDict

def show_batch(batch):
    normed = batch * 0.5 + 0.5
    is_video_batch = len(normed.size()) > 4

    if is_video_batch:
        rows = [vu.make_grid(b.permute(1, 0, 2, 3), nrow=b.size(int(1))).numpy() for b in normed]
        im = np.concatenate(rows, axis=1)
    else:
        im = vu.make_grid(normed).numpy()

    im = im.transpose((1, 2, 0))

    plt.imshow(im)
    plt.show(block=True)


def count_parameters(model):
    return (sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad))



def summary(input_size, model):
    '''
        https://gist.github.com/HTLife/b6640af9d6e7d765411f8aa9aa94b837
    '''
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split('.')[-1].split("'")[0]
            module_idx = len(summary)

            m_key = '%s-%i' % (class_name, module_idx+1)
            summary[m_key] = OrderedDict()
            summary[m_key]['input_shape'] = list(input[0].size())
            summary[m_key]['input_shape'][0] = -1
            summary[m_key]['output_shape'] = list(output.size())
            summary[m_key]['output_shape'][0] = -1

            params = 0
            if hasattr(module, 'weight'):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                if module.weight.requires_grad:
                    summary[m_key]['trainable'] = True
                else:
                    summary[m_key]['trainable'] = False
            if hasattr(module, 'bias') and module.bias is not None:
                params +=  torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]['nb_params'] = params

        if not isinstance(module, torch.nn.Sequential) and \
           not isinstance(module, torch.nn.ModuleList) and \
           not (module == model):
            hooks.append(module.register_forward_hook(hook))

    dtype = torch.cuda.FloatTensor

    # check if there are multiple inputs to the network
    if isinstance(input_size[0], (list, tuple)):
        x = [Variable(torch.rand(1,*in_size)).type(dtype) for in_size in input_size]
    else:
        x = Variable(torch.rand(1,*input_size)).type(dtype)

    x = x.cuda()
    print(x.shape)
    print(type(x[0]))
    # create properties
    summary = OrderedDict()
    hooks = []
    # register hook
    model.apply(register_hook)
    # make a forward pass
    model(x)
    # remove these hooks
    for h in hooks:
        h.remove()

    print('----------------------------------------------------------------')
    line_new = '{:>20}  {:>25} {:>15}'.format('Layer (type)', 'Output Shpae', 'Param #')
    print(line_new)
    print('================================================================')
    total_params = 0
    trainable_params = 0
    for layer in summary:
        ## input_shape, output_shape, trainable, nb_params
        line_new = '{:>20} {:>25} {:>15}'.format(layer, str(summary[layer]['output_shape']), str(summary[layer]['nb_params']))
        total_params += summary[layer]['nb_params']
        if 'trainable' in summary[layer]:
            if summary[layer]['trainable'] == True:
                trainable_params += summary[layer]['nb_params']
        print(line_new)
    print('================================================================')
    print('Total params: ' + str(total_params))
    print('Trainable params: ' + str(trainable_params))
    print('Non-trainable params: ' + str(total_params - trainable_params))
    print('----------------------------------------------------------------')
    return summary
