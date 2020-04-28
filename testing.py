from models import create_modules, Darknet, load_darknet_weights, torch
from utils.parse_config import *

cfg_path = 'cfg/one-stream.cfg'
cfg_path2 = 'cfg/yolov3-spp.cfg'
# cfg_defs = parse_model_cfg(cfg_path)
# modules, rest = create_modules(cfg_defs, (416, 418))
# for i, module in enumerate(modules):
#     print(module.__class__.__name__)
model = Darknet(cfg_path)
model2 = Darknet(cfg_path2)
# for parameter in model.parameters():
#     print(parameter.shape)
weights_path = 'weights/yolov3-spp.weights'


# with open(weights_path, 'rb') as f:
#     weights = np.fromfile(f, dtype=np.float32)  # the rest are weights

# load_darknet_weights(model, weights)

def print_layers_parametrs(self):
    for i, (mdef, module) in enumerate(zip(self.module_defs, self.module_list)):
        if mdef['type'] == 'convolutional':
            conv = module[0]
            if mdef['batch_normalize']:
                bn = module[1]
                a = bn.bias.numel() * 4
            else:
                # Load conv. bias
                nb = conv.bias.numel()
                a = nb
            # Load conv. weights
            nw = conv.weight.numel()  # number of weights
            total = a + nw
            print(total)


# print_layers_parametrs(model)


def insert_layer_weights_to_weights(model, weights, layer_weights, layer_index):
    new_weights_start = 0
    for i, (mdef, module) in enumerate(zip(model.module_defs[:layer_index], model.module_list[:layer_index])):
        if mdef['type'] == 'convolutional':
            conv = module[0]
            if mdef['batch_normalize']:
                bn = module[1]
                new_weights_start += bn.bias.numel() * 4 # 4 - bias, weights, mean, var
            else:
                # Load conv. bias
                nb = conv.bias.numel()
                new_weights_start += nb
            # Load conv. weights
            nw = conv.weight.numel()  # number of weights
            new_weights_start += nw

    before = weights[:new_weights_start]
    print(before.shape)
    after = weights[new_weights_start:]
    print(after.shape)
    print()
    new_weights = torch.cat((before, layer_weights, after), 0)
    return new_weights

# insert_layer_weights_to_weights(model, weights, torch.ones((10,)), 2)


chkpt = torch.load('weights/yolov3-spp.pt')

sdict = model.state_dict()
print(sdict.keys())

sdict2 = model2.state_dict()
print(sdict2.keys())
print('a')
# for k,v in chkpt['model'].items():
#     print(k)
#     print(model.state_dict()[k].numel())
#     print(v.numel())

# load model
try:
    chkpt['model'] = {k: v for k, v in chkpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
    model.load_state_dict(chkpt['model'], strict=False)
except KeyError as e:
    raise KeyError('Error') from e

# load optimizer
if chkpt['optimizer'] is not None:
    optimizer.load_state_dict(chkpt['optimizer'])
    best_fitness = chkpt['best_fitness']

# load results
if chkpt.get('training_results') is not None:
    with open(results_file, 'w') as file:
        file.write(chkpt['training_results'])  # write results.txt

start_epoch = chkpt['epoch'] + 1


print('done')
