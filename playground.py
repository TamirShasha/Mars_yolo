from models import create_modules, Darknet, load_darknet_weights, torch
from utils.parse_config import *

cfg_path = 'cfg/one-stream.cfg'
model = Darknet(cfg_path)
chkpt = torch.load('weights/yolov3-spp.pt')
sdict = model.state_dict()

keys = chkpt['model'].keys()


# new_layers hold the indexes of new layers added to the network, for example, [1,2,3]
def shift_weights_dict_upon_new_layers(wdict, new_layers_ind):
    curr_layers_indexes = np.unique([int(key.split('.')[1]) for key in wdict.keys()])

    final_layers_indexes = curr_layers_indexes
    for layer_ind in new_layers_ind:
        final_layers_indexes = [x + 1 if x >= layer_ind else x for x in final_layers_indexes]

    key_dict = dict(zip(curr_layers_indexes, final_layers_indexes))

    def parse_key(key):
        splitted = key.split('.')
        splitted[1] = str(key_dict[int(splitted[1])])
        return '.'.join(splitted)

    new_wdict = {parse_key(k): v for k, v in wdict.items()}
    return new_wdict


new_wdict = shift_weights_dict_upon_new_layers(chkpt['model'], [1, 2, 3])



print('done')
