import csv
import json
from itertools import chain
import torch 

def load_txt(input_dir, merge_list=False):
    ans = [line for line in csv.reader(open(input_dir, 'r', encoding='UTF-8'), delimiter='\t') if len(line)]
    if merge_list:
        ans = list(chain.from_iterable(ans))
    return ans

def write_txt(info_list, out_dir):
    with open(out_dir, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(info_list)

def load_json(input_dir, serial_key=False):
    ret_dict = json.load(open(input_dir))
    if serial_key:
        ret_dict = {tuple([int(i) for i in k.split('_')]):[tuple(l) for l in v] for k,v in ret_dict.items()}
    return ret_dict

def write_json(info_dict, out_dir, serial_key=False):
    if serial_key:
        info_dict = {'_'.join([str(i) for i in k]):v for k,v in info_dict.items()}
    with open(out_dir, "w") as f:
        json.dump(info_dict, f)

def parse_triple(triples, head_mode=True):
    if head_mode:
        h, r, t = triples
    else:
        t, r, h = triples
    return h, r, t

def divide_chunks(l, n):
    for i in range(0, len(l), n): 
        yield l[i:i + n]