from copy import deepcopy
from tqdm import tqdm
from collections import defaultdict
from random import sample, random, shuffle
import torch
from utils import load_txt, parse_triple, divide_chunks

class RelationDict:
    def __init__(self, data_dir, inv=True):
        self.rel2idx, self.idx2rel = {}, {}
        self.idx = 0; self.noninv_idx = 0
        for rel in load_txt(data_dir, merge_list=True):
            self.add_relation(rel)
            if inv: self.add_relation('inv_'+rel)
            self.noninv_idx += 1

    def add_relation(self, rel):
        if rel not in self.rel2idx:
            self.rel2idx[rel] = self.idx
            self.idx2rel[self.idx] = rel
            self.idx += 1

class Dataset(object):
    def __init__(self, data_dir, logger, sample_worker=10, sparsity=1, inv=True):
        self.logger = logger
        self.sample_worker = sample_worker
        # Entity Dict
        entities = load_txt(data_dir+'entities.txt', merge_list=True)
        self.ent2idx = {ent:idx for idx, ent in enumerate(entities)}
        self.idx2ent = {i:j for j, i in self.ent2idx.items()}
        # Relation Dict
        self.rdict = RelationDict(data_dir+'relations.txt', inv=inv)
        self.head_rdict = deepcopy(self.rdict)
        self.head_rdict.add_relation("None")
        # Fact Triples
        fact_file = 'facts.txt.inv' if inv else 'facts.txt'
        fact = load_txt(data_dir+fact_file)
        self.fact_rdf = sample(fact, int(len(fact)*sparsity))
        # Graph Triples
        self.train_rdf = load_txt(data_dir+'train.txt')
        self.valid_rdf = load_txt(data_dir+'valid.txt')
        self.test_rdf = load_txt(data_dir+'test.txt')
        self.all_rdf = self.fact_rdf + self.train_rdf + self.valid_rdf

    def sample(self, pool, max_path_len, pair_num):
        rel2rdf = defaultdict(list); self.graph_h2rt = defaultdict(list); self.graph_ht2r = defaultdict(list)
        for rdf in self.all_rdf:
            h, r, t = parse_triple(rdf)
            rel2rdf[r].append((h, r, t)); self.graph_h2rt[h].append((r, t)); self.graph_ht2r[(h, t)].append(r)
        self.graph_ht2r = dict(self.graph_ht2r)
        # Sample Pairs
        sampled_rdf = []
        relwise_pairs = pair_num // self.rdict.noninv_idx
        for head in self.head_rdict.rel2idx:
            if head != "None" and "inv_" not in head:
                sampled_rdf_ = sample(rel2rdf[head], relwise_pairs) if relwise_pairs<len(rel2rdf[head]) else rel2rdf[head]
                sampled_rdf.extend(sampled_rdf_)
        train_rule_dict, train_rules, num_sample = defaultdict(list), defaultdict(list), 0
        shuffle(sampled_rdf)
        sampled_rdf = list(divide_chunks(sampled_rdf, len(sampled_rdf)//(self.sample_worker-1)))
        graph_h2rts = [self.graph_h2rt]+[deepcopy(self.graph_h2rt) for _ in tqdm(range(len(sampled_rdf)-1))]
        graph_ht2rs = [self.graph_ht2r]+[deepcopy(self.graph_ht2r) for _ in tqdm(range(len(sampled_rdf)-1))]
        jobs=[pool.apply_async(construct_rule, args=(sampled_rdf[i], graph_h2rts[i], graph_ht2rs[i], max_path_len)) for i in range(len(sampled_rdf))]
        data = [job.get() for job in jobs]
        for d in tqdm(data):
            for rule_seq, record in d: 
                num_sample += len(set(record))
                for rule in rule_seq:
                    body, head = rule.split('-'); body_path = body.split('|')
                    idx = [self.head_rdict.rel2idx[rel] if rel!=-1 else -1 for rel in body_path+[-1, head]]
                    train_rule_dict[head].append(idx)
                    train_rules[len(body_path)].append(torch.LongTensor(idx))
        # Metrics of Sampled Rules
        del sampled_rdf, graph_h2rts, graph_ht2rs
        self.logger.info("Rule Count:{}".format(num_sample))
        self.logger.info("Head Count:{}".format(len(train_rule_dict)))
        for h in train_rule_dict:
            self.logger.info("Head {}:{}".format(h, len(train_rule_dict[h])))
        self.logger.info("Facts number:{} Sample number:{}".format(len(self.fact_rdf), num_sample))
        for rule_len, rules_ in train_rules.items():
            self.logger.info(f"Sampled {len(rules_)} examples for length {rule_len}")
        return train_rules

def construct_rule(sampled_rdf, graph_h2rt, graph_ht2r, max_path_len=2):
    # Sample Rules
    out = []
    for rdf in tqdm(sampled_rdf):
        anchor_h, anchor_r, anchor_t = parse_triple(rdf)
        # Sample Rules for each RDF
        attempt = 0
        stack = [(anchor_h, anchor_r, anchor_t)]
        rule_seq, expended_node, record = [], [], []
        while len(stack) > 0 and attempt<100000:
            cur_h, cur_r, cur_t = stack.pop(-1)
            rt_list = graph_h2rt[cur_t] if cur_t in graph_h2rt else []
            if len(cur_r.split('|')) < max_path_len and len(rt_list) > 0 and cur_t not in expended_node:
                for r_, t_ in rt_list:
                    if t_ not in [cur_h, anchor_h]:
                        stack.append((cur_t, cur_r+'|'+r_, t_)); attempt+=1
            expended_node.append(cur_t)
            head_rel = sample(graph_ht2r[(anchor_h, cur_t)], 1) if (anchor_h, cur_t) in graph_ht2r else []
            if head_rel and cur_t != anchor_t:
                rule = cur_r + '-' + head_rel[0]
                rule_seq.append(rule); record.append((cur_h, r_, t_))
            elif len(head_rel)==0 and random() > 0.9:
                rule = cur_r + '-' + "None"
                rule_seq.append(rule); record.append((cur_h, r_, t_))
        out.append([rule_seq, record])
    return out