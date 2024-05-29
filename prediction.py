import os
from tqdm import tqdm
import random
from itertools import chain
from collections import defaultdict
import numpy as np
from scipy import sparse
from configure import set_configure
from utils import load_txt, parse_triple
from data import Dataset
from torch.utils.data import DataLoader


class RuleDataset:
    def __init__(self, fact_rdf, test_rdf, rules, ent2idx, idx2rel, data_dir):
        self.rules = rules
        self.ent2idx = ent2idx
        self.e_num = len(ent2idx)
        self.idx2rel = idx2rel
        self.data_dir = data_dir
        self.test_ents = set([i[0] for i in test_rdf])
        self.matrices = {r:sparse.dok_matrix((self.e_num, self.e_num)) for r in self.idx2rel.values()}
        self.start_matrices = {r:sparse.dok_matrix((self.e_num, self.e_num)) for r in self.idx2rel.values()}
        for rdf in fact_rdf:
            h, r, t = parse_triple(rdf)
            self.matrices[r][self.ent2idx[h], self.ent2idx[t]] = 1
            if h in self.test_ents:
                self.start_matrices[r][self.ent2idx[h], self.ent2idx[t]] = 1
    
    def __len__(self):
        return len(self.rules)
    @staticmethod
    def collate_fn(data):
        return [i[0] for i in data], [i[1] for i in data]
    def __getitem__(self, idx):
        path_count = sparse.dok_matrix((self.e_num,self.e_num))
        rel = self.idx2rel[idx] if 'dblp' not in self.data_dir else 'AuthorIs'
        for body, conf in self.rules[rel]:
            body_adj = sparse.eye(self.e_num)
            body_adj = body_adj * self.start_matrices[body[0]]
            for i in body[1:]:
                body_adj = body_adj * self.matrices[i]
            path_count += body_adj * conf
        return (rel, path_count)

class Prediction:
    def __init__(self, args, logger, sub_population=True):
        for key, val in vars(args).items(): setattr(self, key, val);
        self.data_dir = f'data/{args.data}/'
        self.dataset = Dataset(self.data_dir, logger, inv=True)
        for key in ['fact_rdf', 'train_rdf', 'valid_rdf', 'test_rdf', 'ent2idx']:
            setattr(self, key, vars(self.dataset)[key])
        self.truth = defaultdict(list)
        for rdf in self.fact_rdf+self.train_rdf+self.valid_rdf+self.test_rdf:
            h, r, t = parse_triple(rdf)
            self.truth[(h, r)].append(self.dataset.ent2idx[t])
        self.rdict = self.dataset.rdict
        self.logger = logger
        self.sub_population = sub_population
        self.to_test = {i:load_txt(os.path.join(self.data_dir, i)) for i in os.listdir(self.data_dir) if 'test_' in i} if sub_population else {'all':self.test_rdf}

    def load_rules(self, lengths):
        self.all_rules = defaultdict(list)
        for L in lengths:
            file = [i for i in os.listdir(self.rule_dir) if f'rule_{L}' in i][0]
            rules = load_txt(f"{self.rule_dir}/{file}", merge_list=True)
            for rule in rules:
                rule, conf = rule.strip('\n').split(': ')
                if float(conf) >= self.threshold:
                    head, body = rule.split("<--")
                    if 'dblp' in self.data_dir and 'AuthorIs' not in head:
                        continue
                    self.all_rules[head].append((tuple(body.split(",")), float(conf)))
        self.all_rules = {i:sorted(j, key=lambda x:x[1], reverse=True) for i,j in self.all_rules.items()}
    
    def KGC(self, lengths, pred_num=5000):
        self.load_rules(lengths)
        rule_dataset = RuleDataset(self.fact_rdf+self.train_rdf+self.valid_rdf, self.test_rdf,
                                   self.all_rules, self.ent2idx, self.rdict.idx2rel, self.data_dir)
        num_workers = 2 if 'yago' in self.data_dir else len(self.all_rules)//10
        pred_num = 2000 if 'yago' else pred_num
        rule_loader = DataLoader(rule_dataset, batch_size=2, num_workers=num_workers, collate_fn=RuleDataset.collate_fn)
        print('Getting Scores')
        scores = dict(chain.from_iterable([list(zip(rel, path_count)) for _, (rel, path_count) in tqdm(enumerate(rule_loader))]))
        hit_1s, hit_10s, mrrs = {}, {}, {}; hit_1s_p, hit_10s_p, mrrs_p = {}, {}, {}
        for file, to_test in self.to_test.items():
            hit_1, hit_10, mrr = defaultdict(list), defaultdict(list), defaultdict(list)
            hit_1_p, hit_10_p, mrr_p = defaultdict(list), defaultdict(list), defaultdict(list)
            test_rdf = [i for i in to_test if i[1] in scores]
            test_rdf = random.sample(test_rdf, pred_num) if len(test_rdf)>pred_num else test_rdf
            for i, rdf in tqdm(enumerate(test_rdf)):
                (q_h, q_r, q_t) = parse_triple(rdf)
                if q_r not in scores: continue
                score = np.array(scores[q_r][self.ent2idx[q_h]].todense()).squeeze()
                filter = list(set(self.truth[(q_h, q_r)])-set([self.ent2idx[q_t]]))
                score[filter] = -1
                rank_ = np.sum(score>score[self.ent2idx[q_t]]).item() + 1
                pred_ranks = np.argsort(score)[::-1]
                rank = (pred_ranks == self.ent2idx[q_t]).nonzero()[0].item() + 1
                # self.logger.info(f"Count {i}: {q_h}\t{q_r}\t{q_t}\t{rank}")
                mrr[q_r].append(1.0/rank)
                hit_1[q_r].append(1 if rank<=1 else 0)
                hit_10[q_r].append(1 if rank<=10 else 0)
                mrr_p[q_r].append(1.0/rank_)
                hit_1_p[q_r].append(1 if rank_<=1 else 0)
                hit_10_p[q_r].append(1 if rank_<=10 else 0)
            mrr_ = np.mean(list(chain.from_iterable(mrr.values()))); mrrs[file] = mrr_
            hit_1_ = np.mean(list(chain.from_iterable(hit_1.values()))); hit_1s[file] = hit_1_
            hit_10_ = np.mean(list(chain.from_iterable(hit_10.values()))); hit_10s[file] = hit_10_
            self.logger.info(f"Results for {file} L {lengths} - MRR {mrr_}\t Hits@1 {hit_1_}\t Hits@10 {hit_10_}")
            mrr_p_ = np.mean(list(chain.from_iterable(mrr_p.values()))); mrrs_p[file] = mrr_p_
            hit_1_p_ = np.mean(list(chain.from_iterable(hit_1_p.values()))); hit_1s_p[file] = hit_1_p_
            hit_10_p_ = np.mean(list(chain.from_iterable(hit_10_p.values()))); hit_10s_p[file] = hit_10_p_
            self.logger.info(f"Results Plus for {file} L {lengths} - MRR {mrr_p_}\t Hits@1 {hit_1_p_}\t Hits@10 {hit_10_p_}")
        if self.sub_population:
            mrrs_mean, mrrs_std = np.mean(list(mrrs.values())), np.std(list(mrrs.values()))
            hit_1s_mean, hit_1s_std = np.mean(list(hit_1s.values())), np.std(list(hit_1s.values()))
            hit_10s_mean, hit_10s_std = np.mean(list(hit_10s.values())), np.std(list(hit_10s.values()))
            self.logger.info(f"Overall for L {lengths} - MRR {mrrs_mean}±{mrrs_std} \t Hits@1 {hit_1s_mean}±{hit_1s_std}\t Hits@10 {hit_10s_mean}±{hit_10s_std}")
            mrrs_mean, mrrs_std = np.mean(list(mrrs_p.values())), np.std(list(mrrs_p.values()))
            hit_1s_mean, hit_1s_std = np.mean(list(hit_1s_p.values())), np.std(list(hit_1s_p.values()))
            hit_10s_mean, hit_10s_std = np.mean(list(hit_10s_p.values())), np.std(list(hit_10s_p.values()))
            self.logger.info(f"Overall Plus for L {lengths} - MRR {mrrs_mean}±{mrrs_std} \t Hits@1 {hit_1s_mean}±{hit_1s_std}\t Hits@10 {hit_10s_mean}±{hit_10s_std}")

if __name__ == '__main__':
    args, logger = set_configure()
    if args.pred:
        predictor = Prediction(args, logger)
        predictor.KGC([2])
        # kg_completion(all_rules, dataset,args)