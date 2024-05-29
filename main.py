import os
import random
import math
from itertools import product, chain, combinations
from tqdm import tqdm
from collections import defaultdict
import torch
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from data import Dataset
from prediction import Prediction
from configure import set_configure
from utils import write_txt, load_json
from model import SRL
import numpy as np
import multiprocessing as mp

class Trainer:
    def __init__(self, args, logger):
        for key, val in vars(args).items(): setattr(self, key, val);
        self.dataset = Dataset(f'data/{args.data}/', logger, sample_worker=args.sample_worker, 
                               sparsity=args.sparsity, inv=True)
        self.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        self.logger = logger
        self.rdict = self.dataset.rdict
        self.head_rdict = self.dataset.head_rdict
        self.relnum = self.rdict.idx
        self.model = SRL(self.relnum, args, self.device).to(self.device)

    def Train(self, train_rules):
        self.model.train()
        self.train_acc = defaultdict(list)
        losses = []
        for rule_len in range(2, self.learn_path_len+1):
            rules = train_rules[rule_len]
            self.logger.info(f"Training Rule length {rule_len}")
            for epoch in range(self.epochs):
                self.model.optimizer.zero_grad()
                rules_ = random.sample(rules, self.batch_size) if len(rules)>self.batch_size else rules
                bodys = torch.stack([r_[0:-2] for r_ in rules_], 0).to(self.device)
                heads = torch.stack([r_[-1] for r_ in rules_], 0).to(self.device)
                pred_head, loss = self.model(bodys, heads, epoch)
                if epoch % (self.epochs//100) == 0:
                    self.logger.info("Epoch {} - Loss:{:.3}".format(epoch, loss.mean()))
                self.train_acc[rule_len].append(((pred_head.argmax(dim=1) == heads.reshape(-1)).sum() / pred_head.shape[0]).cpu().numpy())
                clip_grad_norm_(self.model.base_model.parameters(), 0.5)
                loss.backward()
                self.model.optimizer.step()
                losses.append(loss.mean().detach().cpu().numpy())
        self.model.save_model()
        np.save(self.output_dir+'loss.npy', np.array(losses))

    def Test(self):
        os.makedirs(self.rule_dir)
        self.model.load_model(); self.model.eval()
        for rule_len in range(2, self.test_path_len+1):
            probs = []
            if os.path.exists(f'data/{args.data}/body_test.json'):
                body = [list(map(int, i.split('_'))) for i in load_json(f'data/{args.data}/body_test.json').keys() if len(i.split('_'))==rule_len]
                body = [[self.rdict.idx2rel[j] for j in i] for i in body]
            else:
                body = list(product(self.rdict.rel2idx.keys(), repeat=rule_len))
            epochs = math.ceil(float(len(body))/self.test_size)
            for i in tqdm(range(epochs)):
                bodies = body[i*self.test_size:] if i==epochs-1 else body[i*self.test_size: (i+1)*self.test_size]
                body_idx = [list(map(lambda x:self.head_rdict.rel2idx[x], i)) for i in bodies] if type(bodies[0][0])==str else bodies
                inputs = torch.LongTensor(body_idx).to(self.device)
                with torch.no_grad():
                    pred_head, _ = self.model.base_model(inputs, train=False)
                    probs.append(torch.softmax(pred_head, dim=-1).detach().cpu())
            topk = min(self.topk, len(body))
            if args.sort_relwise:
                info = []
                for r in tqdm(range(self.relnum)):
                    selected_col = torch.cat([i[:,0] for i in probs], dim=0)
                    probs = [j[:,1:] for j in probs]
                    sorted_val, sorted_idx = torch.sort(selected_col, descending=True)
                    info.append(["{}<--{}: {:.3f}".format(self.head_rdict.idx2rel[r], ','.join(body[sorted_idx[i]]), sorted_val[i]) for i in range(topk)])
            else:
                rule_conf = torch.cat(probs, dim=0); del probs
                sorted_val, sorted_idx = torch.sort(rule_conf, 0, descending=True)
                info = [["{}<--{}: {:.3f}".format(self.head_rdict.idx2rel[r], ','.join(body[sorted_idx[i, r]]), sorted_val[i, r]) for i in range(topk)] for r in range(self.relnum)]
            write_txt([[i] for i in chain.from_iterable(info)], self.rule_dir + f"rule_{rule_len}_{topk}.txt")

if __name__ == '__main__':
    args, logger = set_configure()
    if args.train or args.test:
        trainer = Trainer(args, logger)
    if args.train:
        with mp.Pool(args.sample_worker) as pool:
            train_rules = trainer.dataset.sample(pool, args.learn_path_len, args.sample_pairs)
            pool.close(); pool.join()
        trainer.Train(train_rules)
    if args.test:
        trainer.Test()
    if args.pred:
        predictor = Prediction(args, logger)
        lengths = list(chain.from_iterable([combinations(range(2, args.test_path_len+1), i) for i in range(1, args.test_path_len)]))
        for ls in lengths:
            predictor.KGC(lengths=ls)