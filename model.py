import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from torch.distributions import Categorical
import math
import numpy as np
from stable import DWR, StableNet

class SRL(nn.Module):
    def __init__(self, relation_num, args, device):
        super().__init__()
        # Base Model Configs
        self.device = device
        self.base_model = NCRL(relation_num, args.emb_size, device)
        # Stable Model Configs
        self.stable = args.stable
        if args.stable:
            self.stable_device = torch.device(f"cuda:{args.stable_gpu}" if torch.cuda.is_available() else "cpu")
            if args.stable_model == 'lin':
                self.stable_model = DWR(args, self.stable_device)
            elif args.stable_model == 'nonlin':
                self.stable_model = StableNet(args, self.stable_device)
        # Optim Configs
        self.optimizer = torch.optim.Adam(self.base_model.parameters(), lr=args.lr)
        self.model_dir = args.output_dir+'model.pkt'
        
    def forward(self, bodys, heads, epoch):
        pred_head, h = self.base_model(bodys)
        loss = self.base_model.compute_loss(pred_head, heads, bodys)
        if self.stable:
            weight = self.stable_model(h.to(self.stable_device), epoch)
            loss = loss.unsqueeze(0).mm(weight.to(self.device)).squeeze()
        else:
            loss = loss.mean()
        return pred_head, loss
    
    def save_model(self):
        torch.save(self.base_model.state_dict(), self.model_dir)
        
    def load_model(self):
        self.base_model.load_state_dict(torch.load(self.model_dir, map_location=self.device))

class NCRL(nn.Module):
    def __init__(self, relation_num, emb_size, device, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(relation_num+1, emb_size, padding_idx=relation_num)
        self.lstm = nn.LSTM(input_size=emb_size, hidden_size=emb_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(emb_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.fc_k = nn.Linear(emb_size, emb_size)
        self.fc_q = nn.Linear(emb_size, emb_size)
        self.fc_v = nn.Linear(emb_size, emb_size)
        self.loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.relation_num = relation_num
        self.emb_size = emb_size
        self.device = device
    
    def forward(self, bodys, train=True):
        out = self.embedding(bodys) # [batch_size, seq_len, emb_size]
        batch_size, _, emb_size = out.size()
        self.indices = torch.arange(self.relation_num, device=self.device).repeat(batch_size, 1)
        while out.size(1)>2:
            rel_pairs = []
            for idx in range(out.size(1)-1):
                _, (h, _) = self.lstm(out[:,idx:idx+2,:])  # h: [1, batch_size, hid_size]
                rel_pairs.append(h.squeeze(0))
            rel_pairs = torch.stack(rel_pairs,dim=1) # [batch_size, seq_len-1, hid_size]
            rel_pairs_score = self.sigmoid(self.fc(rel_pairs).squeeze(-1)) # [batch_size, seq_len-1]
            selected_rel_pair_idx = torch.argmax(rel_pairs_score, dim=-1) # [batch_size]
            full_batch = torch.arange(batch_size).to(self.device)
            selected_rel_pair = rel_pairs[full_batch, selected_rel_pair_idx, :].unsqueeze(1) # [batch_size, 1, hid_size]
            scores = self.attention(selected_rel_pair) # unnormalized # (batch_size, 1, |R|+1)
            selected_rel_pair = self.weighted_average(scores, selected_rel_pair).squeeze(1) # (batch_size, emb_size)
            out = out.detach().clone()
            out[full_batch, selected_rel_pair_idx, :] = selected_rel_pair
            # out[full_batch, selected_rel_pair_idx+1, :] = torch.zeros(emb_size, device=self.device)
            # out = out[out.sum(dim=-1)!=0].reshape(batch_size, -1, emb_size)
            mask = torch.ones([out.size(0), out.size(1)], device=self.device).scatter_(1, (selected_rel_pair_idx+1).unsqueeze(1), 0)
            out = out[mask.bool()].reshape(out.size(0), -1, self.emb_size)
        _, (h, _) = self.lstm(out)  # h: [1, batch_size, hid_size]
        out = h.transpose(0, 1) # [batch_size, 1, hid_size]
        scores = self.attention(out).squeeze(1)
        return scores, out.squeeze(1)

    def attention(self, inputs):
        key = self.fc_k(torch.cat((self.embedding(self.indices), inputs), dim=1))  
        scores = torch.matmul(self.fc_q(inputs), key.transpose(-2, -1)) / math.sqrt(self.emb_size) # (batch_size, seq_len, |R|+seq_len)
        return scores
    
    def weighted_average(self, scores, inputs):
        batch_size, seq_len, _ = scores.shape
        mask1 = torch.zeros((batch_size, seq_len, self.relation_num), dtype=torch.bool, device=self.device)
        I = torch.eye(seq_len, dtype=torch.bool, device=self.device)
        mask2 = ~I.unsqueeze(0).repeat(batch_size, 1, 1) # (batch_size, seq_len, seq_len)
        mask = torch.cat((mask1, mask2), dim=-1) # (batch_size, seq_len,|R|+seq_len)
        scores[mask] = float('-inf')
        relation_emb = self.embedding(torch.arange(self.relation_num, device=self.device).repeat(batch_size, 1)) # (batch_size, |R|, emb_size)
        all_emb = torch.cat((relation_emb, inputs), dim=1) # (batch_size, |R|+seq_len, emb_size)
        return torch.softmax(scores, dim=-1) @ all_emb

    def compute_loss(self, pred_head, heads, bodys=None):
        return self.loss(pred_head, heads.reshape(-1))

class RLogic(nn.Module):
    def __init__(self, relation_num, emb_size, device, num_layers=1):
        super().__init__()
        hidden_size = emb_size
        self.embedding = nn.Embedding(relation_num, emb_size)
        self.lstm = nn.LSTM(input_size=emb_size, hidden_size=hidden_size, 
                            num_layers=num_layers, batch_first=True)  
        self.hid2rel = nn.Linear(hidden_size, relation_num)
        self.body2head = nn.Linear(hidden_size, relation_num+1)
        self.fc_1 = nn.Linear(emb_size*2, hidden_size)
        self.fc_2 = nn.Linear(hidden_size, relation_num+1)
        self.loss_head = torch.nn.CrossEntropyLoss()
        self.loss_body = torch.nn.CrossEntropyLoss()
        self.relation_num = relation_num
        self.emb_size = emb_size
        self.num_layers = num_layers
        self.device = device
    
    def predict_body(self, inputs):
        if not hasattr(self, 'state'):
            self.state = self.init_state(inputs.size(0))
        self.state = [state.detach() for state in self.state]
        out, self.state = self.lstm(self.embedding(inputs), self.state)
        pred = self.hid2rel(out.reshape(-1, out.size(2)))
        return pred
    
    def predict_head(self, bodys):
        state = self.init_state(bodys.size(0))
        body_hid, (h,c) = self.lstm(self.embedding(bodys), state)
        indices = torch.arange(self.relation_num, device=self.device).repeat(bodys.size(0), 1)
        for i in range(body_hid.size(1)-1):
            if i == 0:
                emb_1 = relation_emb[:, i, :]
            else:
                relation_emb = torch.cat((self.embedding(indices), body_hid[:, i, :].unsqueeze(1)), dim=1)
                prob_ = prob_sf.unsqueeze(1)
                emb_1 = (prob_ @ relation_emb).squeeze(1)
            emb_concat = torch.cat((emb_1, relation_emb[:, i+1, :]), dim=1)
            prob = self.fc_2(F.relu(self.fc_1(emb_concat)))
            prob_sf = prob.softmax(dim=1)
        return prob, emb_concat

    def forward(self, bodys, train=True):
        if train:
            self.pred_body_tmp = self.predict_body(bodys[:, 0:-1])
        pred_head, emb = self.predict_head(bodys)
        return pred_head, emb

    def compute_loss(self, pred_head, heads, bodys):
        loss_body = self.loss_body(self.pred_body_tmp, bodys[:, 1:].reshape(-1))
        loss_head = self.loss_head(pred_head, heads.reshape(-1))
        return loss_body+loss_head

    def init_state(self, batch_size):
        state = (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device),
                 torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device))
        return state