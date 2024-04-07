import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
import numpy as np
import math


class TimeEncode(torch.nn.Module):
    def __init__(self, expand_dim, factor=5):
        super(TimeEncode, self).__init__()
        #init_len = np.array([1e8**(i/(time_dim-1)) for i in range(time_dim)])
        time_dim = expand_dim
        self.factor = factor
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim))).float())
        self.phase = torch.nn.Parameter(torch.zeros(time_dim).float())
        #self.dense = torch.nn.Linear(time_dim, expand_dim, bias=False)
        #torch.nn.init.xavier_normal_(self.dense.weight)
        
    def forward(self, ts):
        # ts: [N, L]
        batch_size = ts.size(0)
        seq_len = ts.size(1)
                
        ts = ts.view(batch_size, seq_len, 1)# [N, L, 1]
        map_ts = ts * self.basis_freq.view(1, 1, -1) # [N, L, time_dim]
        map_ts += self.phase.view(1, 1, -1)
        
        # harmonic = torch.cat((torch.cos(map_ts),torch.sin(map_ts)),-1)
        harmonic = torch.cos(map_ts)

        return harmonic #self.dense(harmonic)

class DBTKT(nn.Module):
    def __init__(self, n_at, n_it, n_exercise, n_question, d_a, d_e, d_k, dropout=0.2, use_at=False,use_it=True):
        super(DBTKT, self).__init__()
        self.use_at = use_at
        self.use_it = use_it
        self.d_k = d_k
        self.d_a = d_a
        self.d_e = d_e
        self.n_question = n_question

        self.at_embed = nn.Embedding(n_at + 10, d_k)
        torch.nn.init.xavier_uniform_(self.at_embed.weight)
        self.it_embed = nn.Embedding(n_it + 10, d_k)
        torch.nn.init.xavier_uniform_(self.it_embed.weight)
        self.e_embed = nn.Embedding(n_exercise + 10, d_k)
        torch.nn.init.xavier_uniform_(self.e_embed.weight)
        self.c_embed = nn.Embedding(n_question + 1, d_k)
        torch.nn.init.xavier_uniform_(self.c_embed.weight)
        self.c_embed_diff = nn.Embedding(n_question + 1, d_k)
        self.difficult_param = nn.Embedding(n_exercise + 10, 1)
        self.e_linear = nn.Linear(2 * d_k, d_k)
        torch.nn.init.xavier_uniform_(self.e_linear.weight)

        self.multi_atten = MultiHeadAttention(d_model=d_k, output_dim=d_k, n_heads=4, dropout=dropout, kq_same=False)
        if use_at:
            self.linear_1 = nn.Linear(d_a + d_e + d_k, d_k)
        else:
            self.linear_1 = nn.Linear(d_a + d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_1.weight)
        self.linear_11 = nn.Linear(d_a + d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_11.weight)
        self.linear_2 = nn.Linear(3 * d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_2.weight)
        self.linear_3 = nn.Linear(3 * d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_3.weight)
        self.linear_4 = nn.Linear(2 * d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_4.weight)
        self.linear_5 = nn.Linear(2 * d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_5.weight)
        self.linear_6 = nn.Linear(3 * d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_6.weight)
        
        self.linear_pred = nn.Linear(2*d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_5.weight)
        self.linear_f = nn.Linear(2*d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_f.weight)

        self.linear_r_trans = nn.Linear(3*d_k, d_k)
        self.linear_w_trans = nn.Linear(3*d_k, d_k)
        self.gate_trans = nn.Linear(2*d_k, d_k)

        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        self.h_pre = nn.init.xavier_uniform_(torch.zeros(self.n_question + 1, self.d_k))
        
        self.time_encode = TimeEncode(d_k)
        

    def device(self):
        return next(self.parameters()).device
    def forward(self, e_data, at_data, r, it_data, c, ti_data):
        batch_size, seq_len = e_data.size(0), e_data.size(1)
        c_embed = self.c_embed(c)
        e_embed_data = self.e_linear(torch.cat((self.e_embed(e_data), c_embed),-1))
        at_embed_data = self.at_embed(at_data)
        it_embed_data = self.it_embed(it_data)
        ti_embed_data = self.time_encode(ti_data)
        a_data = r.view(-1, 1).repeat(1, self.d_a).view(batch_size, -1, self.d_a)
        h_pre = nn.init.xavier_uniform_(torch.zeros(self.n_question + 1, self.d_k)).repeat(batch_size, 1, 1).to(self.device())
        h_tilde_pre = None
        transition_pattern_pre = nn.init.xavier_uniform_(torch.zeros((1,self.d_k))).repeat(batch_size, 1).to(self.device())
        if self.use_at:
            all_learning = self.linear_1(torch.cat((e_embed_data, at_embed_data, a_data), 2))
        else:
            all_learning = self.linear_1(torch.cat((e_embed_data, a_data), 2))
        learning_pre = torch.zeros(batch_size, self.d_k).to(self.device())
        
        pred = torch.zeros(batch_size, seq_len).to(self.device())
        for t in range(0, seq_len - 1):
            
            e = e_data[:, t]
            q_e = F.one_hot(c[:,t], self.n_question + 1).unsqueeze(1).float()
            it = it_embed_data[:, t]
            learning = all_learning[:, t]
            
            if h_tilde_pre is None:
                h_tilde_pre = q_e.bmm(h_pre).view(batch_size, self.d_k)
            if t==0:
                forward_transfer = nn.init.xavier_uniform_(torch.zeros((batch_size,self.d_k)).to(self.device()))
                backward_transfer = nn.init.xavier_uniform_(torch.zeros(self.n_question + 1, self.d_k)).repeat(batch_size, 1, 1).to(self.device())
                concept_matrix = self.c_embed.weight
                correlation_weight = c_embed[:,t] @ concept_matrix.T
            else:
                query = torch.cat((c_embed[:,None,t], ti_embed_data[:,None,t]),-1)
                key = torch.cat((c_embed[:,:t], ti_embed_data[:,:t]),-1)
                # value_forward = all_learning[:,:t].unsqueeze(-2)
                value_backward = torch.cat((learning.unsqueeze(1).expand(-1,self.n_question + 1,-1), h_tilde_pre.unsqueeze(1).expand(-1,self.n_question + 1,-1), h_pre), -1)

                concept_matrix = self.c_embed.weight
                correlation_weight = c_embed[:,t] @ concept_matrix.T
                backward_transfer, for_atten_score = self.multi_atten(
                                    query, key, value_backward, correlation_weight, F.one_hot(c[:,:t], self.n_question + 1).transpose(1,2).float().to(self.device())
                            )
                pre_mask = F.one_hot(c[:,:t], self.n_question + 1).sum(1).bool()
                for_atten_score = for_atten_score.masked_fill(~pre_mask,-1e32)
                for_atten_score = F.softmax(for_atten_score, 1)

                forward_transfer = for_atten_score.view(-1,1)* h_pre.view(-1,h_pre.size(2))
                forward_transfer = torch.sum(forward_transfer.view(-1, h_pre.size(1), h_pre.size(2)), dim=1)
            
            lg_forward = self.linear_2(torch.cat((learning, forward_transfer, h_tilde_pre), 1))
            lg_forward = self.tanh(lg_forward)
            gamma_l_forward = self.linear_3(torch.cat((learning, forward_transfer, h_tilde_pre), 1))
            gamma_l_forward = self.sig(gamma_l_forward)
            LG_forward = gamma_l_forward * ((lg_forward + 1) / 2)
            
            LG_forward_tilde = self.dropout(q_e.transpose(1, 2).bmm(LG_forward.view(batch_size, 1, -1)))
            
            lg_backward = self.linear_4(torch.cat((backward_transfer, h_pre), -1))
            lg_backward = self.tanh(lg_backward)
            gamma_l_backward = self.linear_5(torch.cat((backward_transfer, h_pre), -1))
            gamma_l_backward = self.sig(gamma_l_backward)
            LG_backward = gamma_l_backward * ((lg_backward+1)/2)
            LG_backward = self.dropout(LG_backward)

            mask = F.one_hot(c[:,t], self.n_question + 1).unsqueeze(-1).expand(-1,-1,self.d_k).bool()
            LG_backward = LG_backward.masked_fill(mask, 0)
            if t == 0:
                LG_backward = 0.0
            LG_tilde = LG_forward_tilde + LG_backward

            # Forgetting Module

            n_skill = LG_tilde.size(1)
            temp = ti_embed_data[:,t] * ti_embed_data[:,t+1]
            gamma_f = self.sig(self.linear_6(torch.cat((
                h_pre,
                LG_tilde,
                temp.repeat(1, n_skill).view(batch_size, -1, self.d_k)
            ), 2)))
            h = LG_tilde + gamma_f * h_pre
            h_tilde = F.one_hot(c[:,t+1], self.n_question + 1).unsqueeze(1).float().bmm(h).view(batch_size, self.d_k)
            h_final = h_tilde
            y = self.sig(self.linear_pred(torch.cat((e_embed_data[:, t + 1], h_final), 1))).sum(1) / self.d_k
            pred[:, t + 1] = y
            
            
            # prepare for next prediction
            learning_pre = learning
            h_pre = h
            h_tilde_pre = h_tilde

        return pred, h_pre
    
    def predict(self, q, at, r, it, c, ti):
        q = q.masked_fill(q < 0, 0)
        r = r.masked_fill(r < 0, 0)
        c = c.masked_fill(c < 0, 0)
        return self(q, at, r, it, c, ti)
    def get_eval(self, q, at, r, it, c, ti, n=1):
        pred, *_ = self.predict(q, at, r, it,c, ti)
        mask = r[:,n:]>=0
        masked_pred = pred[:, n:][mask]
        masked_truth = r[:, n:][mask]
        return masked_pred, masked_truth
    def get_loss(self, criterion, q, at, r, it,c, ti, n=1):
        logits, *_ = self.predict(q, at, r, it,c, ti)
        mask = r[:,n:]>=0
        masked_labels = r[:,n:][mask].float()
        masked_logits = logits[:,n:][mask]
        return criterion(masked_logits, masked_labels)
        


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, output_dim, dropout, kq_same=True, bias=True):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.h = n_heads
        self.kq_same = kq_same
        self.q_linear = nn.Linear(2*d_model, d_model, bias=bias)
        if kq_same:
            self.k_linear = self.q_linear
        else:
            self.k_linear = nn.Linear(2*d_model, d_model, bias=bias)
        
        self.v_linear_forward = nn.Linear(d_model, d_model, bias=bias)
        self.v_linear_backward = nn.Linear(3*d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.proj_bias = bias

        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.out_proj_forward = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj_backward = nn.Linear(d_model, d_model, bias=bias)
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        self.lambdas = nn.Parameter(torch.rand(1, n_heads, 1, 1))

        torch.nn.init.xavier_uniform_(self.gammas)
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear_forward.weight)
        xavier_uniform_(self.v_linear_backward.weight)
        if self.kq_same is False:
            xavier_uniform_(self.q_linear.weight)

        if self.proj_bias:
            constant_(self.k_linear.bias, 0.)
            constant_(self.v_linear_forward.bias, 0.)
            constant_(self.v_linear_backward.bias, 0.)
            if self.kq_same is False:
                constant_(self.q_linear.bias, 0.)
            constant_(self.out_proj_forward.bias, 0.)
            constant_(self.out_proj_backward.bias, 0.)
            
    def forward(self, q, k, v_backward, correlation_weight, qe):
        bs = q.size(0)
        if self.kq_same is False:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)

        #attenion score
        d_k = k.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        qe_score = torch.matmul(qe[:,None,:,:].expand(-1, self.h,-1,-1), scores.transpose(-1,-2))
        for_correlation_weight = qe_score[:,0,:,0].contiguous()
        
        back_correlation_weight = self.lambdas*correlation_weight[:,None,:,None].expand(-1,self.h,-1,-1) + (1-self.lambdas)*qe_score
        back_correlation_weight = self.sig(back_correlation_weight)

        v_backward = self.v_linear_backward(v_backward).view(bs, -1, self.h, self.d_k)
        v_backward = v_backward.transpose(1, 2)
        v_backward_ = back_correlation_weight * v_backward
        
        concat_backward = v_backward_.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output_backward = self.out_proj_backward(concat_backward)
        return output_backward, for_correlation_weight
