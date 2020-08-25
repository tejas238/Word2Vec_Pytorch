import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

"""
    u_embedding: Embedding for center word.
    v_embedding: Embedding for neighbor words.
"""


class SkipGramModel(nn.Module):

    def __init__(self, emb_size, emb_dimension, id2word, word2id):
        super(SkipGramModel, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.u_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        self.v_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        self.id2word = id2word
        self.word2id = word2id

        initrange = 1.0 / self.emb_dimension
        init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        init.constant_(self.v_embeddings.weight.data, 0)

    def forward(self, pos_u, pos_v, neg_v):
        #print('pos_u shape', pos_u.shape)
        #print('pos_v shape', pos_v.shape)
        #print('neg_v shape', neg_v.shape)

        emb_u = self.u_embeddings(pos_u)
        emb_v = self.v_embeddings(pos_v)
        emb_neg_v = self.v_embeddings(neg_v)

        #print('emb_u shape', emb_u.shape)
        #print('emb_v shape', emb_v.shape)
        #print('emb_u*emb_v shape', torch.mul(emb_u, emb_v).shape)
        score = torch.sum(torch.mul(emb_u, emb_v), dim=1)
        score = torch.clamp(score, max=10, min=-10)
        score = -F.logsigmoid(score)

        neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)

        return torch.mean(score + neg_score)

    def save_embedding(self, id2word, file_name):
        embedding = self.u_embeddings.weight.cpu().data.numpy()
        with open(file_name, 'w') as f:
            f.write('%d %d\n' % (len(id2word), self.emb_dimension))
            for wid, w in id2word.items():
                e = ' '.join(map(lambda x: str(x), embedding[wid]))
                f.write('%s %s\n' % (w, e))


class CBOWModel(nn.Module):

    def __init__(self, emb_size, emb_dimension, id2word, word2id):
        super(CBOWModel, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.u_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        self.v_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        self.id2word = id2word
        self.word2id = word2id

        initrange = 1.0 / self.emb_dimension
        init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        init.constant_(self.v_embeddings.weight.data, 0)

    def forward(self, pos_u, pos_v, neg_v):
        #print('pos_u shape', pos_u.shape)
        #print('pos_v shape', pos_v.shape)
        #print('neg_v shape', neg_v.shape)

        emb_u = self.u_embeddings(pos_u)
        emb_v = self.v_embeddings(pos_v)
        emb_neg_v = self.v_embeddings(neg_v)

        #emb_u, emb_u_counts = torch.unique(emb_u, sorted=False, return_counts=True, dim=0)
        emb_u_counts = []
        emb_u_temp = []
        for u in emb_u:
          if any([(u == u_).all() for u_ in emb_u_temp]):
              emb_u_counts[-1] += 1
          else:
              emb_u_temp.append(u)
              emb_u_counts.append(1)
        emb_u = torch.stack(emb_u_temp)

        emb_v = torch.split(emb_v, emb_u_counts)
        emb_v = torch.stack([ torch.sum(v,dim=0) for v in emb_v])

        emb_neg_v = torch.split(emb_neg_v, emb_u_counts)
        emb_neg_v = [ torch.flatten(v, start_dim=0, end_dim=1) for v in emb_neg_v ]

        #print('emb_u shape', emb_u.shape)
        #print('emb_v shape', emb_v.shape)
        #print('emb_u*emb_v shape', torch.mul(emb_u, emb_v).shape)

        score = torch.sum(torch.mul(emb_u, emb_v), dim=1)
        score = torch.clamp(score, max=10, min=-10).squeeze()
        score = -F.logsigmoid(score)

        neg_score = torch.stack([ -torch.mean(F.logsigmoid(-torch.clamp( \
            torch.matmul( emb_neg_v[i], v.unsqueeze(1)), max=10, min=-10))) \
            for i,v in enumerate(emb_v)])

        #neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()
        #neg_score = torch.clamp(neg_score, max=10, min=-10)
        #neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)

        return torch.mean(score + neg_score)

    def save_embedding(self, id2word, file_name):
        embedding = self.u_embeddings.weight.cpu().data.numpy()
        with open(file_name, 'w') as f:
            f.write('%d %d\n' % (len(id2word), self.emb_dimension))
            for wid, w in id2word.items():
                e = ' '.join(map(lambda x: str(x), embedding[wid]))
                f.write('%s %s\n' % (w, e))
