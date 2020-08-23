import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from data_reader import DataReader, Word2vecDataset
from model import SkipGramModel, CBOWModel


class Word2VecTrainer:
    def __init__(self, input_file, output_file, model, emb_dimension=100, batch_size=50, window_size=10, iterations=1,
                 initial_lr=0.001, min_count=5):

        self.data = DataReader(input_file, min_count)
        dataset = Word2vecDataset(self.data, window_size)
        self.window_size = window_size
        self.dataloader = DataLoader(dataset, batch_size=batch_size,
                                     shuffle=False, num_workers=0, collate_fn=dataset.collate)

        self.output_file_name = output_file
        self.emb_size = len(self.data.word2id)
        self.emb_dimension = emb_dimension
        self.batch_size = batch_size
        self.iterations = iterations
        self.initial_lr = initial_lr
        self.model_name = model

        self.model = CBOWModel(self.emb_size, self.emb_dimension, id2word=self.data.id2word, word2id=self.data.word2id) \
            if model=='CBOW' else SkipGramModel(self.emb_size, self.emb_dimension, id2word=self.data.id2word, word2id=self.data.word2id)
        
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda and model=='SKIP GRAM' else "cpu")
        if self.use_cuda and model=='SKIP GRAM':
            self.model.cuda()
        print('MODEL:', type(self.model), 'DEVICE:', self.device)

    def train(self):

        try:
          for iteration in range(self.iterations):

              print("\n\n\nIteration: " + str(iteration + 1))
              optimizer = optim.SparseAdam(self.model.parameters(), lr=self.initial_lr)
              scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(self.dataloader))

              running_loss = 0.0
              x = []
              y = []
              approx_words_trained = 0
              for i, sample_batched in enumerate(tqdm(self.dataloader)):
                  #print('i', i)
                  if len(sample_batched[0]) > 1:
                      pos_u = sample_batched[0].to(self.device)
                      pos_v = sample_batched[1].to(self.device)
                      neg_v = sample_batched[2].to(self.device)
                      approx_words_trained += sample_batched[0].shape[0]

                      if self.model_name == 'CBOW':
                        pos_u, index = torch.sort(pos_u)
                        pos_v = torch.gather(pos_v, dim=0, index=index)

                        index = index.unsqueeze(1).expand(len(index),neg_v.shape[1])
                        neg_v = torch.gather(input=neg_v, dim=0, index=index)

                      optimizer.zero_grad()
                      loss = self.model.forward(pos_u, pos_v, neg_v)
                      loss.backward()
                      optimizer.step()
                      scheduler.step()

                      running_loss = running_loss * 0.9 + loss.item() * 0.1
                      x.append(i)
                      y.append(running_loss)
                      if i > 0 and i % 5 == 0:
                        print(" Loss: " + str(running_loss))
                      
                      
        except KeyboardInterrupt:
          print('Keyboard Interrupt')
        finally:
          approx_words_trained /= self.window_size
          torch.save(self.model, 'model.obj')
          self.create_graph(approx_words_trained, x, y)
          self.model.save_embedding(self.data.id2word, self.output_file_name)
      
    def create_graph(self, words, x, y):
      plt.plot(x, y)
      plt.ylabel('Training loss')
      plt.xlabel('Iterations')
      string = 'Train ex:'+str(words)+' Emb dim:'+str(self.emb_dimension) \
        + ' L rate:' + str(self.initial_lr)
      plt.title(string)
      plt.show()


if __name__ == '__main__':
    w2v = Word2VecTrainer(input_file="Aristo-mini.txt", output_file="out.vec", model='CBOW')
    w2v.train()
