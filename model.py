#!/bin/bash python
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.autograd import Variable
import torch.nn.utils.weight_norm as weight_norm
import numpy as np

class CNN(nn.Module):
    def __init__(self, embeddings, sentence_size, 
                 filters=[3,4,5], 
                 num_filters=100, 
                 embedding_freeze=True, 
                 num_classes=3,
                 l2_constraint=3.0,
                 dropout_p=0.5,
                 init_seed=1):

        super(CNN,self).__init__()
        self.dropout_p = dropout_p
        rng = np.random.RandomState(init_seed)
        self.l2_constraint=l2_constraint
        self.trainable_params = []
        vocab_size=embeddings.shape[0]
        embed_size=embeddings.shape[1]
        self.embed=nn.Embedding(vocab_size, embed_size)
        torch_wv = torch.from_numpy(embeddings)

        if embedding_freeze == False:
            self.embed.weight=nn.Parameter(torch_wv)
            for param in self.embed.parameters():
                self.trainable_params.append(param)
        else:
            self.embed.weight=nn.Parameter(torch_wv)

        self.conv_list=nn.ModuleList()
        self.sentence_size=sentence_size
        self.filters=filters
        self.num_filters=num_filters
        for filter in filters:
            conv = nn.Conv2d(1, num_filters, (filter, embed_size))
            for params in conv.parameters():
                self.trainable_params.append(params)
                if params.data.dim()>1:
                    params.data = torch.FloatTensor(rng.uniform(low=-0.01, high=0.01, size=params.data.shape))
                else:
                    params.data.fill_(0.0)
            self.conv_list.append(conv)
        
        self.fc_w = nn.Parameter(torch.zeros([num_classes, len(filters)*num_filters]))
        self.fc_b = nn.Parameter(torch.zeros(num_classes))
        self.trainable_params.append(self.fc_w)
        self.trainable_params.append(self.fc_b)

    def forward(self,x):
        x=self.embed(x)
        x=x.view(x.size(0),1,x.size(1),x.size(2))
        convs_output=[]
        for filter,conv in zip(self.filters,self.conv_list):
            conv_output=conv(x)
            pooling_output=F.max_pool2d(F.relu(conv_output),(self.sentence_size-filter+1,1))
            pooling_output=pooling_output.view(-1,self.num_filters)
            #pooling_output=F.relu(pooling_output)
            convs_output.append(pooling_output)
        output=torch.cat(convs_output,1)
        output=F.dropout(output, p=self.dropout_p, training=True)
        output=F.linear(output, self.fc_w, self.fc_b)
        return output
    
    def normalize_fc_weight(self):
        l2_norm = torch.norm(self.fc_w.data, p=2)
        if l2_norm > self.l2_constraint:
            scale = self.l2_constraint / (l2_norm)
            self.fc_w.data *= scale
    
    def predict(self, x):
        x=self.embed(x)
        x=x.view(x.size(0),1,x.size(1),x.size(2))
        convs_output=[]
        for filter,conv in zip(self.filters,self.conv_list):
            conv_output=conv(x)
            pooling_output=F.max_pool2d(F.relu(conv_output),(self.sentence_size-filter+1,1))
            pooling_output=pooling_output.view(-1,self.num_filters)
            #pooling_output=F.relu(pooling_output)
            convs_output.append(pooling_output)
        output=torch.cat(convs_output,1)
        #output=F.linear(output, self.fc_w, self.fc_b)
        output=F.linear(output, (1-self.dropout_p)*self.fc_w, self.fc_b)
        return output