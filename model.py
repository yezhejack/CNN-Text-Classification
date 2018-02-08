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
                 l2_constraint=3):

        super(CNN,self).__init__()
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
                    nn.init.uniform(params, a=-0.01, b=0.01)
                else:
                    params.data.fill_(0.0)
            self.conv_list.append(conv)
        self.fc_dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(3*num_filters, num_classes)
        for params in self.fc.parameters():
            self.trainable_params.append(params)
            if params.data.dim() > 1:
                nn.init.uniform(params, a=-0.01, b=0.01)
            else:
                params.data.fill_(0.0)

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
        output=self.fc_dropout(output)
        output=self.fc(output)
        return output
    
    def normalize_fc_weight(self):
        for params in self.fc.parameters():
            if params.data.dim() == 2:
                l2_norm = torch.norm(params.data, p=2)
                if l2_norm > self.l2_constraint:
                    scale = self.l2_constraint / (l2_norm)
                    params.data *= scale