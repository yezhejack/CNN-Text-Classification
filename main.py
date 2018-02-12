#!/bin/bash python
import model
import pickle
import numpy as np
import argparse
import model
import torch
from torch.optim import Adadelta
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
import time
is_cuda = torch.cuda.is_available()
print("CUDA is available={}".format(is_cuda))

def train_cnn(datasets, embeddings, epoches=25, batch_size=50, filter_h=5, max_l=56, seeds=[3435, 3435, 0], embedding_freeze=True):
    #shuffle dataset and assign to mini batches. if dataset size is not a multiple of mini batches, replicate
    #extra data (at random)
    np.random.seed(seeds[0])
    if datasets[0].shape[0] % batch_size > 0:
        extra_data_num = batch_size - datasets[0].shape[0] % batch_size
        train_set = np.random.permutation(datasets[0])
        extra_data = train_set[:extra_data_num]
        new_data=np.append(datasets[0],extra_data,axis=0)
    else:
        new_data = datasets[0]
    new_data = np.random.permutation(new_data)
    n_batches = new_data.shape[0]/batch_size
    n_train_batches = int(np.round(n_batches*0.9))
    #divide train set into train/val sets
    img_h = max_l + 2 * (filter_h - 1)
    test_set_x = datasets[1][:,:img_h]
    test_set_y = np.asarray(datasets[1][:,-1],"int32")
    n_test_batches = int(test_set_x.shape[0]/batch_size)
    if test_set_x.shape[0] % batch_size != 0:
        n_test_batches += 1
    train_set = new_data[:n_train_batches*batch_size,:]
    val_set = new_data[n_train_batches*batch_size:,:]
    train_set_x = train_set[:,:img_h]
    train_set_y = train_set[:,-1]
    val_set_x = val_set[:,:img_h]
    val_set_y = val_set[:,-1]
    n_val_batches = int(n_batches - n_train_batches)
    print("#Train:{} #Val:{}".format(train_set_x.shape[0], val_set_x.shape[0]))

    # define model
    cnn = model.CNN(embeddings, img_h, num_classes=2, init_seed=seeds[1], embedding_freeze=embedding_freeze)
    
    # set seed for training
    torch.cuda.manual_seed_all(seeds[2])

    if is_cuda:
        cnn.cuda()
    optimizer = Adadelta(cnn.trainable_params, rho=0.95)
    criterion = CrossEntropyLoss()
    best_val_acc = 0.0
    best_test_acc = 0.0
    for epoch in range(epoches):
        epoch_start_time = time.time()
        output_str = "[Epoch {}] ".format(epoch)

        right_counter = 0
        for minibatch_index in np.random.permutation(range(n_train_batches)):
            optimizer.zero_grad()
            X = Variable(torch.LongTensor(train_set_x[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]))
            y = Variable(torch.LongTensor(train_set_y[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]))
            if is_cuda:
                X = X.cuda()
                y = y.cuda()
            output = cnn.forward(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            cnn.normalize_fc_weight()

            _, pred_y = torch.max(output, 1)
            for pred, gold in zip(pred_y.data, y.data):
                if int(pred) == int(gold):
                    right_counter += 1
        output_str += "Train Acc:{:.2f}%\t".format(right_counter/float(train_set_x.shape[0]))

        right_counter = 0
        for minibatch_index in range(n_val_batches):
            X = Variable(torch.LongTensor(val_set_x[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]))
            y = Variable(torch.LongTensor(val_set_y[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]))
            if is_cuda:
                X = X.cuda()
                y = y.cuda()
            output = cnn.predict(X)
            loss = criterion(output, y)

            _, pred_y = torch.max(output, 1)
            for pred, gold in zip(pred_y.data, y.data):
                if int(pred) == int(gold):
                    right_counter += 1
        val_acc = right_counter/float(val_set_x.shape[0])
        output_str += "Val Acc:{:.2f}%\t".format(100*val_acc)

        right_counter = 0
        for minibatch_index in range(n_test_batches):
            X = Variable(torch.LongTensor(test_set_x[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]))
            y = Variable(torch.LongTensor(test_set_y[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]))
            if is_cuda:
                X = X.cuda()
                y = y.cuda()
            output = cnn.predict(X)
            loss = criterion(output, y)

            _, pred_y = torch.max(output, 1)
            for pred, gold in zip(pred_y.data, y.data):
                if int(pred) == int(gold):
                    right_counter += 1
        test_acc = right_counter/float(test_set_x.shape[0])
        output_str += "Test Acc:{:.2f}%".format(100*test_acc)
        print(output_str+" cost:{:.2f}s".format(time.time()-epoch_start_time))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
    return best_test_acc, best_val_acc

def get_idx_from_sent(sent, word_idx_map, max_l=56, k=300, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = filter_h - 1
    for i in range(pad):
        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l+2*pad:
        x.append(0)
    return x

def make_idx_data_cv(revs, word_idx_map, cv, max_l=56, k=300, filter_h=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    train = []
    test = []
    for rev in revs:
        sent = get_idx_from_sent(rev[b"text"], word_idx_map, max_l, k, filter_h)
        # last int represent label
        sent.append(rev[b"y"])
        if rev[b"split"]==cv:
            test.append(sent)
        else:
            train.append(sent)
    train = np.array(train,dtype="int")
    test = np.array(test,dtype="int")
    return [train, test]

if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--max_l", type=int, default=56)
    parser.add_argument("--embedding_freeze", action="store_true")
    args = parser.parse_args()
    torch.backends.cudnn.deterministic = True

    revs, W, W2, word_idx_map, vocab = pickle.load(open("mr.p", "rb"), encoding='bytes')

    print("Number of word_idx_map:{}".format(len(word_idx_map)))
    idx_to_word = ["" for i in range(len(W))]

    for word in word_idx_map:
        idx = word_idx_map[word]
        idx_to_word[idx]=word
    
    cv_acc = 0.0
    for r in range(10):
        cv_start_time = time.time()
        datasets = make_idx_data_cv(revs, word_idx_map, r, max_l=args.max_l, k=300, filter_h=5)
        print("CV:{} #Training Data:{} #Test Data:{}".format(r+1, len(datasets[0]), len(datasets[1])))
        best_test_acc, best_val_acc = train_cnn(datasets, W, batch_size=args.batch_size, seeds=[3435, 1, args.seed], embedding_freeze=args.embedding_freeze)
        print("Val Acc = {:.2f} Test Acc = {:.2f} Cost:{:.2f}\n".format(100*best_val_acc, 100*best_test_acc, time.time()-cv_start_time))
        cv_acc += best_test_acc
    print("Cross Validation Acc = {} Cost:{:2f}s".format(100*cv_acc/10, time.time()-start_time))
    with open("search_seed.csv", "a") as f:
        f.write("{}\t{}\n".format(args.seed, cv_acc/10))
        
    
