from collections import Counter
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.utils.class_weight import compute_class_weight 
from tqdm import tqdm
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import MultiStepLR
import sys
import shutil


MODEL_1_PATH = "./blstm1.pt"
MODEL_2_PATH = "./blstm2.pt"

# piecewise accuracy 
def accuracy(outputs, labels):
    acc = 0
    count = 0
    for i in range(outputs.shape[0]):
        sentence_pred = outputs[i]
        for j, word in enumerate(sentence_pred):
            word_pred = torch.argmax(word).item()
            label = labels[i][j].item()
            if label == -1:
                continue
            count += 1
            if word_pred == label:
                acc += 1
    return acc/count

#evaluate function for dev test during training
def evaluate(model, criterion, dataloader, device):
  with torch.no_grad():
    dev_loss, dev_acc, dev_f1 = 0.0, 0.0, 0.0
    for batch_x, batch_y in tqdm(dataloader):
      batch_x = batch_x.to(device)
      batch_y = batch_y.to(device)
      outputs = model(batch_x)
      seq_lengths = torch.count_nonzero(batch_x, dim=1).to('cpu')
      packed_y = pack_padded_sequence(batch_y, seq_lengths, batch_first=True, enforce_sorted=False)
      unpacked_y, unpacked_len = pad_packed_sequence(packed_y, batch_first=True, padding_value=-1)
      unpacked_y = unpacked_y.to(device)
      loss = criterion(outputs.permute(0, 2, 1), unpacked_y)
      dev_loss += loss.item()
      out_for_f1 = torch.argmax(outputs, dim = -1)
      mask = (unpacked_y >= 0)
      f1 = f1_score(out_for_f1[mask].cpu(), unpacked_y[mask].cpu(), average='weighted')
#       acc = accuracy(outputs, batch_y) -->costly operation, uncomment to see accuracy
#       dev_acc += acc
      dev_f1 += f1
    dev_loss /= len(dataloader)
    dev_acc /= len(dataloader)
    dev_f1 /= len(dataloader)

    print(f"Average Dev Loss: {dev_loss}")
#     print(f"Average Dev accuracy: {dev_acc}")
    print(f"Average Dev F1: {dev_f1}")
    return dev_loss, dev_f1
  
def make_dev_for_perl_helper(idx2tag, reverse_test_word2idx_untouched, outputs, batch_x_untouched, batch_gold, batch_ind, file):
    acc = 0
    for i in range(outputs.shape[0]):
        sentence_pred = outputs[i]
        one_x = batch_x_untouched[i]
        sentence_index = batch_ind[i]
        goldens = batch_gold[i]
        for j, word_probs in enumerate(sentence_pred):
            word_pred = torch.argmax(word_probs).item()
            # tag = tag2idx[str(word_pred)]
            # print(sentence_index)
            index = sentence_index[j].item()
            # with open("")
            if index == 0:
              break
            word = reverse_test_word2idx_untouched[one_x[j].item()]
            tag = idx2tag[word_pred]
            gold = goldens[j].item()
            gold = idx2tag[gold]

            if index == 1:
              # print('\n')
              file.write('\n')
            file.write(str(index) + ' ' + word + ' ' + gold+ ' ' + tag + '\n')
            # print(index, word, tag)

def make_dev_for_perl(model, idx2tag, reverse_test_word2idx_untouched, dataloader, file_name, device):
  with torch.no_grad():
    with open(file_name, 'w') as file:
      dev_acc = 0.0
      for batch_x, batch_gold, batch_ind, batch_x_untouched in tqdm(dataloader):
        batch_x = batch_x.to(device)
        batch_gold = batch_gold.to(device)
#         print(batch_x.shape)
        outputs = model(batch_x)
        make_dev_for_perl_helper(idx2tag, reverse_test_word2idx_untouched, outputs, batch_x_untouched, batch_gold, batch_ind, file)
        acc = accuracy(outputs, batch_gold)
        dev_acc += acc
    dev_acc /= len(dataloader)
    print(f"Average accuracy: {dev_acc}")

def make_output_file_helper(idx2tag, reverse_test_word2idx_untouched, outputs, batch_x_untouched, batch_ind, file):
    for i in range(outputs.shape[0]):
        sentence_pred = outputs[i]
        one_x = batch_x_untouched[i]
        sentence_index = batch_ind[i]
        for j, word_probs in enumerate(sentence_pred):
            word_pred = torch.argmax(word_probs).item()
            # tag = tag2idx[str(word_pred)]
            # print(sentence_index)
            index = sentence_index[j].item()
            # with open("")
            if index == 0:
              break
            word = reverse_test_word2idx_untouched[one_x[j].item()]
            tag = idx2tag[word_pred]

            if index == 1:
              # print('\n')
              file.write('\n')
            file.write(str(index) + ' ' + word + ' ' + tag + '\n')
            # print(index, word, tag)

def make_output_file(model, idx2tag, reverse_test_word2idx_untouched, dataloader, file_name, device):
  with torch.no_grad():
    with open(file_name, 'w') as file:
      for batch_x, batch_x_untouched, batch_ind in tqdm(dataloader):
        batch_x = batch_x.to(device)
        outputs = model(batch_x)
        make_output_file_helper(idx2tag, reverse_test_word2idx_untouched, outputs, batch_x_untouched, batch_ind, file)
  

#Make a wrapper class for our Dataset -- Take care of no tag in test data
#Make a wrapper class for our Dataset -- Take care of no tag in test data

class NERDataset(Dataset):
    def __init__(self, data, word2idx, tag2idx, mode = 'train', test_word2idx_untouched = None):
        self.data = data
        self.word2idx = word2idx
        self.tag2idx = tag2idx
        self.maxlen=125
        self.mode = mode
        self.test_word2idx_untouched = test_word2idx_untouched

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mode = self.mode
        if mode == 'train' or mode == 'devtrain' or mode == 'dev_out' or mode == 'dev_perl':
          index, words, tags = self.data[idx]
        if mode == 'test_out':
          index, words, _ = self.data[idx]

        
        # print(words)
        temp = [self.word2idx.get(word, self.word2idx['<unk>']) for word in words]
        if len(temp) < self.maxlen:
            pad_arr = np.pad(temp, (0, self.maxlen-len(temp)), 'constant', constant_values=(-1, 0))
        elif len(temp) > self.maxlen:
            pad_arr = temp[:self.maxlen]
        else:
            pad_arr = temp
        x = torch.tensor(pad_arr)

        if mode == 'test_out' or mode =='dev_out' or mode == 'dev_perl':
#           words_untouched = deepcopy(words)
#           print(word)
          temp = [self.test_word2idx_untouched.get(word, -1) for word in words]
          # print(index)
          if len(temp) < self.maxlen:
              pad_arr = np.pad(temp, (0, self.maxlen-len(temp)), 'constant', constant_values=(-1, 0))
          elif len(temp) > self.maxlen:
              pad_arr = temp[:self.maxlen]
          else:
              pad_arr = temp
          k = torch.tensor(pad_arr)

        # print(words)
        # print(index)
        temp = [int(ind) for ind in index]
        if len(temp) < self.maxlen:
            pad_arr = np.pad(temp, (0, self.maxlen-len(temp)), 'constant', constant_values=(-1, 0))
        elif len(temp) > self.maxlen:
            pad_arr = temp[:self.maxlen]
        else:
            pad_arr = temp
        z = torch.tensor(pad_arr)

        if mode == 'train' or mode == 'devtrain' or mode =="dev_out" or mode == 'dev_perl':
          temp = [self.tag2idx[tag] for tag in tags]
          if len(temp) < self.maxlen:
              pad_arr = np.pad(temp, (0,self.maxlen-len(temp)), 'constant', constant_values=(-1, -1))
          elif len(temp) > self.maxlen:
              pad_arr = temp[:self.maxlen]
          else:
              pad_arr = temp
          y = torch.LongTensor(pad_arr)
        
        if mode == 'train' or mode == 'devtrain':
          return x, y
        # print(x, z, k)
        if mode == 'test_out' or mode == 'dev_out':
            return x, k, z
        if mode == 'dev_perl':
            return x, y, z, k


#Define Model
class BLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout, pretrained_embedding = None):
        super(BLSTM, self).__init__()
        if pretrained_embedding:
            self.embedding = pretrained_embedding
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
#         init.xavier_uniform_(self.embedding.weight)
        self.blstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(hidden_dim * 2, 128)
        self.activation = nn.ELU()
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        # print(x.shape)
        embedded = self.embedding(x)
        seq_lengths = torch.count_nonzero(x, dim=1).cpu()
        x = pack_padded_sequence(embedded, seq_lengths, batch_first=True, enforce_sorted=False)
        x, _ = self.blstm(x)
        x, unpacked_len = pad_packed_sequence(x, batch_first=True)
        x = self.dropout1(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.classifier(x)
        return x

#Load train, test and dev data as list of tuples
def load_data(file_path, mode):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        index, words, tags = [], [], []
        maxlen = 0
        for line in f:
            line = line.strip()
            if not line:
                if words:
                    data.append((index, words, tags))
                    maxlen = max(maxlen, len(words))
                    index, words, tags = [], [], []
            else:
                if mode == 'train' or mode == 'dev':
                  ind, word, tag = line.split(' ')[0], line.split(' ')[1], line.split(' ')[2]
                elif mode == 'test':
                  ind, word = line.split(' ')[0], line.split(' ')[1]

                if mode == 'test':
                  index.append(ind)
                  words.append(word)
                if mode == 'train' or mode == 'dev':
                  index.append(ind)
                  words.append(word)
                  tags.append(tag)
                
    print(f"max sentence length = {maxlen}")
    return data

# Build vocabulary and word<->tag maps 
def build_vocab(data):
    word_counts = Counter(word for _, sentence, _ in data for word in sentence)
    filtered_dict = {key: value for key, value in word_counts.items()}
    vocabulary = ['<pad>', '<unk>'] + sorted(filtered_dict)
    word2idx = {word: idx for idx, word in enumerate(vocabulary)}
    return vocabulary, word2idx

def build_tag_map(data):
    tags = set(tag for _,_, tags in data for tag in tags)
    # tags.add('<pad>')
    tag2idx = {tag: idx for idx, tag in enumerate(sorted(tags))}
    return tag2idx

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def get_class_weights(data, device):
    all_y = []
    for data in data:
        all_y.extend(data[2])

    class_weights = compute_class_weight(
                                            class_weight = "balanced",
                                            classes = np.unique(all_y),
                                            y = all_y                                                    
                                        )
    class_weights=torch.tensor(class_weights,dtype=torch.float).to(device)
    return class_weights


def train(model, train_loader, dev_loader, optimizer, criterion, scheduler, device, epochs, save_path):
    model.train()
    SAVE_PATH = save_path
    best_f1 = -1
    for epoch in range(epochs):
      print(f"Epoch: {epoch}")
      train_loss, train_acc, train_f1 = 0.0, 0.0, 0.0
      for batch_x, batch_y in tqdm(train_loader):
          batch_x = batch_x.to(device)
          batch_y = batch_y.to(device)
          outputs = model(batch_x)

          seq_lengths = torch.count_nonzero(batch_x, dim=1).to('cpu')
          packed_y = pack_padded_sequence(batch_y, seq_lengths, batch_first=True, enforce_sorted=False)
          unpacked_y, unpacked_len = pad_packed_sequence(packed_y, batch_first=True, padding_value = -1)
          unpacked_y = unpacked_y.to(device)
            
          loss = criterion(outputs.permute(0, 2, 1), unpacked_y)
          train_loss += loss.item()
  
        #   mask = (unpacked_y >= 0) -->costly operation, uncomment to see accuracy, f1
        #   acc = accuracy(outputs, batch_y)
        #   out_for_f1 = torch.argmax(outputs, dim = -1)
        #   f1 = f1_score(out_for_f1[mask].cpu(), unpacked_y[mask].cpu(), average='weighted')
        #   train_acc += acc
        #   train_f1 += f1

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

      train_loss /= len(train_loader)
    #   train_acc /= len(train_loader)
    #   train_f1 /= len(train_loader)
      
    #   print(f"Average train accuracy: {train_acc}")
    #   print(f"Average train f1: {train_f1}")
      val_loss, val_f1 = evaluate(model, criterion, dev_loader, device)
      if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), SAVE_PATH)
      scheduler.step(val_loss)
      print(f"Average train Loss: {train_loss}")
      print(f"Current Learning Rate: {get_lr(optimizer)}")
      print(f"Best sklearn masked F1: {best_f1}")

    return model

def build_test_vocab(data):
    # print(data)
    word_counts = Counter(word for _, sentence, _ in data for word in sentence)
    vocabulary = ['<pad>', '<unk>'] + sorted(word_counts)
    word2idx = {word: idx for idx, word in enumerate(vocabulary)}
    return vocabulary, word2idx

def load_glove():
    print('Loading Glove...')
    embeddings_index = {}
    with open('./checkpoints/glove.6B.100d', 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0].lower()
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

def make_weight_matrix(vocabulary, word2idx, embeddings_index):
    weights_matrix = np.zeros((len(vocabulary), 100))
    hits = misses = 0
    # Initialize the unk and pad vector randomly using a normal distribution
    unk_weight = np.random.normal(scale=0.8, size=(100,))
    pad_weight = np.random.normal(scale=0.8, size=(100,))
#     pad_weight = np.zeros(100)
    for word, i in word2idx.items():
    #     print(word)
        embedding_vector = embeddings_index.get(word.lower())
        if embedding_vector is not None:
            weights_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1
            if word == '<pad>':
                weights_matrix[i] = pad_weight
            else:
                weights_matrix[i] = unk_weight
    print(f"Hits: {hits} Misses: {misses} Hit Ratio: {hits/(hits+misses)}")
    return weights_matrix

def load_best_blstm(vocabulary, tag2idx, device, path):
    model = BLSTM(len(vocabulary), embedding_dim=100, hidden_dim=256, output_dim=len(tag2idx), dropout=0.33).to(device)
    # model.load_state_dict(torch.load(path))
    model.load_state_dict(torch.load(path, map_location=torch.device(device)))
    model.eval()
    return model

def train_blstm_with_glove(vocabulary, word2idx, tag2idx, class_weights, train_loader, dev_loader, device, epochs):
    embeddings_index = load_glove()
    weights_matrix = make_weight_matrix(vocabulary, word2idx, embeddings_index)
    embedding_layer = nn.Embedding(len(vocabulary), 100)
    embedding_layer.weight.data.copy_(torch.from_numpy(weights_matrix))
    embedding_layer.weight.requires_grad = True
    model = BLSTM(len(vocabulary), embedding_dim=100, hidden_dim=256, output_dim=len(tag2idx), dropout=0.33, pretrained_embedding = embedding_layer).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9, weight_decay = 1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.55, patience = 10, threshold=0.01, verbose=True, min_lr=5e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean', weight=class_weights)
    model = train(model, train_loader, dev_loader, optimizer, criterion, scheduler, device, epochs = epochs, save_path=MODEL_2_PATH)
    return model

def train_blstm_vanilla(vocabulary, word2idx, tag2idx, class_weights, train_loader, dev_loader, device, epochs):
    model = BLSTM(len(vocabulary), embedding_dim=100, hidden_dim=256, output_dim=len(tag2idx), dropout=0.33).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9, weight_decay = 1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.55, patience = 3, threshold=0.1, verbose=True, min_lr=5e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean', weight=class_weights)
    model = train(model, train_loader, dev_loader, optimizer, criterion, scheduler, device, epochs = epochs, save_path=MODEL_1_PATH)
    return model


        
def print_utility(model, device, dev_data, test_data, word2idx, tag2idx, batch_size, file_name, mode):
    #DEFINE MODE
    if mode == 'test_out':
        mode_data = test_data
    elif mode == 'dev_out' or mode == 'dev_perl':
        mode_data = dev_data

    _, test_word2idx_untouched = build_test_vocab(mode_data)
    reverse_test_word2idx_untouched = {v: k for k, v in test_word2idx_untouched.items()}
    idx2tag = {v: k for k, v in tag2idx.items()}

    print_dataset = NERDataset(mode_data, word2idx, tag2idx, mode, test_word2idx_untouched)
    print_loader = DataLoader(print_dataset, batch_size=batch_size)

    if mode == 'dev_perl':
        make_dev_for_perl(model, idx2tag, reverse_test_word2idx_untouched, print_loader, file_name, device)

    if mode == 'dev_out':
        make_output_file(model, idx2tag, reverse_test_word2idx_untouched, print_loader, file_name, device)    

    if mode == 'test_out':
        make_output_file(model, idx2tag, reverse_test_word2idx_untouched, print_loader, file_name, device)    


def main():
    # function = sys.argv[1:][0]
    function = 'train'

    epochs = 300

    train_data = load_data('./checkpoints/data/train', mode = 'train')
    dev_data = load_data('./checkpoints/data/dev', mode = 'dev')
    test_data = load_data('./checkpoints/data/test', mode = 'test')
    vocabulary, word2idx = build_vocab(train_data + dev_data + test_data)
    tag2idx = build_tag_map(train_data + dev_data)

    train_dataset = NERDataset(train_data, word2idx, tag2idx, mode = 'train')
    dev_dataset = NERDataset(dev_data, word2idx, tag2idx, mode = 'devtrain')

    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    class_weights = get_class_weights(train_data + dev_data, device)


    if function == 'train':

        print("\nTraining Glove BLSTM Model....\n")
        blstm_glove_model = train_blstm_with_glove(vocabulary, word2idx, tag2idx, class_weights, train_loader, dev_loader, device, epochs)                                     
        blstm_glove_model = load_best_blstm(vocabulary, tag2idx, device, MODEL_2_PATH)
        print("\nEvaluating and making files for Vanilla BLSTM Model....\n")
        print_utility(blstm_glove_model, device, dev_data, test_data, word2idx, tag2idx, batch_size, './checkpoints/dev2_perl.out', 'dev_perl')
        print_utility(blstm_glove_model, device, dev_data, test_data, word2idx, tag2idx, batch_size, './checkpoints/dev2.out', 'dev_out')
        print_utility(blstm_glove_model, device, dev_data, test_data, word2idx, tag2idx, batch_size, './checkpoints/test2.out', 'test_out')

        src = MODEL_2_PATH
        dst = './checkpoints/blstm2.pt'
        shutil.copyfile(src, dst)

        print("\nTraining Vanilla BLSTM Model....\n")
        blstm_vanilla_model =  train_blstm_vanilla(vocabulary, word2idx, tag2idx, class_weights, train_loader, dev_loader, device, epochs)
        blstm_vanilla_model = load_best_blstm(vocabulary, tag2idx, device, MODEL_1_PATH)
        print("\nEvaluating and making files for Vanilla BLSTM Model....\n")
        print_utility(blstm_vanilla_model, device, dev_data, test_data, word2idx, tag2idx, batch_size, './checkpoints/dev1_perl.out', 'dev_perl')
        print_utility(blstm_vanilla_model, device, dev_data, test_data, word2idx, tag2idx, batch_size, './checkpoints/dev1.out', 'dev_out')
        print_utility(blstm_vanilla_model, device, dev_data, test_data, word2idx, tag2idx, batch_size, './checkpoints/test1.out', 'test_out')


        src = MODEL_1_PATH
        dst = './checkpoints/blstm1.pt'
        shutil.copyfile(src, dst)
    
    if function == 'load':
        print("\nEvaluating and making files for Vanilla BLSTM Model....\n")
        blstm_vanilla_model = load_best_blstm(vocabulary, tag2idx, device, MODEL_1_PATH)
        print_utility(blstm_vanilla_model, device, dev_data, test_data, word2idx, tag2idx, batch_size, './checkpoints/dev1_perl.out', 'dev_perl')
        print_utility(blstm_vanilla_model, device, dev_data, test_data, word2idx, tag2idx, batch_size, './checkpoints/dev1.out', 'dev_out')
        print_utility(blstm_vanilla_model, device, dev_data, test_data, word2idx, tag2idx, batch_size, './checkpoints/test1.out', 'test_out')
        blstm_glove_model = load_best_blstm(vocabulary, tag2idx, device, MODEL_2_PATH)
        print("\nEvaluating and making files for Vanilla BLSTM Model....\n")
        print_utility(blstm_glove_model, device, dev_data, test_data, word2idx, tag2idx, batch_size, './checkpoints/dev2_perl.out', 'dev_perl')
        print_utility(blstm_glove_model, device, dev_data, test_data, word2idx, tag2idx, batch_size, './checkpoints/dev2.out', 'dev_out')
        print_utility(blstm_glove_model, device, dev_data, test_data, word2idx, tag2idx, batch_size, './checkpoints/test2.out', 'test_out')
    

       


if __name__ == "__main__":
   main()

