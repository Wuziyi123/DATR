import torch
import torch.utils.data as data
import os
import numpy as np
import json
import torch.backends.cudnn as cudnn

from torch import nn
from collections import OrderedDict
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(self, data_path, data_split, vocab):
        self.vocab = vocab
        loc = data_path + '/'

        # load the raw captions
        self.captions = []
        with open(loc + '%s_precaps_stan.txt' % data_split, 'rb') as f:
            for line in f:
                self.captions.append(line.strip())

        # load the image features
        self.images = np.load(loc+'%s_ims.npy' % data_split)
        self.length = len(self.captions)

        if self.images.shape[0] != self.length:
            self.im_div = 5
        else:
            self.im_div = 1

        # the development set for coco is large and so validation would be slow
        if data_split == 'dev':
            self.length = 5000

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = index//self.im_div
        image = torch.Tensor(self.images[img_id])
        caption = self.captions[index]

        # convert caption (string) to word ids.
        cap = []
        cap.extend(caption.decode('utf-8').split(','))
        #cap = [int(item) for _,item in enumerate(cap)]
        cap = list(map(int, cap))

        caption = torch.Tensor(cap)

        return image, caption, index, img_id

    def __len__(self):
        return self.length


def collate_fn(data):
    """
    Build mini-batch tensors from a list of (image, caption, index, img_id) tuples.
    Args:
        data: list of (image, target, index, img_id) tuple.
            - image: torch tensor of shape (36, 2048).
            - target: torch tensor of shape (?) variable length.
    Returns:
        - images: torch tensor of shape (batch_size, 36, 2048).
        - targets: torch tensor of shape (batch_size, padded_length).
        - lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, ids, img_ids = zip(*data)

    # Merge images (convert tuple of 2D tensor to 3D tensor)
    images = torch.stack(images, 0)
    # Merge adjacent matrix
    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return images, targets, lengths, ids


def get_loaders(dpath, vocab, batch_size):
    # get the train_loader
    train_loader = get_precomp_loader(dpath, 'train', vocab,
                                      batch_size, True)
    # get the val_loader
    val_loader = get_precomp_loader(dpath, 'dev', vocab,
                                    100, False)
    return train_loader, val_loader


class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def deserialize_vocab(src):
    with open(src) as f:
        d = json.load(f)
    vocab = Vocabulary()
    vocab.word2idx = d['word2idx']
    vocab.idx2word = d['idx2word']
    vocab.idx = d['idx']
    return vocab


def get_precomp_loader(data_path, data_split, vocab, batch_size=100,
                       shuffle=True, num_workers=2):
    dset = PrecompDataset(data_path, data_split, vocab)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=collate_fn)
    return data_loader


class EncoderImage(nn.Module):
    """
    Build local region representations by common-used FC-layer.
    Args: - images: raw local detected regions, shape: (batch_size, 36, 2048).
    Returns: - img_emb: finial local region embeddings, shape:  (batch_size, 36, 1024).
    """
    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImage, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, embed_size)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer"""
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized
        img_emb = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            img_emb = l2norm(img_emb, dim=-1)

        return img_emb

    def load_state_dict(self, state_dict):
        """Overwrite the default one to accept state_dict from Full model"""
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImage, self).load_state_dict(new_state)


class EncoderText(nn.Module):
    """
    Build local word representations by common-used Bi-GRU or GRU.
    Args: - images: raw local word ids, shape: (batch_size, L).
    Returns: - img_emb: final local word embeddings, shape: (batch_size, L, 1024).
    """
    def __init__(self, vocab_size, word_dim, embed_size, num_layers,
                 use_bi_gru=False, no_txtnorm=False):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)
        self.dropout = nn.Dropout(0.4)

        # caption embedding
        self.use_bi_gru = use_bi_gru
        self.cap_rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True, bidirectional=use_bi_gru)

        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, captions, lengths):
        """Handles variable size captions"""
        # embed word ids to vectors
        cap_emb = self.embed(captions)
        cap_emb = self.dropout(cap_emb)

        # pack the caption
        packed = pack_padded_sequence(cap_emb, lengths, batch_first=True)

        # forward propagate RNN
        out, _ = self.cap_rnn(packed)

        # reshape output to (batch_size, hidden_size)
        cap_emb, _ = pad_packed_sequence(out, batch_first=True)

        if self.use_bi_gru:
            cap_emb = (cap_emb[:, :, :cap_emb.size(2)//2] + cap_emb[:, :, cap_emb.size(2)//2:])/2

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)

        return cap_emb


class CMCAN(object):

    def __init__(self, vocab_size):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Build Models
        self.vocab_size = vocab_size
        self.img_enc = EncoderImage(2048, 1024,
                                    no_imgnorm=False)
        self.txt_enc = EncoderText(self.vocab_size, 300,
                                   1024, 1,
                                   use_bi_gru=True,
                                   no_txtnorm=False)
        # self.sim_embt = GraphEmbt(opt.embed_size, opt.sim_dim)
        # self.sim_embv = GraphEmbv(opt.embed_size, opt.sim_dim)
        # self.sim_enc = EncoderSimilarity(opt.embed_size, opt.sim_dim)

        if torch.cuda.is_available():
            self.img_enc.to(self.device)
            self.txt_enc.to(self.device)
            cudnn.benchmark = True

        # Loss and Optimizer
        # self.criterion = ContrastiveLoss(margin=opt.margin,
        #                                  max_violation=opt.max_violation)
        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.parameters())
        # params += list(self.sim_embt.parameters())
        # params += list(self.sim_embv.parameters())
        # params += list(self.sim_enc.parameters())
        # self.params = params
        #
        # self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)
        # self.Eiters = 0

    def train_start(self):
        """switch to train mode"""
        self.img_enc.train()
        self.txt_enc.train()
        # self.sim_embt.train()
        # self.sim_embv.train()
        # self.sim_enc.train()

    def val_start(self):
        """switch to evaluate mode"""
        self.img_enc.eval()
        self.txt_enc.eval()
        # self.sim_embt.eval()
        # self.sim_embv.eval()
        # self.sim_enc.eval()

    def forward_emb(self, images, captions, lengths):
        """Compute the image and caption embeddings"""
        if torch.cuda.is_available():
            images = images.to(self.device)
            captions = captions.to(self.device)
        # Forward feature encoding
        img_embs = self.img_enc(images)
        cap_embs = self.txt_enc(captions, lengths)
        return img_embs, cap_embs, lengths

    def train_emb(self, images, captions, lengths, ids=None, *args):
        """One training step given images and captions.
        """
        # compute the embeddings
        img_embs, cap_embs, cap_lens = self.forward_emb(images, captions, lengths)
        # sims = self.forward_sim(img_embs, cap_embs, cap_lens, adjs, depends)
        #
        # # measure accuracy and record loss
        # self.optimizer.zero_grad()
        # loss = self.forward_loss(sims)
        #
        # # compute gradient and do SGD step
        # loss.backward()
        # if self.grad_clip > 0:
        #     clip_grad_norm_(self.params, self.grad_clip)
        # self.optimizer.step()


if __name__ == "__main__":
    dpath = os.path.join("./DATA/", "f30k_precomp")

    # get the train_loader
    vocab = deserialize_vocab(os.path.join('./DATA/vocab/', '%s_vocab.json' % 'f30k_precomp'))
    vocab_size = len(vocab)

    train_loader, val_loader = get_loaders(dpath, vocab, 48)

    model = CMCAN(vocab_size)

    for i in range(60):
        for i, train_data in enumerate(train_loader):
            # switch to train mode
            model.train_emb(*train_data)
