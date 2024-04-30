

import numpy as  np
import torch
import random
import os
import chardet
from sentence_transformers import SentenceTransformer, models, util
from sentence_transformers import losses
from sentence_transformers import datasets
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
import codecs
import sys
# from  pca import pca


# 载入预训练好的模型
def load_model(input_model_path, max_seq_length=128):

    word_embedding_model = models.Transformer(input_model_path, max_seq_length= 128)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), 'mean')
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    return model



def load_sentence2(path):
    sentences = []
    all_tokens_len = 0
    file_input = 'sentences_bert.txt'
    # path = sys.path[0]
    f = codecs.open(path + '/' + file_input, 'r', 'utf-8')
    for line in f.readlines():
        line = line.strip()
        if len(line) >= 30:
            sentences.append(line)
            all_tokens_len += len(line)
    print('number of sentenes is', len(sentences))
    print('average length of each sentences is :', all_tokens_len/len(sentences))
    print('number of tokens used is :', all_tokens_len)
    return  sentences


# 训练TSDAE模型
def train(model_location=''):
    path = sys.path[0]
    input_model_path = path + '/bert-base-chinese/'
    model = load_model(input_model_path)
    sentences = load_sentence2(path)

    model_output_path= './audit_bert_model/'
    train_dataset = datasets.DenoisingAutoEncoderDataset(sentences)
    train_dataloader = DataLoader(train_dataset, batch_size= 64, shuffle=True, drop_last=True, num_workers= 2)
    train_loss = losses.DenoisingAutoEncoderLoss(model, decoder_name_or_path = input_model_path, tie_encoder_decoder=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=15,
        weight_decay=0,
        scheduler='constantlr',
        optimizer_params={'lr': 4e-5},
        show_progress_bar=True,
        checkpoint_path = model_output_path,
        use_amp=True,
        checkpoint_save_steps=2000,
        warmup_steps=500,
        output_path = model_output_path + '/fineTuningBert/'

    )
    print('finish fine tuning!!!!')
    print('the model is save in :', model_output_path + '/fineTuningBert/')




def pca():
    new_dimension = 200
    path = sys.path[0]
    sentences = load_sentence2(path)
    random.shuffle(sentences)

    model = SentenceTransformer(path +'/fineTuningBert/')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    embeddings = model.encode(sentences, convert_to_numpy=True, show_progress_bar=True)
    pca = PCA(n_components=new_dimension)
    pca.fit(embeddings)
    pca_comp = np.asarray(pca.components_)
    dense = models.Dense(in_features=model.get_sentence_embedding_dimension(), out_features=new_dimension, bias=False, activation_function=torch.nn.Identity())
    dense.linear.weight = torch.nn.Parameter(torch.tensor(pca_comp))
    model.add_module('dense', dense)

    model.save(path + '/bert_pca-200')

def test_model(sens):
    path = sys.path[0]
    bert_dir =  path + '/fineTuningBert/'
    #bert_dir =  path + '/bert-base-chinese/'
    model = SentenceTransformer(bert_dir)
    #Compute embedding for both lists
    embeddings1 = model.encode(sens, convert_to_tensor=True)
    #Compute cosine-similarities
    cosine_scores = util.cos_sim(embeddings1, embeddings1)
    #Output the pairs with their score
    print(cosine_scores)



if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    #load_sentence()
    #train()
    #pca()
    sens = ['保险收入', '财务报表日后资本公积转增资本',
            '长期资产的减值准备', '长期银行借款',
            '加工贸易电子化手册','加工贸易业务流程',
            '违规提取个人住房公积金账号内存储余额','违规提取个人住房公积金账号' ]
    test_model(sens)


