import os
import fool
import math
import numpy as np
def load_data(path):
    """
    加载数据
    path为数据目录路径
    需要用fool工具分词
    
    """
    voca_frequency={}
    f=open(path,encoding='utf-8')
    corpus=f.read()
    corpus=corpus.replace('\n','').replace('\r','').replace(' ','').replace('\u3000','')
    #corpus_chars=corpus[0:30000]
    corpus_chars=fool.cut(corpus_chars)
    corpus_chars=corpus_chars[0]

    #获得词频表
    sum_num=0
    for item in corpus_chars:
        if item not in voca_frequency.keys():
            voca_frequency[item]=1
        elif item in voca_frequency.keys():
            voca_frequency[item]=voca_frequency[item]+1
        sum_num+=1

    for key in list(voca_frequency.keys()):
        voca_frequency[key]=voca_frequency[key]/sum_num
    
    voca_frequency=dict(sorted(voca_frequency.items(),key= lambda var:var[1],reverse=True))#频率词汇表为降序
   
    idx_to_chars=list(voca_frequency.keys())#把语料转化成列表
    
    chars_to_idx=dict([(char,i) for i,char in enumerate(idx_to_chars)])#词库中字符到索引的映射

    vocab_size=len(idx_to_chars)#词库大小

    corpus_indices=[chars_to_idx[char] for char in corpus_chars]#预料索引，既读入的文本，并通过chars_to_idx转化成索引
    
    return corpus_indices,chars_to_idx,idx_to_chars,vocab_size,voca_frequency


   

