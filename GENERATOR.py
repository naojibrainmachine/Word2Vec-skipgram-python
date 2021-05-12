import numpy as np
import random
import tensorflow as tf
import math
data_index=0
def generate_batch(data_indices,batch_size,skip_window):
    '''
    生成一个个batch，batch由数个中心词，和它窗口大小的词组成
    '''
    def get_context(data_indices,k):
        #print(data_indices)
        #if(end!=len(data_indices)-1)
        #    end=len(data_indices)-1+2
        #if(start!=0)
        #    start=start+2
        #print(data_indices,"data_indices")
        Y1=[]
        if(k-skip_window<0):
            for j in range(0,k+skip_window+1):
                if(j!=k):
                    
                    Y1.append(data_indices[j])
                    #print(Y1,"i==0")
        elif k+skip_window>len(data_indices)-1:
            for j in range(k-skip_window,len(data_indices)):
                if(j!=k):
                   #print(Y1,"i==0")
                    Y1.append(data_indices[j])
                    #print(Y1,"i==len")
        else:
            for j in range(k-skip_window,k+skip_window+1):
                #print(range(i-skip_window,i+skip_window+1))
                if(j!=k):
                    #print(j,"j")
                    Y1.append(data_indices[j])
                    #print(Y1," safasf")
        return Y1
    num_example=math.ceil(len(data_indices)/batch_size)

    example_indices=list(range(num_example))
    random.shuffle(example_indices)
    
    #print(data_indices)
    #print(len(data_indices))
    #print(len(data_indices))
    #print(example_indices)
    for i in example_indices:
        start=i*batch_size
        if start >(len(data_indices)-1):
            start=(len(data_indices)-1)
            
        #start= start if start <=(len(data_indices)-1) else (len(data_indices)-1)
        end=i*batch_size+batch_size
        if end >(len(data_indices)-1):
            end=(len(data_indices)-1)+1
        #end= end if end <=(len(data_indices)-1) else (len(data_indices)-1)
        X=data_indices[start:end]
        Y=[]
        #print(start,"start",end,"end")
        #print(X)
        for k in range(start,end):
            y=get_context(data_indices,k)
            if len(y)<2*skip_window:
                for j1 in range(0,2*skip_window-len(y)):
                    #print(i)
                    #print(y,"y[j-len(y)]")
                    y.append(y[j1])
                    #print(y,"y[j-len(y)]")
            Y.append(tf.reshape(y,[1,2*skip_window]))
            #print(Y,"Y")
        
        
        yield X,tf.concat(Y,0)#tf.reshape(Y,[X.shape[0],skip_window*2])

def init_unigram_table(vocab_size,voca_frequency,idx_to_chars,sample_norm=0.001,table_size=10e8):
    '''
    构建负采样的概率表
    voca_frequency为带有频率的词汇表
    sampling_pro：词汇保留的概率
    table：负采样概率表
    
    '''
    
    sampling_pro=voca_frequency#sampling_pro为抽样率，指词汇被留下的概率
    for key in list(sampling_pro.keys()):
        sampling_pro[key]=(math.sqrt((sampling_pro[key]/sample_norm))+1)*(sample_norm/sampling_pro[key])
    
    train_words_pow = 0
    power = 0.75
    d1=0.0
    table=[]#一元模型分布
    
    #pow(x, y)计算x的y次方;train_words_pow表示总的词的概率，不是直接用每个词的频率，而是频率的0.75次方幂
    for a in range(vocab_size):
        train_words_pow += math.pow(sampling_pro[idx_to_chars[a]], power)
    
    i = 0
    d1 = pow(sampling_pro[idx_to_chars[a]], power) / train_words_pow
    #每个词在table中占的小格子数是不一样的，频率高的词，占的格子数显然多
    for a in range(int(table_size)):
        table.append(i)
        if (a / table_size > d1):
            i+=1
            if (i < vocab_size):
                d1 += math.pow(sampling_pro[idx_to_chars[i]], power) / train_words_pow
        if (i >= vocab_size):
            i = vocab_size - 1

    return table 
