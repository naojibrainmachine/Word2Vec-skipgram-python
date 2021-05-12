import tensorflow as tf
import collections
import numpy as np
import random
import fool
import argparse
from LOADDATA import load_data
from GENERATOR import generate_batch,init_unigram_table



class skGram:
    '''
    该类主要完成上下文的预测和训练
    '''
    def __init__(self,batch_size,embedding_size,vocabulary_size,lr,pro_table):
        '''
        embedding_size:是词向量的大小
        vocabulary_size:字典大小
        batch_size:批次大小
        lr:学习率
        pro_table:概率表，通过抽样负样本
        '''
        random.seed(0)

        self.center_word=-1#中心词

        self.context_word=-1#上下文

        self.batch_size=batch_size#批次大小

        self.embedding_size=embedding_size#嵌入大小

        self.vocabulary_size=vocabulary_size#词库大小

        self.pro_table=pro_table

        self.embeddings=tf.Variable(tf.random.uniform([self.vocabulary_size,self.embedding_size],-1.0,1.0))

        self.softmax_weights=tf.Variable(tf.random.truncated_normal([self.embedding_size,self.vocabulary_size],stddev=np.sqrt(2.0/(self.vocabulary_size+self.embedding_size))))
        
        self.softmax_biases=tf.Variable(tf.zeros([self.vocabulary_size]))

        self.optimizer=tf.keras.optimizers.SGD(learning_rate=lr)

        
    def embedding_lookup(self,data,embeddings):#查找embedding

        '''
        data必须为one_hot编码，且一行代表一个分词
        '''
        outputs=[]
        for i in range(data.shape[0]):
            row=tf.math.argmax(data[i][:],output_type=tf.dtypes.int32).numpy()
            outputs.append(tf.reshape(embeddings[row,:],[1,embeddings.shape[-1]]))
            '''left=0
            right=data[i,:].shape[-1]-1
            while(left<=right):
                if i==3:
                    print()
                if(data[i][left].numpy()==1.0):
                    #print(i,"i")
                    outputs.append(tf.reshape(embeddings[left,:],[1,embeddings.shape[-1]]))
                    break
                if(data[i][right].numpy()==1.0):
                    #print(i,"i")
                    outputs.append(tf.reshape(embeddings[right,:],[1,embeddings.shape[-1]]))
                    break
                left+=1
                right-=1
            ''' 
             
        
        return tf.concat(outputs,0)
                
    def __call__(self,data,labels,context_to_center,params,train_predict='predict'):
        '''
        对输入的数据data进行预测或者训练
        context_to_center：是根据上下文查找中心词的查找表
        '''
        self.center_word=tf.math.argmax(data,axis=len(data.shape)-1,output_type=tf.dtypes.int32)#获取当前中心词
        if labels is not None:
            self.context_word=tf.math.argmax(labels,axis=len(labels.shape)-1,output_type=tf.dtypes.int32)#获取当前中心词对应的上下文
        
        embed=self.embedding_lookup(data,self.embeddings)#查找embedding
        
        output=tf.matmul(embed,self.softmax_weights)+self.softmax_biases#对查到的embedding进行线性操作
        output=tf.math.sigmoid(output)

        if train_predict=='train':
            neg_sample=self.neg_sampling(data.shape[0],context_to_center)#负采样

            small_output,small_labels,sample=self.completeVec_to_negSampling(output,data,labels,context_to_center,neg_sample)#把整个词库大小的输出，转化成只包括5个负样本和一个正样本的形式
            
            output=tf.concat(small_output,0)

            labels=tf.concat(small_labels,0)
            
            return output,labels,sample
        elif train_predict=='predict':
            return tf.nn.softmax(output)
        
    def get_params(self):
        return [self.embeddings,self.softmax_weights,self.softmax_biases]
            
    def gradient_descent(self,grads,params):
        self.optimizer.apply_gradients(zip(grads,params))


    def neg_sampling(self,X_num,context_to_center,num=5):
        neg_sample=[]
        for j in range(X_num):
            sample=[]
            for i in range(num):
                ran_num=random.randint(0,len(self.pro_table)-1)
                while((self.pro_table[ran_num]==self.center_word[j]) or (self.pro_table[ran_num] in list(context_to_center[j].keys()))):
                    ran_num=random.randint(0,len(self.pro_table)-1)
                sample.append(self.pro_table[ran_num])
            neg_sample.append(tf.reshape(sample,[1,5]))

        return tf.concat(neg_sample,0)
    
    def loss(self,output,labels):
        return tf.reduce_sum(labels*(tf.math.log(output))+(1-labels)*(tf.math.log(1-output)))

    def completeVec_to_negSampling(self,output,data,labels,context_to_center,neg_sample):
        '''
        把词库大小的输出output转化成只包含一个正样本和几个负样本大小的输出
        '''
        context_word=[]
        for i in range(self.context_word.shape[0]):
            rand_num=random.randint(0,self.context_word.shape[-1]-1)
            context_word.append(self.context_word[i][rand_num])
        
        context_word=tf.reshape(context_word,[-1,1])
        
        neg_sample=tf.concat([neg_sample,context_word],1)
        
        return_outputs,return_labels=[],[]
        for row in range(neg_sample.shape[0]):
            small_output=[]
            small_labels=[]
            for col in range(neg_sample.shape[-1]):
                if(context_word[row][0].numpy()==neg_sample[row][col].numpy()):
                    small_labels.append(1.0)
                else:
                    small_labels.append(0.0)
                small_output.append(output[row][neg_sample[row][col]]) 
            return_outputs.append(tf.reshape(small_output,[1,-1]))
            return_labels.append(tf.reshape(small_labels,[1,-1]))
       
        return return_outputs,return_labels,neg_sample



def main():
    parser = argparse.ArgumentParser(description="输入程序运行必要参数")
    parser.add_argument('-bs','--batch_size',default=32,type=int)
    parser.add_argument('-sw','--skip_window',default=2,type=int)
    parser.add_argument('-es','--embedding_size',default=300,type=int)
    parser.add_argument('-ep','--epochs',default=10,type=int)
    parser.add_argument('-dp','--data_path',default='jaychou_lyrics.txt',type=str)
    parser.add_argument('-lr','--learning_rate',default=1e-3,type=float)
    parser.add_argument('-ts','--table_size',default=10e8,type=float)
    parser.add_argument('-test','--test',default=False,type=bool)
    
    args = parser.parse_args()
    
    batch_size = args.batch_size
    skip_window = args.skip_window
    data_path = args.data_path
    embedding_size = args.embedding_size
    epochs = args.epochs#训练轮次
    table_size = args.table_size
    lr = args.learning_rate
    
    data_path="data//"+data_path

    
    #print(batch_size,skip_window,data_path,embedding_size,epochs,table_size)
    
    def to_oneHot(indices,depth):
        return tf.one_hot(indices, depth)
    
    clipNorm=1.0#梯度裁剪阈值
    corpus_indices,char_to_idx,idx_to_char,vocab_size,voca_frequency=load_data(data_path)#数据预处理

    if(args.test):
        table_size=3*vocab_size
    
    pro_table=init_unigram_table(vocab_size,voca_frequency,idx_to_char,table_size=table_size)#概率表
    skg=skGram(batch_size=int(batch_size),embedding_size=int(embedding_size),vocabulary_size=vocab_size,lr=lr,pro_table=pro_table)#实例化skGram对象  
    params=skg.get_params()#获得模型参数
    
    #epochs=10
    
    for i in range(epochs):
        #print(i)
        data_iter=generate_batch(corpus_indices,batch_size,skip_window)
        for X,Y in data_iter:
            
            context_to_center=[]
            for j in range(Y.shape[0]):
                ctc={}
                for k in range(Y.shape[1]):
                    if Y[j][k].numpy() not in ctc.keys():
                        ctc[Y[j][k].numpy()]=X[j]
                context_to_center.append(ctc)
            
            X=tf.one_hot(X,vocab_size,dtype=tf.float32)
            Y=tf.one_hot(Y,vocab_size,dtype=tf.float32)
            
            with tf.GradientTape() as tape:#梯度带
                tape.watch(params)
                scale_output,scale_labels,_=skg(X,Y,context_to_center=context_to_center,params=params,train_predict='train')#训练
                loss=skg.loss(scale_output,scale_labels)#损失函数
                print("loss %f"%loss)
            grads=tape.gradient(loss,params)#求导
            grads,globalNorm=tf.clip_by_global_norm(grads, clipNorm)#梯度裁剪
            skg.gradient_descent(grads,params)#参数更新


    predict_word="温柔"
    predict_word=fool.cut(predict_word)#分词
    predict_wrd_idx=[]
    #把文本转化成词库的索引
    for i in range(len(predict_word)):
        predict_wrd_idx.append(char_to_idx[predict_word[i][0]])
    
    predict_wrd_oneHot=tf.one_hot(predict_wrd_idx,vocab_size,dtype=tf.float32)
    
    output=skg(predict_wrd_oneHot,labels=None,context_to_center=None,params=params,train_predict='predict')
    
    print(idx_to_char[tf.math.argmax(output,axis=1,output_type=tf.dtypes.int32)[0].numpy()])




if __name__ == "__main__":
    main()
