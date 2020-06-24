import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import  optim



class Model(nn.Module):
    def __init__(self,WordVocabSize , WordEmbedDim, LstmHiddenDim ,SeqLen ,g , max_charlen):
        super(Model,self).__init__()     #### set hidden units acc to paper
        self.LstmHiddenDim = LstmHiddenDim
        self.LstmHiddenDim_ = int(WordEmbedDim/2)
        self.g = g
        self.SeqLen = SeqLen
        self.WordEmbedDim = WordEmbedDim
        self.max_charlen = max_charlen
        #Word
        self.wordembed = nn.Embedding(WordVocabSize , self.WordEmbedDim)

        #Character
        self.charbilstm = nn.LSTM( 1 , self.LstmHiddenDim_ , num_layers = 1 , bidirectional = True , batch_first = True )
        self.charlinearfwd = nn.Linear( self.max_charlen * self.LstmHiddenDim_ , self.WordEmbedDim )    ## Output is WordEmbedDim because later we need to add both the words and char, and output of word is WordEmbedDim so.
        self.charlinearrev = nn.Linear( self.max_charlen * self.LstmHiddenDim_ , self.WordEmbedDim )

        #Output
        self.lstm = nn.LSTM(self.WordEmbedDim , LstmHiddenDim , num_layers = 2 , bidirectional = False ,batch_first=True)
        self.dense = nn.Linear( LstmHiddenDim , WordVocabSize )

    def  forward(self , Wordinputs , Charinputs):

        #Word
        wordembed = self.wordembed(Wordinputs)
        #print(self.LstmHiddenDim_)

        #Char

        BiOutFwd = []
        BiOutRev = []

        for char_ in Charinputs:  
            out_ ,(hidden_,cellstate_) = self.charbilstm( char_.float().view([ char_.shape[0], char_.shape[1],-1 ]) ) 
            fwd =  out_[:,:,:self.LstmHiddenDim_]
            rev = out_[:,:,self.LstmHiddenDim_:]
            #print(fwd , fwd.shape , 'convert to ' , self.SeqLen ,(self.max_charlen * self.LstmHiddenDim_))
            #print(self.max_charlen , self.LstmHiddenDim_)
            BiOutFwd.append(fwd.reshape([self.SeqLen ,(self.max_charlen * self.LstmHiddenDim_) ]) )
            BiOutRev.append( rev.reshape([self.SeqLen ,(self.max_charlen * self.LstmHiddenDim_) ]) )
            #print('Char out',out_.shape )

        BiOutFwd = torch.stack(BiOutFwd , dim = 0)
        BiOutRev = torch.stack(BiOutRev , dim = 0)
        #print( 'BiOutFwd , BiOutRev' , BiOutFwd.shape ,BiOutRev.shape )  

        charlinearfwd = self.charlinearfwd(BiOutFwd)
        charlinearrev = self.charlinearrev(BiOutRev)
        charembed = charlinearfwd  +  charlinearrev

        if wordembed.shape != charembed.shape:
            return "Error , check the shapes in code"

        #Output
        wordchar = ((1-self.g) * wordembed) + (self.g * charembed)

        out ,(hidden,cellstate) = self.lstm( wordchar )     ## bath, timestamps ,hidden
        out  = out[:,-1,:]  ## To take the hiddenvalues from only the last cell ( as it is next value pred)
        dense = F.softmax(self.dense( out ))

        return dense
  

if __name__ == "__main__": 
    import numpy as np
    model1 = Model(WordVocabSize = 1500 , WordEmbedDim=300 , LstmHiddenDim=150 , SeqLen=14,  g=0.25,max_charlen = 16 )
    out = model1(Wordinputs = torch.tensor(np.random.randint(0, 80 , (32,14)) , dtype =  torch.long) ,Charinputs = torch.tensor(np.random.randint(0, 16 , (32,14,16)) , dtype =  torch.long))
    print("Output Shape - ",out.shape)  ### 32 batch size , each with a sigmoid output of 1500 word vocab size