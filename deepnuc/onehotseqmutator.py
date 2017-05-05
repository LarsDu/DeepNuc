import numpy as np
import dubiotools as dbt




class OnehotSeqMutator:
    """An iterator for making pointwise mutations for a 4xn onehot numpy array
       representing nucleic acid sequence. Useful for passing mutation variants
       to deep neural networks for making mutation maps as described in Alipanahi 2015
    
    """

    def __init__(self,oh_nuc):
        """

        :param oh_nuc: an 4xn onehot numpy array

        """
        self.seq_len = oh_nuc.shape[1]
        self.oh_nuc = np.copy(oh_nuc)
        self.i = 0
        self.nuci=0
        self.tri_count=0
        self.n = oh_nuc.shape[1]*3 #Number of iterations (3 mutations per column)
        self.onei = None

        #copy of numpy array
        
        
    def __iter__(self):
        return self

    def next(self):
        if self.i<self.n:
            self.increment_column()
            self.i += 1
            
            if self.tri_count==3:                
                #self.oh_nuc = np.copy(self.oh_nuc_orig)#Note: replace copy with prev increment
                self.increment_prev_column() #reset to original position
                self.tri_count=0
                self.nuci += 1
                self.increment_column() #increment the next column
           
            self.tri_count +=1
            
            return self.oh_nuc
        else:
            raise StopIteration()


    def increment_column(self):
        self.onei = np.nonzero(self.oh_nuc[:,self.nuci])[0][0]
        self.oh_nuc[(self.onei+1)%4,self.nuci]=self.oh_nuc[self.onei,self.nuci]
        self.oh_nuc[self.onei,self.nuci]=0

    def increment_prev_column(self):
        if self.nuci>0:
            onei = np.nonzero(self.oh_nuc[:,self.nuci])[0][0]
            self.oh_nuc[(onei+1)%4,self.nuci-1]=self.oh_nuc[onei,self.nuci-1]
            self.oh_nuc[onei,self.nuci-1]=0


    def pull_batch(self,batch_size):
        '''
        Pull a batch with dims [batch_size,seq_len,4]
        Note reversal of last two dimensions
        '''
        
        full_batch = np.zeros((batch_size,self.seq_len,4))
        for i in range(batch_size):
            full_batch[i,:,:] = self.next().T
        return full_batch
    
    def pack():
        pass
            
def main():
    #Test code for this class
    oh_nuc = np.asarray([[1.,0,0,0,0],[0,1,0,0,0],[0,0,1,1,0],[0,0,0,0,1]])
    print oh_nuc.shape

    print "oh_nuc"
    print dbt.onehot_to_nuc(oh_nuc)
   
    ohi = OnehotSeqMutator(oh_nuc)
    print oh_nuc.shape

    '''
    print "SHAPE",ohi.next().shape
    print ohi.next()
    print ohi.next()
    print ohi.next()
    '''
    
    nucl = [dbt.onehot_to_nuc(next_seq) for next_seq in ohi]
    print dbt.onehot_to_nuc(oh_nuc)
    print nucl
    
    #print ohi.pull_batch(1)
    #print ohi.pull_batch(1)

    
        
if __name__ == "__main__":
    main()
