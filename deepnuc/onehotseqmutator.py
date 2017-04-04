import numpy as np

def main():
    oh_nuc = np.asarray([[1.,0,0,0,0],[0,1,0,0,0],[0,0,1,1,0],[0,0,0,0,1]])
    print oh_nuc.shape

    print "oh_nuc"
    print oh_nuc
    
    #for i in range(oh_nuc.shape[1]):
    #    print oh_nuc[:,i]

    ohi = OnehotSeqMutator(oh_nuc)
    print oh_nuc.shape

    
    print ohi.next()
    print ohi.next()
    print ohi.next()
    print ohi.next()
    print ohi.next()
    print ohi.next()
    print ohi.next()
    print ohi.next()
    print ohi.next()
    


class OnehotSeqMutator:
    """An iterator for making pointwise mutations for a 4xn onehot numpy array
       representing nucleic acid sequence. Useful for passing mutation variants
       to deep neural networks for making mutation maps as described in Alipanhi 2015
    
    """

    def __init__(self,oh_nuc):
        """

        :param oh_nuc: an nx4 onehot numpy array

        """
        self.oh_nuc = oh_nuc
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

        
if __name__ == "__main__":
    main()
