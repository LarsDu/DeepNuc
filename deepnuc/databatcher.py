import numpy as np

class DataBatcher:
    """
    Pull batches of sequences from BaseNucData derived objects (ie: NucHdf,NucMemory)
    Also:
    	- Keep track of epochs,
    	- Shuffles indices at the end of every epoch
    You can instantiate two data_batchers with different indices but the same
    nuc_data to create training and test splits
    """
    
    def __init__(self,nuc_data,indices=None,use_onehot_labels=True,seed=None):
        self.nuc_data = nuc_data

        self.num_classes = nuc_data.num_classes
        self.seq_len = nuc_data.seq_len
        self.use_onehot_labels = use_onehot_labels

        if indices == None:
            self.indices = range(self.nuc_data.num_records)
        else:
            self.indices = indices

        self.num_records = len(self.indices)

        
        self.seed=seed
        if self.seed==None:
            self.perm_indices = np.random.permutation(self.indices)
        else:
            self.perm_indices = np.random.RandomState(self.seed).\
                                            permutation(self.indices)

        self.epoch_tracker = EpochTracker(self.num_records)



    
    def get_label_min_max(self):
        """Used to calculate mean and standard deviation
         Necessary for y feature normalization for regression"""

        all_labels = np.zeros((self.num_records))
        for i in range(self.num_records):
            label,_ = self.nuc_data.pull_index_onehot(i)
            all_labels[i] = label

        return np.min(all_labels),np.max(all_labels)
    
    def get_label_mean_std(self):
        """Used to calculate mean and standard deviation
         Necessary for y feature normalization for regression"""

        all_labels = np.zeros((self.num_records))
        for i,ind in enumerate(self.indices):
            label,_ = self.nuc_data.pull_index_onehot(ind)
            all_labels[i] = label

        mean = np.mean(all_labels)
        std = np.std(all_labels)

        return mean,std    
        
    def set_perm_indices(self,new_indices):
        """
        Used for setting different perm_indices
        :param new_indices:  A list of indices in nuc_data
        """
        self.perm_indices = new_indices
        self.num_records = len(self.perm_indices)
        

         
    def rand_perm_indices_from_seed(self,seed):
        """
        Perform a random seeded shuffle of self.perm_indices
        :param seed:  An integer to seed a random number generator
        """
        self.perm_indices = np.random.RandomState(seed).\
                permutation(range(self.num_records))
        #self.num_records = len(self.perm_indices)
        #self.epoch_tracker = EpochTracker(self.num_records)
 

    def _pull_batch_no_epoch(self,batch_size):
        """Pull a batch from self.nuc_data without incrementing self.epoch_tracker
        :param batch_size: 
        :returns: dna_seq_batch, labels_batch
        :rtype: numpy array (uint8), numpy array (int)
        """
        
        #Preallocate
        dna_seq_batch = np.zeros((batch_size,self.seq_len,4))
        labels_batch = np.zeros((batch_size,self.num_classes))
        
        batch_start = self.epoch_tracker.cur_index
        batch_end = batch_start + batch_size
        batch_indices = self.perm_indices[batch_start:batch_end]
        for bi, pindex in enumerate(batch_indices):
            numeric_label, dna_seq_data = self.nuc_data.pull_index_onehot(pindex)
            dna_seq_batch[bi,:,:] = dna_seq_data #Must be onehot
            if self.use_onehot_labels == True:
                labels_data = self.numeric_to_onehot_label(numeric_label)#ie: returns [0 1]
            else:
                labels_data = numeric_label #ie: 1.2345 for regression
            labels_batch[bi,:] = labels_data


        return labels_batch,dna_seq_batch


    def numeric_to_onehot_label(self,numeric_label):
        onehot = np.zeros(self.num_classes,dtype = np.float32)
        onehot[int(numeric_label)] = 1
        return onehot
    
    def pull_batch(self,batch_size):
        """Pull a batch, incrementing self.epoch_tracker
        :param batch_size: 
        :returns: dna_seq_batch, labels_batch
        :rtype: numpy array (uint8), numpy array (int)
        """        
        labels_batch,dna_seq_batch= self._pull_batch_no_epoch(batch_size)
        
        epoch_completed = self.epoch_tracker.increment(batch_size)
        if epoch_completed:
            #print "Starting on epoch",self.train_epoch_tracker.num_epochs
            self.perm_indices = np.random.permutation(self.perm_indices)
        return labels_batch,dna_seq_batch


    def pull_batch_by_index(self,index,batch_size):
        """ Pull a batch from self.indices by index
        
           :returns:
                    labels_batch: numpy array with shape (batch_size, num_classes)
                    dna_seq_batch: numpy array with shape (batch_size,seq_len,4)
        """
        dna_seq_batch = np.zeros((batch_size,self.seq_len,4))
        labels_batch = np.zeros((batch_size,self.num_classes))
        batch_start = index
        batch_end = batch_start + batch_size
        batch_indices = self.indices[batch_start:batch_end]
        for bi, index in enumerate(batch_indices):
            numeric_label, dna_seq_data = self.nuc_data.pull_index_onehot(index)
            dna_seq_batch[bi,:,:] = dna_seq_data #Must be onehot
            if self.use_onehot_labels == True:
                labels_data = self.numeric_to_onehot_label(numeric_label)#ie: [0 1]
            else:
                labels_data = numeric_label #ie: 1.2345 for regression
            labels_batch[bi,:] = labels_data #ie: [0 1]
        return labels_batch,dna_seq_batch

    
    
class EpochTracker:
    def __init__(self,num_examples):
        """
        Initialize EpochTracker with the number of examples it tracks
        :param num_examples: An integer 
        """
        
        self.num_examples = num_examples
        self.num_epochs = 0 #The number of epochs that have been passed
        self.cur_index = 0 #The index position on current epoch

    def increment(self,increment_size):
        #Returns true if end of current epoch
        new_index = self.cur_index + increment_size
        #Reset epoch counter if end of current epoch has been reached.
        if ( new_index >= self.num_examples):
            self.num_epochs += 1
            self.cur_index = 0
            #Reshuffle indices
            return True
        else:
            self.cur_index = new_index
            return False


       
