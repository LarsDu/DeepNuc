import h5py
import dubiotools as dbt
import nucnibble
import numpy as np
import nucnibble

class ConvertToHdf5:

    def __init__(self,reader_list,output_fname):


        self.reader_list = reader_list
        print "Reader list", self.reader_list
        self.output_fname = output_fname
        self.output_files =  []
        self.seq_len = reader_list[0].seq_len

        if len(reader_list)>255:
            print "Error, must have fewer than 255 readers"
            print "Labels are encoded as single uint8 values"
        
        self.num_classes = len(self.reader_list)
        
        
        '''Array of possible labels. For 3 classes, creates 3x3 array.
        Each row is one possible label'''
        #self.labels  = dbt.numeric_labels_to_onehot(range(self.num_classes))
        
        '''
        #Get names of each file, and create corresponding output_file name
        #These names are only used if a output_fname is not specified 
        for reader in self.reader_list:
            basename = os.path.splitext(reader.name)[0]
            single_output_fname = basename+'.h5'
            self.output_files.append(single_output_fname)
        '''

        self.convert_to_single_hdf5(self.reader_list,self.output_fname)

        
    def close(self):
        for reader in self.reader_list:
            print reader.name,"closed."
            reader.close()
      

    def convert_to_single_hdf5(self,
                               reader_list,
                               output_file,                              
                               compression=None):
        """
        Convert a list of reader objects into a single hdf5 file. This is the file
        that will be used for training. Random access from a single training file
        should greatly increase training speed.


        The first column (uint8) is reserved for encoding the numeric label
        Each 4-bit nibble encodes a nucleotide, and each column encodes a uint8
        consisting of two nibbles. If there is an odd number of nucleotides,
        the last 4 bits of the last column (uint8) is ignored
        
        """
        num_total_rows = int(np.sum([reader.get_num_records() for reader in reader_list]))
        num_cols = int(1+ self.seq_len//2 + self.seq_len%2 +4 )



        with h5py.File(output_file,'w') as of:
            of.attrs['seq_len'] = self.seq_len
            of.attrs['num_classes'] = self.num_classes
                
            dset = of.create_dataset('dna',dtype='int8',
                                     shape=(num_total_rows,num_cols),
                                     chunks=(64,num_cols),
                                     compression=compression,
                                     maxshape = (1000000,num_cols))


            #genidx_file = os.path.splitext(output_file)[0]+'.genidx'
            #genidx_handle = open(genidx_file,'w')
            
            offset = 0
            for lbli, reader in enumerate(reader_list):
                print "On reader",lbli
                reader.open()
                for row in range (reader.num_records):
                    if row%2000==0:
                        print "Wrote",row,"entries to",self.output_fname
                    nuc_seqs,rec_ids = reader.pull(1) #Outputs are lists
                    np_nibble_seq =  nucnibble.nuc_to_uint8_numpy(nuc_seqs[0]) 
                    dset[offset+row,0] = lbli #First element in row is the numeric label
                    dset[offset+row,1:] = np_nibble_seq #Following elements are nibble coded nuc seq
                     
                    #genidx_handle.write(rec_ids[0]+'\n')
                #Add offset of current reader
                offset = offset+reader.num_records
                reader.close()
            print "Wrote", num_total_rows,"to",self.output_fname
            #print "Saving annotations to", genidx_file
            #genidx_handle.close()

    
            
    '''
    def encode_nd_to_1d(self, dna_onehot,label):
        #Flattens a batch to 1-D for saving into an HDF5 file
        dna_onehot_1d = np.ravel(np.squeeze(dna_onehot))
        return np.concatenate((dna_onehot_1d,label),axis=0)
    '''
    '''
    def decode_1d_to_nd(input_1d, shape_list):
        #Decode a 1D matrix into the shapes in shape_list
        return_list = []
        start_ind = 0
        #Slice out return data in their proper shapes
        for shape in shape_list:
            elem_len= np.prod(shape)
            return_list.append(np.reshape(input_1d[start_ind:elem_len+start_ind],shape))
            start_ind += elem_len
        return return_list
    '''
    '''
    def determine_shapes(self,*argv):
        #Determine the shapes of numpy arrays
        shape_list = []
        for arg in argv:
            shape_list.append(arg.shape)
        return shape_list
    '''
