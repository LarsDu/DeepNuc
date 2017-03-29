Readme. Setting up training requires the following steps:

1. Convert nucleotide data into an 'HDF5' format. Since training data is randomly drawn, this will
greatly speed up the training process.
2. Write a json file specifying training parameters
3. (Optional) Run k-folds with different
4. Run training

5. Feed examples back into the learned model and use DeepTaylor decomposition











Step 1: Convert files:

For fasta files:
python FastaToHdf5.py --file_list=file1.fa,file2.fa,file3.fa --seq_len=600 --output_fname=myfile.h5 


For genomic coordinates:
python CoordsToHdf5.py --file_list=coord1.bed,coord2.bed --seq_len=600 
                       --output_fname=myfile.h5 --genome_file=h619.fa
                       --chrom_sizes_file=hg19.chrom.sizes


Site for obtaining fetchChromosomeSizes program (for Linux):
http://hgdownload.cse.ucsc.edu/admin/exe/linux.x86_64/

Note: If output_fname is not specified, it will be set to 'default.h5' by default. Also every 
entry in coord files need to have a seq_len that matches with user specified seq_len.





Step 2: Write json file with training parameters.

An example json file is provided in this directory as example_params.json. You may want validate 
your json file on http://www.jsonlint.com.



Step 3: (Optional) Run training with k-folds cross validation to optimize model parameters.
Inference models can be tweaked in the file NucInfModels.py

Example:
python run.py --json_params=example_params.json --mode=validate


Step 4: Run training:  

Example:
python run.py --json_params=example_params.json --mode=train


Step 5:
 
