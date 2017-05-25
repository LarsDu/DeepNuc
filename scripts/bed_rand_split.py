import sys
import os

import numpy as np

#Go one directory up and append to path to access the deepdna package
#The following statement is equiv to sys.path.append("../")
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__),os.path.pardir)))

def main(argv):
    bed_file = argv[1]
    #split_frac = 0.3
    split_frac = float(argv[2])


    rand_split(bed_file,split_frac)
    

def rand_split(bed_file,split_frac):
    """
    Randomly split a bed file into two bed files
    """
    num_lines = sum(1 for line in open(bed_file))
    num_f1_lines = int(num_lines*split_frac)
    num_f2_lines = num_lines-num_f1_lines
    perm_indices = np.random.permutation(range(num_lines))

    f1_lines = perm_indices[0:num_f1_lines].tolist()
    f1_lines.sort()

    
    #f2_lines = np.setdiff1d(perm_indices,f1_lines).tolist().sort()

    
    out_file1 = "{}_randsplit_{}_entries_{}_frac.bed".format(os.path.splitext(bed_file)[0],num_f1_lines,split_frac)
    out_file2 = "{}_randsplit_{}_entries_{}_frac.bed".format(os.path.splitext(bed_file)[0],num_f2_lines,1-split_frac)

    
    with open(bed_file,'r') as bf, open(out_file1,'w') as of1, open(out_file2,'w') as of2:
        pp=0
        for i,l in enumerate(bf):
            if i== 0 and l.startswith('#'):
                of1.write(l)
                of2.write(l)
            if i == f1_lines[pp] and i>0:
                of1.write(l)
                if pp<num_f1_lines-1:
                    pp += 1
            else:
                of2.write(l)

        print "Wrote files {} and {}".format(out_file1,out_file2)

if __name__ == "__main__":
    main(sys.argv)
    
    
    
