import sys
import os


#Go one directory up and append to path to access the deepdna package
#The following statement is equiv to sys.path.append("../")
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__),os.path.pardir)))





def main(argv):
   
    assert coords_overlap(('chr1',40,50),('chr1',50,60)) == True
    assert coords_overlap(('chr1',40,50),('chr1',49,60)) == True
    assert coords_overlap(('chr1',40,50),('chr1',30,60)) == True
    assert coords_overlap(('chr1',40,50),('chr1',30,39)) == False

    bed_file = argv[1]

    if argv[2] and argv[3]:
        start_window = [int(argv[2]),int(argv[3])]
    else:
        start_window = None
    
    remove_overlaps(bed_file,start_window)
    
    
def remove_overlaps(bed_file,start_window=None):
    """
    Output a bedfile without overlaps
    If keep_first is True, only keep the first example visted.
    Else select randomly which overlapping segment is to be retained.

    If start window is specified, instead of using bed file start and end columns,
    this function will use start+start_window[0] and start+start_window[1]
    """
    if os.path.splitext(bed_file)[1] != '.bed':
        print "File must have extension \'.bed\'!"
        return 

    print "Start window is specified as {}".format(start_window)
     
    out_file = "{}_no_overlaps.bed".format(os.path.splitext(bed_file)[0])

    orig_count =0
    new_count=0
    prev_coord = ('asdfasdf',0,0)
    with open(bed_file,'r') as bf, open(out_file,'w') as of:
        for l in bf:
            if l.startswith('#'):
                of.write(l)
            else:
                coord = l.strip('\n').split('\t')[:3]
                if start_window and len(start_window)==2:
                    new_start = str(int(coord[1]) + start_window[0])
                    new_end = str(int(coord[1]) + start_window[1])
                    coord = [coord[0],new_start,new_end]
                
                orig_count += 1
                if not coords_overlap(prev_coord,coord):
                    '''
                    Note that the coordinates preceding any series of overlapping
                    elements will always be written to the file.
                    '''
                    of.write(l)
                    new_count += 1
                    #Only reassign prev_coord if overlap not detected
                    prev_coord = coord

        print "Wrote file {}".format(out_file)
        print "Original file had {} entries".format(orig_count)
        print "No overlapping coordinate file has {} entries".format(new_count)
        print "{} entries were removed".format(orig_count-new_count)
        
        

def coords_overlap(c1,c2):
    """Check for coordinates between c1 and c2 where
       both are coordinates in the form ('contig',start (int),end (int))
       Boundaries are exclusive
       
    :param c1: tuple or list with coordinates
    :param c2: tuple or list with coordinates
    :returns: True if overlap, False if no overlap
    :rtype: boolean
    """
    if c1[0] != c2[0]:
        return False

    if c1[2] >= c2[1] and c2[2]>=c1[1]:
        return True
    
    return False
    
    
    
if __name__ == "__main__":
    main(sys.argv)
