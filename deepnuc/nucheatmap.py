import matplotlib.pyplot as plt
import numpy as np

def main():
    #nucs = 'ATGCA'
    #print nucs
    #lmat = np.asarray([[1.,4,5,2,4],[3,4,5,6,6],[6,5,3,1,1],[8,7,3,4,2]])
    #lmat = lmat/np.sum(lmat,axis=0)

    nucs = 'ATCTAGCGTCATGCATGCAG'
    lmat = np.random.rand(4,20)
    nuc_heatmap(nucs,lmat)

    
    
def nuc_heatmap(seq,mat,dims=[.25,2.25]):
    """
    :param seq: n length nucleotide sequence
    :param mat: 4xn numpy array containing weights
    :size: figure height, width in inches
    :returns: numpy array with height of each column (nuc position)
    """
    seq_len = len(seq)
    if seq_len != mat.shape[1]:
        print "Seq len",seq_len,"does not match mat dim[1]",mat.shape[1]
        return None
    if mat.shape[0] != 4:
        print "mat dim[0] must be 4"
        return None

    
    ylets = ['T','C','A','G'] #y axis letter labels
    nucs = list(seq) 
    
    fig,ax = plt.subplots(2,1,sharex=True)
    fig.set_size_inches((dims[0]*seq_len),dims[1])
    #Plot histogram
    hmat = np.sum(mat,axis=0)
    ax[0].bar(range(hmat.shape[0]),hmat,width=1,color='blue')
    #ax[0].tick_params(axis=u'both',which=u'both',length=0)
    
    #Plot heatmap
    ax[1].imshow(mat,cmap='Blues')
    ax[1].tick_params(axis=u'both',which=u'both',length=0)
    ax[1].set_xticks(np.arange(0,seq_len))
    ax[1].set_xticklabels(nucs)
    ax[1].set_yticks([0.,1,2,3])
    ax[1].set_yticklabels(ylets)
    color_ax_nucs(ax[1])
    plt.show()
    return hmat

def color_ax_nucs(ax):
    color_dict= {'A':"red",'G':"blue",'T':"lime",'C': "orange"}
    xticks = ax.get_xticklabels()
    for l in ax.get_xticklabels()+ax.get_yticklabels():
        cur_l = l.get_text()
        if cur_l in color_dict:
            l.set_color(color_dict[cur_l])
            l.set_weight("bold")
            l.set_fontsize(10)
    
    
if __name__=="__main__":
    main()

