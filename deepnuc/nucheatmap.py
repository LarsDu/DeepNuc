import matplotlib.pyplot as plt
import numpy as np

import os
import sys
sys.path.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import duseqlogo.LogoTools as LogoTools

def main():
    #nucs = 'ATGCA'
    #print nucs
    #lmat = np.asarray([[1.,4,5,2,4],[3,4,5,6,6],[6,5,3,1,1],[8,7,3,4,2]])
    #lmat = lmat/np.sum(lmat,axis=0)

    nucs = 'ATCTAGCGTCATGCATGCAG'
    lmat = np.random.rand(4,20)
    #nuc_heatmap(nucs,lmat)
    nuc_height_plot_matplotlib(nucs,np.sum(lmat,axis=0))
        
    
def nuc_heatmap(seq,mat,save_fig='',show_plot=False,dims=[.25,2.25],):
    """
    :param seq: n length nucleotide sequence
    :param mat: 4xn numpy array containing weights
    :dims: figure height, width in inches
    :returns: numpy array with height of each column (nuc position)
    """
    seq_len = len(seq)
    if seq_len != mat.shape[1]:
        print "nuc_heatmap: Seq len",seq_len,"does not match mat.shape[1]",mat.shape
        return None
    if mat.shape[0] != 4:
        print "nuc_heatmap: mat dim[0] must be 4",mat.shape
        return None
    
    ylets = ['T','C','A','G'] #y axis letter labels
    nucs = list(seq)


    fig,ax = plt.subplots(2,1,sharex=True)


    fig.set_size_inches((dims[0]*seq_len),dims[1])
    #Plot histogram
    hmat = np.sum(mat,axis=0)
    #ax[0].autoscale(False)
    ax[0].bar(range(hmat.shape[0]),hmat,width=1,color='blue')
    #ax[0].tick_params(axis=u'both',which=u'both',length=0)
    
    #Plot heatmap
    #ax[1].autoscale(False)
    ax[1].imshow(mat,cmap='Blues')
    ax[1].tick_params(axis=u'both',which=u'both',length=0)
    ax[1].set_xticks(np.arange(0,seq_len))
    ax[1].set_xticklabels(nucs)
    ax[1].set_yticks([0.,1,2,3])
    ax[1].set_yticklabels(ylets)
    color_ax_nucs(ax[1])
    if save_fig != '':
        fig.savefig(save_fig)
    if show_plot:
        fig.show()

    return hmat


def nuc_height_plot_matplotlib(seq,heights):

    '''
    There is no font distortion in matplotlib, so this method doesn't quite work
    References:
    https://matplotlib.org/users/text_props.html
    https://matplotlib.org/api/transformations.html
    '''
    ylets = ['T','C','A','G'] #y axis letter labels
    nucs = list(seq)
    seq_len = len(seq)
    #for nuc in nucs:
    fig,ax = plt.subplots(1)

    fig.set_size_inches(.2*(seq_len),.5)
    ax.set_ylim([0,1])
    ax.set_xlim([0,seq_len])

    color_dict= {'A':"red",'G':"blue",'T':"lime",'C': "orange"}
    for i,nuc in enumerate(nucs):
        print nuc
        ax.text(float(i)*1./seq_len, 0, nuc,
                ha='left',
                va='bottom',
                weight='bold',
                size=int(np.floor(heights[i])*20),
                color=color_dict[nuc],
                transform=ax.transAxes,
                )


            
        
    plt.show()
    
    #color_ax_nucs(ax)

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

