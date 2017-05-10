import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

import os
import sys
sys.path.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import duseqlogo.LogoTools as LogoTools

def main():
    nucs = 'ATGCA'
    print nucs
    lmat = np.asarray([[1.,4,5,2,4],[3,4,5,6,6],[6,5,3,1,1],[8,7,3,4,2]])
    lmat2 = np.asarray([[1.,4,5,2,4],[3,4,5,-1,6],[6,-1.2,3,1,1],[8,7,3,4,2]])
    lmat2 = lmat2/np.sum(lmat2,axis=0)

    print lmat2
    #nucs = 'ATCTAGCGTCATGCATGCAG'
    
    nuc_heatmap(nucs,lmat2,'',True)

        
    
def nuc_heatmap(seq,
                mat,
                save_fig='',
                clims=[-1,1],
                cmap='bwr_r',
                dims=[.25,2.25],):
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
    height_mat = np.sum(mat,axis=0)
    #ax[0].autoscale(False)
    ax[0].bar(range(height_mat.shape[0]),height_mat,width=1,color='green')
    #ax[0].tick_params(axis=u'both',which=u'both',length=0)
    
    #Plot heatmap
    #ax[1].autoscale(False)
    heatmap = ax[1].pcolor(mat,cmap=cmap,vmin=clims[0],vmax=clims[1])
    ax[1].tick_params(axis=u'both',which=u'both',length=0)
    ax[1].set_xticks(np.arange(0.5,seq_len+0.5))
    ax[1].set_xticklabels(nucs)
    ax[1].set_yticks([0.,1,2,3])
    ax[1].set_yticklabels(ylets)
    color_ax_nucs(ax[1])
    cbar = fig.colorbar(heatmap,ax=ax.ravel().tolist())
    cbar.set_clim(clims[0],clims[1])

    if save_fig != '':
        print "Saving nuc_heatmap to",save_fig
        fig.savefig(save_fig)
    
    return fig,ax


def nuc_height_plot_matplotlib(seq,heights):

    '''
    NOTE: THIS IS DEPRECATED
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
        ax.text(float(i)*1./seq_len, 0, ' '+nuc,
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

