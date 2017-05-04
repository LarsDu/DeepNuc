import cairocffi as cairo
import numpy as np
from subprocess import call


import os
import math

import sys
sys.path.append('../') 
import deepnuc.dubiotools as dbt

#import pygtk
#pygtk.require('2.0')
#import gtk
#import gtk.gdk as gdk






class LogoSheet:
    '''
    A sheet object for drawing multiple SeqLogos.
    Draws a set of Seqlogos onto a single sheet.
    Base class
    
    '''
    
    def __init__(self,logo_mats,input_type ='pfm',label_list=None):

        """
        Args:
    	logo_mats: list of matrices holding logo data
    	nuc_onehots: list of one-hot matrices with nucleotide sequence (4xn onehot format)
        input_type:
        	'ppm' for a 4xn position probability matrix
            'pfm' for a 4xn position frequency matrix
            'heights' for 4xn specified heights

        """
        
        self.base_init(logo_mats,input_type,label_list)

        #Pixel dimensions of sheet
        self.width = self.logo_list[0].width + 10
        self.height = self.logo_list[0].height*self.num_logos
        self.surface = cairo.ImageSurface(cairo.FORMAT_ARGB32,self.width,self.height)
        self.context = cairo.Context(self.surface)
        
        
        #Draw 
        if self.input_type == 'heights':
            self.draw_heights()
        elif self.input_type == 'pfm' or self.input_type == 'ppm':
            self.draw_pwms()
        


    def base_init(self,logo_mats,input_type,label_list):
        '''
        I split this method off from __init__ to share
        portions of initialization with subclasses - Larry
        '''
        
        self.logo_mats = logo_mats
        
        #Labels: Place a user specified string label attached to each sequence
        self.label_list = label_list
        
        self.input_type = input_type
        if self.label_list != None:
            self.do_draw_label = True #Draws a numbered label #MODIFY FROM HERE
        else:
            self.do_draw_label = False
        self.num_logos = len(self.logo_mats)

       
        #Init logos depending on mode
        if self.input_type == 'pfm':
            self.logo_list = [SeqLogo(pfm) for pfm in self.logo_mats]
        elif self.input_type == 'ppm':
            #Assuming sample size of 10,000
            self.logo_list = [SeqLogo(pfm*10000) for pfm in self.logo_mats]
        elif self.input_type == 'heights':
            self.logo_list = [HeightLogo(height_mat) for height_mat in self.logo_mats]
        #elif self.input_type == 'nucs':
        #    #Note: this will only draw a single logo
        #    nuc_pfm = PwmTools.pfm_from_nucs(self.logo_data)
        #    self.draw_pwm(nuc_pfm)
        else:
            print "Error, inappropriate option passed."
            print "Type either \'pfm\', \'ppm\', or \'heights\'."

                    
        #TODO: some logos/pfms might have different widths
        #make the program check for largest width
        #Initialize dynamic variables.             
  
  
            
    def draw_pwms(self):
        for i,logo in enumerate(self.logo_list):
            # print "Drawing index ", i, " at ", logo_list[i].height*i
            #self.draw_index_label(self.width-50,logo_list[i].height*i+20,i)
            logo.draw_pwm()
            #set x,y position of logo
            self.context.set_source_surface(logo.surface,0,logo.height*i)
            if self.do_draw_label and self.label_list != None:
                logo.draw_label(self.label_list[i]+' '+str(i))
            self.context.paint()

    def draw_heights(self):
         for i,logo in enumerate(self.logo_list):
            logo.draw()
            self.context.set_source_surface(logo.surface,0,logo.height*i)
            if self.do_draw_label and self.label_list!=None:
                logo.draw_label(self.label_list[i]+' '+str(i))
            self.context.paint()
            
    def write_to_png(self,fname):
        self.surface.write_to_png(fname)
        
    def write_to_svg(self,fname):
        #svg_surf = cairo.SVGSurface(fname,self.width,self.height)
        svg_surf = cairo.SVGSurface(fname,self.width,self.height)
        svg_context = cairo.Context(svg_surf)
        svg_context.set_source_surface(self.surface,0,0)
        svg_context.paint()
        #svg_surf.finish()
        #svg_context.show_page()

    def write_to_pdf(self,fname):
        pdf_surf = cairo.PDFSurface(fname,self.width,self.height)
        pdf_context = cairo.Context(pdf_surf)
        pdf_context.set_source_surface(self.surface,0,0)
        pdf_context.paint()
        #pdf_surf.finish()

    #def show(self,fname):
    #    pass

    

class SimpleHeightLogoSheet:
    def __init__(self,heights,nuc_seqs):

        if len(logo_mats) != len(nuc_seqs):
            print "Error! Number of logo_mats must match number of nuc_seq!"
            return None

        self.heights = heights
        self.nuc_seqs = nuc_seqs
        ##Specific to SimpleHeightLogoSheet
        self.nuc_height = 20

        #Number of pixels between logo and nuc sequence
        self.nuc_spacer = 10
        
        #Number of pixels between drawn logos
        self.bottom_spacer = 30
            
        #Pixel dimensions of sheet
        self.width = self.logo_list[0].width + 10
        self.height = ((self.logo_list[0].height+self.bottom_spacer)*self.num_logos)
        
        self.surface = cairo.ImageSurface(cairo.FORMAT_ARGB32,self.width,self.height)
        self.context = cairo.Context(self.surface)

        self.simple_heights_list = [SimpleHeightLogo(height,seq) for \
                                     height,seq in zip(self.heights,self.nuc_seqs) ]
        self.draw_simple_heights()


    def draw_simple_heights(self):
        #SimpleHeightLogoSheet
        for i,simp in enumerate(self.simple_heights_list):
            #Draw item
            simp.draw()
            #set x,y position of logo (y increases from top to down
            total_height = (simp.height+
                           self.nuc_spacer+
                           self.nuc_height+
                           self.bottom_spacer)
                
            self.context.set_source_surface(simp.surface,0,total_height*i)
            #if self.do_draw_label:
            #    logo.draw_label(''+' '+str(i))
            self.context.paint()
                


    
class LogoNucSheet(LogoSheet):
    """
    Displays a LogoSheet with a user specified nucleotide sequence below
    each logo. This class is useful for displaying relevance map
    below the sequence being evaluated

    Args:
    	logo_mats: list of matrices holding logo data
    	nuc_onehots: list of one-hot matrices with nucleotide sequence (4xn onehot format)
        input_type:
        	'ppm' for a 4xn position probability matrix
            'pfm' for a 4xn position frequency matrix
            'heights' for 4xn specified heights
        
    """
    def __init__(self,logo_mats,nuc_onehots,input_type='pfm',label_list = None):
        self.base_init(logo_mats,input_type,label_list)
        

        self.nuc_onehots= nuc_onehots
        ##Specific to LogoNucSheet
        self.nuc_height = 20

        #Number of pixels between logo and nuc sequence
        self.nuc_spacer = 10
        
        #Number of pixels between drawn logos
        self.bottom_spacer = 30
            
        #Pixel dimensions of sheet
        self.width = self.logo_list[0].width + 10
        self.height = ((self.logo_list[0].height+
                       self.nuc_spacer+
                       self.nuc_height+
                       self.bottom_spacer)*self.num_logos)


        
        self.surface = cairo.ImageSurface(cairo.FORMAT_ARGB32,self.width,self.height)
        self.context = cairo.Context(self.surface)

        if self.input_type == 'heights':
            self.draw_heights()
        elif self.input_type == 'pfm' or self.input_type == 'ppm':
            self.draw_pwms()


    def draw_pwms(self):
        #LogoNucsheet
        for i,logo in enumerate(self.logo_list):
            #Draw logo
            logo.draw_pwm()
            #set x,y position of logo (y increases from top to down
            total_height = (logo.height+
                           self.nuc_spacer+
                           self.nuc_height+
                           self.bottom_spacer)
                
            self.context.set_source_surface(logo.surface,0,total_height*i)
            if self.do_draw_label:
                logo.draw_label(''+' '+str(i))
            self.context.paint()
                
            seq = self.nuc_onehots[i]
            
            #Draw nucleotide sequence
            seq_img = SeqImg(seq)
            seq_img.draw()
            seq_ypos = total_height*i +self.nuc_spacer
            #Set x,y pos of nucleotide sequence
            self.context.set_source_surface(seq_img.surface,0,seq_ypos)
            self.context.paint()


    def draw_heights(self):
        #LogoNucsheet
        for i,logo in enumerate(self.logo_list):
            #Draw logo
            logo.draw()
            #set x,y position of logo (y increases from top to down
            total_height = (logo.height+
                           self.nuc_spacer+
                           self.nuc_height+
                           self.bottom_spacer)
                
            self.context.set_source_surface(logo.surface,0,total_height*i)
            if self.do_draw_label:
                logo.draw_label(''+' '+str(i))
            self.context.paint()
                
            seq = self.nuc_onehots[i]
            
            #Draw nucleotide sequence
            seq_img = SeqImg(seq)
            seq_img.draw()
            seq_ypos = total_height*i + logo.height+ self.nuc_spacer
            #Set x,y pos of nucleotide sequence
            self.context.set_source_surface(seq_img.surface,0,seq_ypos)
            self.context.paint()

            
        
       

    
class BaseLogo(object):

    #These are cairo glyph indices for specific letters
    BIT_SCALE = 50 #Where to draw the x-axis tick
    A_IND = 36
    T_IND = 55
    G_IND = 42
    C_IND = 38
        
    glyph_dict = { 'A':A_IND,'T':T_IND,'G':G_IND,'C':C_IND}

    NUC_FREQ = 0.25

    def draw_label(self,label):
        label_x=self.x+self.width-80
        label_y=self.y+20
        self.context.save()
        self.context.set_source_rgb(0,0,0)
        self.context.select_font_face("Arial",cairo.FONT_SLANT_NORMAL,
                                      cairo.FONT_WEIGHT_BOLD)
        self.context.move_to(label_x,label_y)
        #self.context.rectangle(self.x,self.y,30,self.height)
        #self.context.fill()
        #self.context.rectangle(self.x,self.y,5,5)
        #self.context.fill()
        self.context.set_font_size(18)
        self.context.show_text(label)
        self.context.restore()
        #save scale show_glyphs restore

    
    def draw_axes(self):
        self.context.save()

        #Draw x-axis line
        self.context.set_line_width(0.5)
        self.context.move_to(self.x_offset,self.x_axis_line)
        self.context.line_to(self.width,self.x_axis_line)
        self.context.stroke()
        #Draw y axis line
        self.context.move_to(self.x_offset,self.x_axis_line)
        self.context.line_to(self.x_offset,0)
        self.context.stroke()        
        self.context.restore()

        num_yticks = int(np.floor(self.height/self.ytick))

        #Draw positive ticks
        for i in range(num_yticks+1):
            self.context.set_line_width(0.5)
            self.context.move_to(self.x_offset,self.x_axis_line-i*self.ytick)
            self.context.line_to(self.x_offset+2,self.x_axis_line-i*self.ytick)
            self.context.stroke()
        


        
            
    def write_to_png(self,fname):
        self.surface.write_to_png(fname)
        
    def write_to_svg(self,fname):
        #svg_surf = cairo.SVGSurface(fname,self.width,self.height)
        svg_surf = cairo.SVGSurface(fname,self.width,self.height)
        svg_context = cairo.Context(svg_surf)
        svg_context.set_source_surface(self.surface,0,0)
        svg_context.paint()
        #svg_context.show_page()

    

    def write_to_pdf(self,fname):
        #ref:http://zetcode.com/gfx/pycairo/backends/
        pdf_surf = cairo.PDFSurface(fname,self.width,self.height)
        pdf_context = cairo.Context(pdf_surf)
        pdf_context.set_source_surface(self.surface,0,0)
        pdf_context.paint()

    def show_imagej_unix(self,fname):
        #Shows the current image by making a system call to imagej
        #This only works on unix systems with imagej installed
        #I wrote this for debugging.
        os.system('imagej'+' '+fname+'&')

    #def show_gtk(self):
    #    #Note: pass self.draw_pwm not self.draw_pwm()
    #    #print self.draw_pwm
    #    display_gtk = DisplayPwmGtk(self.draw_pwm,self.width,self.height)
        



                
class HeightLogo(BaseLogo):
    '''
    This is similar to SeqLogo only the values within the input logo matrix are treated as
    raw letter height values.

    This class treates a 4xn numpy matrix as a map of letter heights with the rows
    corresponding to TCAG.

    

    Letters are sorted by rank, with top ranked letter placed on top
    All four letters are drawn
    '''
    
    def __init__(self,logo_matrix,ytick=10):
        #BaseLogo.__init__(self)
        self.logo_matrix = logo_matrix
        self.ytick = ytick #Set y-ticks
        
        self.min_value = np.min(self.logo_matrix)
        self.max_value = np.max(self.logo_matrix)

        if self.min_value < 0:
            self.has_neg_values = True
        else:
            self.has_neg_values = False 
            
        self.x =0
        self.y =0

        self.seq_len = self.logo_matrix.shape[1]        
        self.font_size =30
        #Width of each nucleotide in the figure
        self.bp_width = self.font_size +1

        if self.has_neg_values:
            self.height=256 #This is distinct from pwm code
        else:
            self.height=128
            
        #self.height = int(self.max_value)
        #print "Height of figure", self.height
        #if self.height > 300:
        #    print "Height of HeightLogo",self.height,"is too high"
        #    print "Setting to 300 pixels"
        #    self.height=300
        self.x_axis_line = self.height-5 
        self.x_offset = 15
        self.x_axis_pad = 2
        self.width = self.bp_width*self.seq_len+self.x_offset+self.x_axis_pad
                    
        self.surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.width,self.height)
        self.context = cairo.Context(self.surface)
        self.context.select_font_face("Monospace",
                                  cairo.FONT_SLANT_NORMAL,
                                  cairo.FONT_WEIGHT_BOLD)
        self.context.set_font_size(self.font_size)
        self.context.set_source_rgb(0.1, 0.1, 0.1)
        self.context.scale(1,1)

    
        
            
    def draw_neg_axes(self):
        self.context.save()

        #Draw x-axis line
        self.context.set_line_width(0.5)
        self.context.move_to(self.x_offset,self.x_axis_line)
        self.context.line_to(self.width,self.x_axis_line)
        self.context.stroke()

        #Draw y axis line
        self.context.move_to(self.x_offset,0)
        self.context.line_to(self.x_offset,self.height)
        self.context.stroke()        
        self.context.restore()

        num_yticks = int(np.floor(self.height/self.ytick))

        #Draw positive ticks
        for i in range(num_yticks+1):
            self.context.set_line_width(0.5)
            self.context.move_to(self.x_offset,self.x_axis_line-i*self.ytick)
            self.context.line_to(self.x_offset+2,self.x_axis_line-i*self.ytick)
            self.context.stroke()
            
        #Draw negative ticks
        for i in range(num_yticks+1):
            self.context.set_line_width(0.5)
            self.context.move_to(self.x_offset,self.x_axis_line+i*self.ytick)
            self.context.line_to(self.x_offset+2,self.x_axis_line+i*self.tick)
            self.context.stroke()

        
               
    def draw(self):
        #HeightLogo
        if self.has_neg_values:
            self.draw_neg_axes()
        else:
            self.draw_axes()
        self.context.save()
        #Get ranks of each column.
        #Biggest value gets highest rank number and get drawn first
        filter_ranks = self.logo_matrix.argsort(axis=0)
        nucs = self.logo_matrix.shape[1]
        letters = self.logo_matrix.shape[0] #T,C,A,G
        row_dict = {0:'T',1:'C',2:'A',3:'G'}
        
        #i is nucleotide index, j is letter index
        for i in range(nucs):
            x_start_spacer = 4
            #Note:x_axis pad just adds some space between x=0 and the first
            
            xpos = self.bp_width*i+self.x_offset+self.x_axis_pad
            ranks = np.ndarray.tolist(filter_ranks[:,i])
            dy_pos = self.x_axis_line
            dy_neg = self.x_axis_line

            for rank_ind  in ranks:
                
                cur_let_str = row_dict[rank_ind]
                cur_height = self.logo_matrix[rank_ind,i]
                #ucLetter.BIT_SCALE = 100
                cur_let = NucLetter(self.context,
                                        cur_let_str,
                                        xpos,
                                        0,
                                        cur_height,
                                        rank_ind)
                if cur_let.signed_height>0: #If positive weight
                    cur_let.move(xpos,dy_pos)
                    dy_pos = dy_pos-cur_let.signed_height
                    self.context.scale(1,1)
                    cur_let.draw()
                elif cur_let.signed_height<0: #If negative weight
                    dy_neg = dy_neg-cur_let.signed_height
                    cur_let.move(xpos,dy_neg)
                    self.context.scale(1,1)
                    cur_let.draw()
                    

        self.context.restore()

    
   
class SimpleHeightLogo(HeightLogo):
    '''
    Similar to height logo, except this class
    takes a nucleotide string, and a 1D array of heights
    and scales each letter to its corresponding height.

    Only one letter is drawn per position
    
    '''

    def __init__(self,heights,nuc_seq,ytick=10):
        self.nuc_seq = list(nuc_seq)
        self.col_heights = np.sum(heights,axis=0)
        
        super(SimpleHeightLogo, self).__init__(heights,ytick)
        if self.col_heights.shape[0] != len(self.nuc_seq):
            print "SimpleHeightLogo init error dims do not match"
            print "heights.shape[1]:{}\tlength of nuc_seq:{}".\
                                             format(heights.shape[1],len(self.nuc_seq))

       
    def draw(self):
        #SimpleHeightLogo
        if self.has_neg_values:
            self.draw_neg_axes()
        else:
            self.draw_axes()
        self.context.save()

        

        row_dict = {0:'T',1:'C',2:'A',3:'G'}
        
        #i is nucleotide index, j is letter index
        for i,nuc in enumerate(self.nuc_seq):
            x_start_spacer = 4
            #Note:x_axis pad just adds some space between x=0 and the first
            
            xpos = self.bp_width*i+self.x_offset+self.x_axis_pad
            #ranks = np.ndarray.tolist(filter_ranks[:,i])
            dy_pos = self.x_axis_line
            dy_neg = self.x_axis_line

            cur_let = NucLetter(self.context,
                                nuc,
                                xpos,
                                0,
                                self.col_heights[i],
                                0)
            if cur_let.signed_height>0: #If positive weight
                cur_let.move(xpos,dy_pos)
                dy_pos = dy_pos-cur_let.signed_height
                self.context.scale(1,1)
                cur_let.draw()
            elif cur_let.signed_height<0: #If negative weight
                dy_neg = dy_neg-cur_let.signed_height
                cur_let.move(xpos,dy_neg)
                self.context.scale(1,1)
                cur_let.draw()
                    
        self.context.restore()

         
        
class SeqImg(BaseLogo):
    '''
    A class for drawing an image of a nucleotide sequence
    No height transformations. Just a simple sequence
    '''

    def __init__(self,nuc_onehot_matrix,font_size=10):
        self.nuc_onehot = nuc_onehot_matrix
        self.nuc_string = dbt.onehot_to_nuc(self.nuc_onehot)
        self.seq_len = int(self.nuc_onehot.shape[1])
        if self.seq_len != int(np.sum(self.nuc_onehot)):
            print "Error! nuc matrix is not one-hot for SeqImg"
        
        self.x = 0
        self.y = 0
        self.font_size =30
        #Width of each nucleotide in the figure
        self.bp_width = self.font_size +1

        #the following dimension params are used to keep this sequence aligned
        #with those produced by SeqLogo

        
        self.x_offset = 15
        self.x_axis_pad = 2
        self.width = self.bp_width*self.seq_len+self.x_offset+self.x_axis_pad
        self.height = 32
        self.bp_height=.25
        
        self.surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.width,self.height)
        self.context = cairo.Context(self.surface)
        self.context.select_font_face("Monospace",
                                  cairo.FONT_SLANT_NORMAL,
                                  cairo.FONT_WEIGHT_BOLD)
        self.context.set_font_size(self.font_size)
        self.context.set_source_rgb(0.1, 0.1, 0.1)
        self.context.scale(1,1)
       
        
        #self.draw()


    def draw(self):
        #SeqImg
        self.context.save()
        for i,letter in enumerate(list(self.nuc_string)):
            xpos = self.bp_width*i+self.x_offset+self.x_axis_pad
            nuclet = NucLetter(self.context,letter,xpos,20,self.bp_height)
            self.context.scale(1,1)
            nuclet.draw()
        self.context.restore()



        
class SeqLogo(BaseLogo):

    def __init__(self,np_pfm):
        self.init_from_pfm(np_pfm)
        self.ytick = BaseLogo.BIT_SCALE

            
    @classmethod
    def init_from_nuc_list(cls,nuc_str_list):
        '''
        Initialize from a list of nucleotides (strings, not Biopython objects)
        Call SeqLogo.init_from_nuc_list(my_nucs)
        to initialize in this manner
        '''
        pfm = PwmTools.pfm_from_nucs(nuc_str_list)
        cls.init_from_pfm(pfm)
        

    def init_from_pfm(self,np_pfm):
        '''Initialize from a position frequecy matrix'''
        
        '''Tip: If passing a convolution filter as a pfm,
           multiply the convolution matrix by the number of samples
           that was used to determine the makeup of the convolution filter '''
        #BaseLogo.__init__(self)
        self.x =0
        self.y =0
        self.pfm = np_pfm
        self.ppm = PwmTools.pfm_to_ppm(self.pfm)
        self.pwm = PwmTools.pfm_to_pwm(self.pfm)
        self.ic  = PwmTools.pfm_to_ic(self.pfm)
        self.seq_len = np_pfm.shape[1]        
        
        self.font_size =30
        #Width of each nucleotide in the figure
        self.bp_width = self.font_size +1

        self.height=128
        self.x_offset = 15
        self.x_axis_pad = 2
        self.width = self.bp_width*self.seq_len+self.x_offset+self.x_axis_pad
                    
        self.surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.width,self.height)
        self.context = cairo.Context(self.surface)
        self.context.select_font_face("Monospace",
                                  cairo.FONT_SLANT_NORMAL,
                                  cairo.FONT_WEIGHT_BOLD)
        self.context.set_font_size(self.font_size)
        self.context.set_source_rgb(0.1, 0.1, 0.1)
        self.context.scale(1,1)
        self.x_axis_line = self.height-5
        
        
    def set_pfm(self,new_pfm):
        self.__init__(new_pfm)
        #self.pfm = new_pfm
        #self.ppm = PwmTools.pfm_to_ppm(new_pfm,True)
        #self.pwm = PwmTools.pfm_to_pwm(new_pfm,True)
        #self.ic = PwmTools.pfm_to_ic(new_pfm,True)

    def set_surface_dims(self,width,height):
        self.surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width,height)
        self.context = cairo.Context(self.surface)
    
            
    def print_ic(self):
        print self.ic
        return self.ic
        
    def print_pfm(self):
        print self.pfm
        return self.pfm
    
    def print_ppm(self):
        print self.ppm
        return self.ppm

    def print_logo_ic_heights(self):
        
        logo_heights = self.get_logo_ic_heights()
        
        return logo_heights

    def get_logo_ic_heights(self):
        np.set_printoptions(threshold=np.nan)
                
        logo_heights = self.ppm*self.ic
        
        return logo_heights


    
    
    def draw_pwm(self):
        self.draw_axes()
        self.context.save()
        logo_heights = self.get_logo_ic_heights()
    
        #Get ranks of each column.
        #Biggest value gets highest rank number and get drawn first
        logo_ranks = logo_heights.argsort(axis=0)
        seq_len= logo_heights.shape[1]
        #num_letters = logo_heights.shape[0]


        row_dict = {0:'T',1:'C',2:'A',3:'G'}
        
        #i is nucleotide index, j is letter index
        for i in range(seq_len):
            x_start_spacer = 4
            #Note:x_axis pad just adds some space between x=0 and the first
            #nuc
            xpos = self.bp_width*i+self.x_offset+self.x_axis_pad
            
            #First number indicates index of lowest height number
            #Lowest number should get drawn first
            ranks = np.ndarray.tolist(logo_ranks[:,i])

            dy = self.x_axis_line
            for rank_ind in ranks:
                #use 4-rank_ind b/c of rank reversal 
                cur_let_str = row_dict[rank_ind]
                cur_height = logo_heights[rank_ind,i]
                #self.height here is the height of the seq logo figure.
                #Populate dictionary to store Each letter needs to be initialized to calculate height 
                cur_let = NucLetter(self.context,cur_let_str,xpos,0,cur_height,rank_ind)
                cur_let.move(xpos,dy)
                dy = dy-cur_let.height
                self.context.scale(1,1)
                cur_let.draw()
        self.context.restore()

               
        
    


class NucLetter:
    #Cario glyph indices for the font monospace 

    A_IND = 36
    T_IND = 55
    G_IND = 42
    C_IND = 38

    glyph_dict = { 'A':A_IND,'T':T_IND,'G':G_IND,'C':C_IND}

    BIT_SCALE= BaseLogo.BIT_SCALE #the height in pixels equivalent to 1 bit on our final scale

    def __init__(self,cairo_context,nuc_letter,xpos=0,ypos=0, input_height = 1.0,rank=0):
        self.x=xpos
        self.y=ypos
        self.context = cairo_context
        self.nuc_letter = nuc_letter
        self.letter_ind = NucLetter.glyph_dict[self.nuc_letter]
        #print self.context.glyph_extents([(self.letter_ind,0,0)])
        self.base_width = self.context.glyph_extents([(self.letter_ind,0,0)])[2]
        self.base_height = self.context.glyph_extents([(self.letter_ind,0,0)])[3]
        #print "Base_height",self.base_height
        self.width = self.base_width
        #Rescale letter to specified height
        self.signed_height= math.floor(input_height*NucLetter.BIT_SCALE)
        self.y_scale = abs(input_height*NucLetter.BIT_SCALE/self.base_height)
        self.height = math.floor(self.base_height * self.y_scale)
        #print "Height",self.height
        #self.height=self.base_height
        self.bottom = self.y+self.height
        self.right = self.x+self.width
        self.color = self.color_by_letter()
        self.rank=rank

    def move(self,xpos,ypos):
       self.x = xpos
       self.y = ypos

    def color_by_letter(self):
        if self.nuc_letter =='A':
            return (1,1.0,0,0) #red
        elif self.nuc_letter =='T':
            return (1,0,1.0,0)#green
        elif self.nuc_letter =='G':
            return (1,0,0,1.0,1) #blue
        elif self.nuc_letter=='C':
            return (1,.8,.8,0) #yellow



    def draw(self):
        self.context.save()
        self.context.set_source_rgb(self.color[1],self.color[2],self.color[3])
        if (self.y_scale>0):
            self.context.scale(1.0,self.y_scale)
            self.context.translate(0,-self.y*(self.y_scale-1)/self.y_scale)
            #self.context.rectangle(self.x,self.y,30,self.height)
            #self.context.fill()
            #self.context.rectangle(self.x,self.y,5,5)
            #self.context.fill()
            self.context.show_glyphs([(self.letter_ind,self.x,self.y)])
        self.context.restore()
        #save scale show_glyphs restore


class PwmTools:
    #Deciding on the pseudocount value has been done in a paper from 2009
    #It is recommended to add 0.8/4 to each entry of the pfm
    #http://www.ncbi.nlm.nih.gov/pmc/articles/PMC2647310/
    PSEUDOCOUNT = 0.8
    #PSEUDOCOUNT = 0.00000001  
    #This pseudocount can have a large
    #impact if the pfm was derived from few sequences


    #Decent refreshers on how to build these:
    #http://biologie.univ-mrs.fr/upload/p202/01.4.PSSM_theory.pdf
    #https://en.wikipedia.org/wiki/Sequence_logo
    
    @staticmethod
    def pfm_from_nucs(nuc_str_list):
        num_nucs = len(nuc_str_list)
        nuc_len = len(nuc_str_list[0])
        pfm = np.zeros((4,nuc_len))
        for nuc_str in nuc_str_list:
            pfm += dbt.seq_to_onehot(nuc_str)
        return pfm
    
    @staticmethod
    def pfm_to_ppm(pfm_arr,use_pseudocounts=False):
        '''Convert a numpy position frequency matrix (PFM) to a
           position probability matrix (PPM).
           Rows are in order 'TCAG'
        '''
        
        pfm_arr = np.asarray(pfm_arr, dtype='float32')
        if use_pseudocounts:
            #Add pseudocounts to the ppm to avoid infinities
            pseudo_sums = np.sum(pfm_arr,axis=0)+PwmTools.PSEUDOCOUNT
            return np.true_divide((pfm_arr+(PwmTools.PSEUDOCOUNT/4.)),pseudo_sums)
        else:
            sums = np.sum(pfm_arr,axis=0)
            return np.true_divide(pfm_arr,sums)

    @staticmethod
    def ppm_to_pwm(ppm_arr):
        '''Convert a numpy position probability matrix (PPM) to a
           position weight matrix (PWM).
           Rows are in order 'TCAG'
        '''
        NUC_FREQ = 0.25
        #Clipping is to avoid getting infinite or NaN values
        return np.log(np.clip( np.true_divide((ppm_arr),NUC_FREQ) ,1e-10,1.0))
        
    @staticmethod
    def pfm_to_pwm(pfm_arr,use_pseudocounts = True):
        '''Convert a numpy position frequency matrix (PFM) to a
           position weight matrix (PWM).
           Rows are in order 'ATGC'
        '''
        if (use_pseudocounts):
            return PwmTools.ppm_to_pwm(
                PwmTools.pfm_to_ppm(pfm_arr,True))
        return PwmTools.ppm_to_pwm(PwmTools.pfm_to_ppm(pfm_arr))
        
    @staticmethod
    def pfm_to_ic(pfm_arr,use_pseudocounts=True):
        #ic stands for information content
        #Add pseudocounts to avoid inifinites
        if (use_pseudocounts):
            ppm = PwmTools.pfm_to_ppm(pfm_arr,True)
        else:
            ppm = PwmTools.pfm_to_ppm(pfm_arr)
        #en is the small-sample correction. This value is 0 for num_seqs > 3
        #This method of calculating num_seqs might throw an error in certain cases    
        num_seqs = np.sum(pfm_arr[:,0],axis=0)
        en = (1/np.log(2.))*((4.-1)/(2.*num_seqs))
        return np.log2(ppm.shape[0]) -( -np.sum(ppm*np.log2(np.clip(ppm,1e-10,1.0)),axis=0) + en)


    @staticmethod
    def ppm_to_ic(ppm,use_pseudocounts = True):
        #Note: for this function, the small-sample correction is not applied
        # since the number of sequences in unknown.
        #This should be valid if num_seqs > 3
        return np.log2(ppm.shape[0]) -( -np.sum(ppm*np.log2(np.clip(ppm,1e-10,1.0)),axis=0))

    @staticmethod
    def ppm_to_logo_heights(ppm):
        ic = PwmTools.ppm_to_ic(ppm)
        return ppm*ic
    
    @staticmethod
    def pfm_to_logo_heights(pfm_arr):
        ppm = PwmTools.pfm_to_ppm(pfm_arr) 
        ic = PwmTools.pfm_to_ic(pfm_arr)
        return ppm*ic

        
'''
class DisplayPwmGtk():
    #ref: http://zetcode.com/gfx/pycairo/images/        

    
    def __init__(self,draw_func,width,height):
        self.draw_func =draw_func
        self.width = width
        self.height =height
        self.window = gtk.Window(gtk.WINDOW_TOPLEVEL)
        self.window.resize(width,height)
        self.window.set_title("PWM")
        self.window.set_position(gtk.WIN_POS_CENTER)
        #self.window.set_position(gtk.WindowPosition.CENTER)
        gtk.set_from_pixmap
        #self.draw_area = gtk.DrawingArea()
        #self.draw_area.connect("draw",self.on_draw)
        self.window.add(draw_area)


        #self.button = gtk.Button("Close")
        #self.button.connect("clicked",self.destroy)
        #self.window.add(self.button)
        self.window.connect("destroy",gtk.main_quit)

        #self.button.show()
        self.window.show()

    def on_draw(self,wid,cr):
        self.draw_func()
        
        
    def main(self):
        gtk.main()

    def destroy(self,widget,data=None):
        gtk.main_quit()

'''
       

        
