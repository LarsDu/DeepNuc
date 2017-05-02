
from LogoTools import SeqLogo
from LogoTools import PwmTools
from LogoTools import LogoSheet
from LogoTools import LogoNucSheet
from LogoTools import HeightLogo
from LogoTools import SimpleHeightLogo
import numpy as np


nuc1 = np.array([[ 0. ,   0.,   0.,  0.0,  0.0,  0. ,  0. ,  1.0 ],
                 [ 1. ,  0. ,  1.0,  0.0,  1.0,  0.0,  1.0,  0.0],
                 [ 0. ,  1.0,   0.,  1.0,  0.0,  1.0,  0.0,  0.0],
                 [ 0. ,  0. ,   0.,  0.0,  0.0,  0. ,  0. ,  0. ]])


ppm1 = np.array([[ 0. ,  0. ,  0. ,  0.3,  0.2,  0. ,  0. ,  0. ],
                 [ 0.8,  0.2,  0.8,  0.3,  0.4,  0.2,  0.8,  0.2],
                 [ 0.2,  0.8,  0.2,  0.4,  0.3,  0.8,  0.2,  0.8],
                 [ 0. ,  0. ,  0. ,  0. ,  0.1,  0. ,  0. ,  0. ]])

ppm3 = np.array([[ 0. ,  0. ,  0.1 ,  0.3,  0.2,  0. ,  0. ,  0.05 ],
                 [ 0.8,  0.2,  0.7,  0.3,  0.4,  0.2,  0.8,  0.05],
                 [ 0.2,  0.,  0.1,  0.4,  0.3,  0.8,  0.2,  0.1],
                 [ 0. ,  0.8 ,  0.1 ,  0. ,  0.1,  0. ,  0. ,  0.8 ]])

pfm3 = ppm3*1000

pfm1 = ppm1*11000

print pfm1


ppm2 = np.array([[.8, .7, .05, .8, 0, .2, .5, .6],
                [.2, .1, .7,   .2, 0, .25, .1, .3],
                [0, .1, .05,   0, .5, .25, .1, .05],
                [0, .1, .2,    0, .5, .05, .3, .05]])


ppm2 = np.array([[.8, .7, .05, .8, 0, .2, .5, .6],
                [.2, .1, .7,   .2, 0, .25, .1, .3],
                [0, .1, .05,   0, .5, .25, .1, .05],
                [0, .1, .2,    0, .5, .05, .3, .05]])

ppm3 = np.array([[.8, .7, .05, .1, 0, .2, .5, .6],
                [.2, .1, .7,   .2, 0, .25, .1, .3],
                [0, .1, .05,   .35, .5, .25, .1, .05],
                [0, .1, .2,    .35, .5, .05, .3, .05]])

ppm4=np.array([[ 239.,  250.,  244.,  253.,  240.,  262.,  255.,  251.,  244.,  261.,  240.,  258.],
 [ 248.,  255.,  257.,  240.,  245.,  234.,  254.,  258.,  253.,  250.,  264.,  255.],
 [ 270.,  255.,  247.,  259.,  261.,  244.,  247.,  228.,  261.,  255.,  246.,  230.],
 [ 241.,  239.,  250.,  245.,  252.,  258.,  242.,  261.,  239.,  232.,  247.,  255.]])

pfm2 = ppm2*1000
pfm3 = ppm3*1000

hlogo_test = HeightLogo(ppm3)
hlogo_test.draw()
hlogo_test.write_to_png('hlogo_test.png')

shlogo = SimpleHeightLogo(ppm3,'ATACGTAC')
shlogo.draw()
shlogo.write_to_png('shlogo_test.png')
'''
print pfm1.argsort(axis=0)
mylogo = SeqLogo(pfm3)
mylogo.print_ic()
print "IC heights"
print pfm3.shape
#print mylogo.get_logo_ic_heights()
#mylogo.draw_pwm()
#mylogo.write_to_png('test2.png')
#mylogo.write_to_svg('test2.svg')
#mylogo.write_to_pdf('test2.pdf')
#mylogo.show_gtk()


nuc_sheet = LogoSheet([pfm1],input_type='pfm')
nuc_under_sheet = LogoNucSheet([pfm1],[nuc1],input_type='pfm')


pwm_sheet = LogoSheet([pfm1,pfm2,pfm3,ppm4],input_type='pfm') #draw called by init
height_sheet = LogoSheet([ppm3*5],input_type='heights')

nuc_sheet.write_to_png('nuc_test1.png')
nuc_under_sheet.write_to_png('nuc_under_test1.png')

pwm_sheet.write_to_png('test_refactor.png')
#height_sheet.write_to_svg('test_refactor.svg')
#pwm_sheet.write_to_pdf('test_refactor.pdf')
'''

