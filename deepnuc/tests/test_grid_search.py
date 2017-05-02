import itertools
from collections import OrderedDict

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__),os.path.pardir))

from modelparams import GridParams
import nucconvmodel


def test():
    gp = GridParams(
              seq_len=100,
              num_epochs=[50,40],
              learning_rate=[1e-4,1e-5],
              batch_size=[24],
              keep_prob = [0.5],
              beta1 = [0.9],
              concat_revcom_input=[True,False],
              inference_method=[nucconvmodel.inferenceA])

    print "Num models:\t",len(gp.grid_params),"\n"

    for param in gp.grid_params:
        param.print_params()


if __name__=="__main__":
    test()
