import logging
from .CVAE import *
from .Discriminator import *


def build_CVAD_net(dataset_name, net_name, capacity, channel):   
    
    logger = logging.getLogger()
    logger.info("Build CVAD embnet for {}".format(dataset_name))
    assert dataset_name in ['rsna', 'siim']
    
    netG = CVAE(capacity,channel)
    netD = Discriminator(capacity, channel)

    return netG, netD
