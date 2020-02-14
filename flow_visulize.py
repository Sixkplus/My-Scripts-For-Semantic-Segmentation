# convert .flo to rgb and visualize
import numpy as np
import cvbase as cvb
from skimage.io import imsave

flow_name = 'out.flo'
flow = cvb.read_flow(flow_name)
a = cvb.flow2rgb(flow)
imsave('a.png', a)