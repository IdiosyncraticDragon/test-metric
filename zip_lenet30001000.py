import sys
sys.path.insert(0, './python/')
import caffe
import numpy as np
import pdb
weights='./models/lenet30001000/pruned_lenet30001000.caffemodel'
proto='./models/lenet30001000/lenet_train_test_fortest.prototxt'
net=caffe.Net(proto, weights, caffe.TEST)
total=0
aa=0
# for each layer, a mask is applied to the original weights and bias.
# here, for net.params['ip1'], net.params['ip1'][0] is the weights, net.params['ip1'][1] is the bias,
#       net.params['ip1'][2] is the mask for the weights, net.params['ip1'][3] is the mask for the bias.
#       if one of the element value in the mask is 0, the corresponding element in network is pruned.
w_m=2
b_m=3

index_dict = {'ip1_w': None, 'ip1_b':None, 'ip2_w':None, 'ip2_b':None, 'ip3_w':None, 'ip3_b':None}
param_dict = {'ip1_w': None, 'ip1_b':None, 'ip2_w':None, 'ip2_b':None, 'ip3_w':None, 'ip3_b':None}

index_dict['ip1_b'] =  np.where(net.params['ip1'][b_m].data != 0)[0]
index_dict['ip1_w'] =  np.where(net.params['ip1'][w_m].data != 0)[0]
index_dict['ip2_b'] =  np.where(net.params['ip2'][b_m].data != 0)[0]
index_dict['ip2_w'] =  np.where(net.params['ip2'][w_m].data != 0)[0]
index_dict['ip3_b'] =  np.where(net.params['ip3'][b_m].data != 0)[0]
index_dict['ip3_w'] =  np.where(net.params['ip3'][w_m].data != 0)[0]
a1=len(index_dict['ip1_b'])
a2=len(index_dict['ip1_w'])
a3=len(index_dict['ip2_b'])
a4=len(index_dict['ip2_w'])
a5=len(index_dict['ip3_b'])
a6=len(index_dict['ip3_w'])

param_dict['ip1_b'] =  net.params['ip1'][1].data[index_dict['ip1_b']].astype(np.float16)
param_dict['ip1_w'] =  net.params['ip1'][0].data[index_dict['ip1_w']].astype(np.float16)
param_dict['ip2_b'] =  net.params['ip2'][1].data[index_dict['ip2_b']].astype(np.float16)
param_dict['ip2_w'] =  net.params['ip2'][0].data[index_dict['ip2_w']].astype(np.float16)
param_dict['ip3_b'] =  net.params['ip3'][1].data[index_dict['ip3_b']].astype(np.float16)
param_dict['ip3_w'] =  net.params['ip3'][0].data[index_dict['ip3_w']].astype(np.float16)
b1=net.params['ip1'][0].data.size+net.params['ip1'][1].data.size
b2=net.params['ip2'][0].data.size+net.params['ip2'][1].data.size
b3=net.params['ip3'][0].data.size+net.params['ip3'][1].data.size
#pdb.set_trace()

aa = a1+a2+a3+a4+a5+a6
total = b1+b2+b3

print 'Compression rate :{}% ({}x)'.format(1- aa*1./total,total*1./aa)
print 'ip1:{}%'.format((a1+a2)*100./b1)
print 'ip2:{}%'.format((a3+a4)*100./b2)
print 'ip3:{}%'.format((a5+a6)*100./b3)
np.savez_compressed('lenet300100_ziped', index_dict, param_dict)
