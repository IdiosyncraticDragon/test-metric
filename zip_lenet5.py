import sys
sys.path.insert(0, './python/')
import caffe
import numpy as np
import pdb
weights='./models/lenet5/pruned_lenet5.caffemodel'
proto='./models/lenet5/lenet_train_test_fortest.prototxt'
net=caffe.Net(proto, weights, caffe.TEST)
total=0
aa=0
w_m=2
b_m=3

index_dict = {'conv1_w': None, 'conv1_b':None, 'conv2_w':None, 'conv2_b':None, 'ip1_w':None, 'ip1_b':None, 'ip2_w':None, 'ip2_b':None}
param_dict = {'conv1_w': None, 'conv1_b':None, 'conv2_w':None, 'conv2_b':None, 'ip1_w':None, 'ip1_b':None, 'ip2_w':None, 'ip2_b':None}

index_dict['conv1_b'] = np.where(net.params['conv1'][b_m].data != 0)[0]
index_dict['conv1_w'] = np.where(net.params['conv1'][w_m].data != 0)[0]
index_dict['conv2_b'] = np.where(net.params['conv2'][b_m].data != 0)[0]
index_dict['conv2_w'] = np.where(net.params['conv2'][w_m].data != 0)[0]
index_dict['ip1_b'] = np.where(net.params['ip1'][b_m].data != 0)[0]
index_dict['ip1_w'] = np.where(net.params['ip1'][w_m].data != 0)[0]
index_dict['ip2_b'] = np.where(net.params['ip2'][b_m].data != 0)[0]
index_dict['ip2_w'] = np.where(net.params['ip2'][w_m].data != 0)[0]

a1=len(index_dict['conv1_b'])
a2=len(index_dict['conv1_w'])
a3=len(index_dict['conv2_b'])
a4=len(index_dict['conv2_w'])
a5=len(index_dict['ip1_b'])
a6=len(index_dict['ip1_w'])
a7=len(index_dict['ip2_b'])
a8=len(index_dict['ip2_w'])

param_dict['conv1_b'] = net.params['conv1'][1].data[index_dict['conv1_b']].astype(np.float16)
param_dict['conv1_w'] = net.params['conv1'][0].data[index_dict['conv1_w']].astype(np.float16)
param_dict['conv2_b'] = net.params['conv2'][1].data[index_dict['conv2_b']].astype(np.float16)
param_dict['conv2_w'] = net.params['conv2'][0].data[index_dict['conv2_w']].astype(np.float16)
param_dict['ip1_b'] = net.params['ip1'][1].data[index_dict['ip1_b']].astype(np.float16)
param_dict['ip1_w'] = net.params['ip1'][0].data[index_dict['ip1_w']].astype(np.float16)
param_dict['ip2_b'] = net.params['ip2'][1].data[index_dict['ip2_b']].astype(np.float16)
param_dict['ip2_w'] = net.params['ip2'][0].data[index_dict['ip2_w']].astype(np.float16)

b1=net.params['conv1'][0].data.size+net.params['conv1'][1].data.size
b2=net.params['conv2'][0].data.size+net.params['conv2'][1].data.size
b3=net.params['ip1'][0].data.size+net.params['ip1'][1].data.size
b4=net.params['ip2'][0].data.size+net.params['ip2'][1].data.size

aa = a1+a2+a3+a4+a5+a6+a7+a8
total = b1+b2+b3+b4

print 'Compression rate :{}% ({}x)'.format(1- aa*1./total,total*1./aa)
print 'conv1:{}%'.format((a1+a2)*100./b1)
print 'conv2:{}%'.format((a3+a4)*100./b2)
print 'ip1:{}%'.format((a5+a6)*100./b3)
print 'ip2:{}%'.format((a7+a8)*100./b4)
np.savez_compressed('lenet5_zipped', index_dict, param_dict)
