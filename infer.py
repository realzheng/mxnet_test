# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
import sys
sys.path.insert(0, "../../python")
import mxnet as mx
import numpy as np
import cv2

import time
import datetime


def get_ocrnet():

    data = mx.symbol.Variable('data')

    conv1 = mx.symbol.Convolution(data=data, kernel=(3,3), num_filter=32)
    relu1 = mx.symbol.Activation(data=conv1, act_type="relu")
    pool1 = mx.symbol.Pooling(data=relu1, pool_type="max", kernel=(2,2), stride=(2, 2))

    conv2 = mx.symbol.Convolution(data=pool1, kernel=(3,3), num_filter=64)
    relu2 = mx.symbol.Activation(data=conv2, act_type="relu")
    pool2 = mx.symbol.Pooling(data=relu2, pool_type="max", kernel=(2,2), stride=(2, 2))

    conv3 = mx.symbol.Convolution(data=pool2, kernel=(3,3), num_filter=64)
    relu3 = mx.symbol.Activation(data=conv3, act_type="relu")
    pool3 = mx.symbol.Pooling(data=relu3, pool_type="max", kernel=(2,2), stride=(2, 2))

    flatten = mx.symbol.Flatten(data = pool3)

    fc4 = mx.symbol.FullyConnected(data = flatten, num_hidden = 128)
    relu4 = mx.symbol.Activation(data=fc4, act_type="relu")
    drop4 = mx.symbol.Dropout(data=relu4, p=0.3)

    fc5 = mx.symbol.FullyConnected(data = drop4, num_hidden = 256)
    relu5 = mx.symbol.Activation(data=fc5, act_type="relu")
    drop5 = mx.symbol.Dropout(data=relu5, p=0.3)
  
    fc61 = mx.symbol.FullyConnected(data = drop5, num_hidden = 33)
    fc62 = mx.symbol.FullyConnected(data = drop5, num_hidden = 33)
    fc63 = mx.symbol.FullyConnected(data = drop5, num_hidden = 33)
    fc64 = mx.symbol.FullyConnected(data = drop5, num_hidden = 33)

    fc6 = mx.symbol.Concat(*[fc61, fc62, fc63, fc64], dim = 0)
    #fc6 = mx.symbol.Concat(fc61, fc62, fc63, fc64, dim = 0)

    return mx.symbol.SoftmaxOutput(data = fc6, name = "softmax")

if __name__ == '__main__':

    datapath = "example/"
    
    batch_size = 1
    _, arg_params, __ = mx.model.load_checkpoint("multitask",990)

    taglist=[]

    for line in file(datapath+"tag.txt"):
        line = line.rstrip()
        taglist.append(line)
    
    
    data_shape = [("data", (batch_size, 3, 32, 90))]
    input_shapes = dict(data_shape)
    sym = get_ocrnet()

  
    executor = sym.simple_bind(ctx = mx.cpu(), **input_shapes)
    for key in executor.arg_dict.keys():
        if key in arg_params:
            arg_params[key].copyto(executor.arg_dict[key])
            #print np.shape(arg_params[key].asnumpy())


    img = cv2.imread(datapath+"test.jpg",cv2.IMREAD_COLOR)
    img = np.multiply(img,1/255.0)
    img = img.transpose(2, 0, 1) # (32,90,3)->(3,32,90)

    start = time.time()
    #print start

    executor.forward(is_train = False, data = mx.nd.array([img]))

    probs = executor.outputs[0].asnumpy()

    #print probs

    end = time.time()

    #print end

    print "running time: %f s" % (end - start)

 
    line = ''
    for i in range(probs.shape[0]):
        #line += str(np.argmax(probs[i]))
        line += taglist[np.argmax(probs[i])]
        line +=" "
    print 'predicted: ' + line
