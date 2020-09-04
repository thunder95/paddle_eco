import paddle.fluid as fluid
import numpy as np
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear, Conv3D, Dropout

#conv_bn_relu
class ConvBNLayer(fluid.dygraph.Layer):
    def __init__(self,
                 name_scope,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1,
                 act='relu', is_3d=False):
        super(ConvBNLayer, self).__init__(name_scope)

        self._conv = None
        if is_3d:
            self._conv = Conv3D(
                num_channels=num_channels,
                num_filters=num_filters,
                filter_size=filter_size,
                stride=stride,
                padding=(filter_size - 1) // 2,
                act=None,
                bias_attr=False)
        else:
            self._conv = Conv2D(
                num_channels=num_channels,
                num_filters=num_filters,
                filter_size=filter_size,
                stride=stride,
                padding=(filter_size - 1) // 2,
                act=None,
                bias_attr=False)

        self._batch_norm = BatchNorm(num_filters, act=act)
        
    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        return y

#inceptionv2(bn-inception): 参考https://blog.csdn.net/stesha_chen/article/details/81662405
#总体上分为三种结构，分别是max_pool(中间)， max_pool(最后)， 另一个是avg_pool, 后两者通过传参合并处理

#inception_avg, 卷积核结构一样， 输入和每个num_filters不一样
class InceptionBasic(fluid.dygraph.Layer):
    def __init__(self, name_scope, input_channels, filter_list, pool_mode="avg"):
        super(InceptionBasic, self).__init__(name_scope)
        #1*1
        self.branch_1 = ConvBNLayer(
            self.full_name(),
            num_channels=input_channels,
            num_filters=filter_list[0],
            filter_size=1,
            stride = 1,
            act='relu')

        #1*1 + 3*3
        self.branch_2_a = ConvBNLayer(
            self.full_name(),
            num_channels=input_channels,
            num_filters=filter_list[1],
            filter_size=1,
            stride = 1,
            act='relu')
        self.branch_2_b = ConvBNLayer(
            self.full_name(),
            num_channels=filter_list[1],
            num_filters=filter_list[2],
            filter_size=3,
            stride = 1,
            act='relu') #注意padding = 1， 也就是(3-1)//2
            

        #1*1 + 3*3 + 3*3
        self.branch_3_a = ConvBNLayer(
            self.full_name(),
            num_channels=input_channels,
            num_filters=filter_list[3],
            filter_size=1,
            stride = 1,
            act='relu')
        self.branch_3_b = ConvBNLayer(
            self.full_name(),
            num_channels=filter_list[3],
            num_filters=filter_list[4],
            filter_size=3,
            stride = 1,
            act='relu')
        self.branch_3_c = ConvBNLayer(
            self.full_name(),
            num_channels=filter_list[4],
            num_filters=filter_list[5],
            filter_size=3,
            stride = 1,
            act='relu')

        #avg_pool3*3 + 1*1
        self.branch_4_a = Pool2D(pool_size=3, pool_stride=1, pool_padding=1, pool_type=pool_mode, ceil_mode=True)
        self.branch_4_b = ConvBNLayer(
            self.full_name(),
            num_channels=input_channels,
            num_filters=filter_list[6],
            filter_size=1,
            stride = 1,
            act='relu')

    def forward(self, inputs):
        b1 = self.branch_1(inputs)

        b2 = self.branch_2_a(inputs)
        b2 = self.branch_2_b(b2)

        b3 = self.branch_3_a(inputs)
        b3 = self.branch_3_b(b3)
        b3 = self.branch_3_c(b3)

        b4 = self.branch_4_a(inputs)
        b4 = self.branch_4_b(b4)

        # print(b1.shape)
        # print(b2.shape)
        # print(b3.shape)
        # print(b4.shape)

        concat = fluid.layers.concat([b1, b2, b3, b4], axis=1)
        return concat


#inception_max(中间), 分成3组实现
class Inception3c_1(fluid.dygraph.Layer):
    def __init__(self, name_scope,  input_channels, filter_list):
        super(Inception3c_1, self).__init__(name_scope)
        #1*1 + 3*3
        self.conv_bn_0 = ConvBNLayer(
            self.full_name(),
            num_channels=input_channels,
            num_filters=filter_list[0],
            filter_size=1,
            stride = 1,
            act='relu')
        
        self.conv_bn_1 = ConvBNLayer(
            self.full_name(),
            num_channels=filter_list[0],
            num_filters=filter_list[1],
            filter_size=3,
            stride = 2,
            act='relu') #stride=2

    def forward(self, inputs):
        x = self.conv_bn_0(inputs)
        x = self.conv_bn_1(x)
        return x

class Inception3c_2a(fluid.dygraph.Layer):
    def __init__(self, name_scope,  input_channels, filter_list):
        super(Inception3c_2a, self).__init__(name_scope)
        #1*1 + 3*3 + 3*3
        self.conv_bn_2 = ConvBNLayer(
            self.full_name(),
            num_channels=input_channels,
            num_filters=filter_list[2],
            filter_size=1,
            stride = 1,
            act='relu')

        #新分支：注意将此送到3d网络中去
        self.conv_bn_3 = ConvBNLayer(
            self.full_name(),
            num_channels=filter_list[2],
            num_filters=filter_list[3],
            filter_size=3,
            stride = 1,
            act='relu') #注意padding = 1， 也就是(3-1)//2

    def forward(self, inputs):
        x = self.conv_bn_2(inputs)
        x = self.conv_bn_3(x)
        return x

class Inception4e(fluid.dygraph.Layer):
    def __init__(self, name_scope,  input_channels, filter_list):
        super(Inception4e, self).__init__(name_scope)
        #1*1 + 3*3
        self.conv_bn_0 = ConvBNLayer(
            self.full_name(),
            num_channels=input_channels,
            num_filters=filter_list[0],
            filter_size=1,
            stride = 1,
            act='relu')

        self.conv_bn_1 = ConvBNLayer(
            self.full_name(),
            num_channels=filter_list[0],
            num_filters=filter_list[1],
            filter_size=3,
            stride = 2,
            act='relu') #stride = 2

        #1*1 + 3*3 + 3*3
        self.conv_bn_2 = ConvBNLayer(
            self.full_name(),
            num_channels=input_channels,
            num_filters=filter_list[2],
            filter_size=1,
            stride = 1,
            act='relu')

        #新分支：注意将此送到3d网络中去
        self.conv_bn_3 = ConvBNLayer(
            self.full_name(),
            num_channels=filter_list[2],
            num_filters=filter_list[3],
            filter_size=3,
            stride = 1,
            act='relu') #注意padding = 1， 也就是(3-1)//2

        self.conv_bn_4 = ConvBNLayer(
            self.full_name(),
            num_channels=filter_list[3],
            num_filters=filter_list[4],
            filter_size=3,
            stride = 2,
            act='relu') #注意padding = 1， 也就是(3-1)//2, stride= 2

        #maxpool, 4e的maxpool后没有bn+relu
        self.pool = Pool2D(pool_size=3, pool_stride=2, pool_type="max", ceil_mode=True)

    def forward(self, inputs):
        b1 = self.conv_bn_0(inputs)
        b1 = self.conv_bn_1(b1)

        b2 = self.conv_bn_2(inputs)
        b2 = self.conv_bn_3(b2)
        b2 = self.conv_bn_4(b2)

        b3 = self.pool(inputs)

        concat = fluid.layers.concat([b1, b2, b3], axis=1)
        return concat


#主干网络搭建
'''
InceptionBasic(self.full_name(), 192, [64, 64, 64, 64, 96, 96, 32])
InceptionBasic(self.full_name(), 256, [64, 64, 96, 64, 96, 96, 64])
Inception3c(self.full_name(), 320, [128, 160, 64, 96])

InceptionBasic(self.full_name(), 576, [224, 64, 96, 96, 128, 128, 128])
InceptionBasic(self.full_name(), 576, [192, 96, 128, 96, 128, 128, 128])
InceptionBasic(self.full_name(), 576, [160, 128, 160, 128, 160, 160, 96])
InceptionBasic(self.full_name(), 576, [96, 128, 192, 160, 192, 192, 96])
Inception4e(self.full_name(), 576, [128, 192, 192, 256, 256])
InceptionBasic(self.full_name(), 1024, [352, 192, 320, 160, 224, 224, 128])
InceptionBasic(self.full_name(), 1024, [352, 192, 320, 192, 224, 224, 128], 'max')
'''

#3d网络
class ECO3dNet(fluid.dygraph.Layer):
    def __init__(self, name_scope):
        super(ECO3dNet, self).__init__(name_scope)
        self.res3a_2 = ConvBNLayer(
            self.full_name(),
            num_channels=96,
            num_filters=128,
            filter_size=3,
            stride = 1,
            act='relu', 
            is_3d=True)

        self.res3b_1 = ConvBNLayer(
            self.full_name(),
            num_channels=128,
            num_filters=128,
            filter_size=3,
            stride = 1,
            act='relu', 
            is_3d=True)

        self.res3b_2 = ConvBNLayer(
            self.full_name(),
            num_channels=128,
            num_filters=128,
            filter_size=3,
            stride = 1,
            act='relu', 
            is_3d=True)

        self.res3b_2 = ConvBNLayer(
            self.full_name(),
            num_channels=128,
            num_filters=128,
            filter_size=3,
            stride = 1,
            act='relu', 
            is_3d=True)

        self.res4a_1 = ConvBNLayer(
            self.full_name(),
            num_channels=128,
            num_filters=256,
            filter_size=3,
            stride = 2,
            act='relu', 
            is_3d=True)

        self.res4a_2 = Conv3D(
                num_channels=256,
                num_filters=256,
                filter_size=3,
                stride=1,
                padding=1,
                act=None,
                bias_attr=False)

        self.res4a_down = Conv3D(
                num_channels=128,
                num_filters=256,
                filter_size=3,
                stride=2,
                padding=1,
                act=None,
                bias_attr=False)

        self.res_4a_bn = BatchNorm(256, act="relu")

        self.res4b_1 = ConvBNLayer(
            self.full_name(),
            num_channels=256,
            num_filters=256,
            filter_size=3,
            stride = 1,
            act='relu', 
            is_3d=True)

        self.res4b_2 = Conv3D(
                num_channels=256,
                num_filters=256,
                filter_size=3,
                stride=1,
                padding=1,
                act=None,
                bias_attr=False)

        self.res_4b_bn = BatchNorm(256, act="relu")

        self.res5a_1 = ConvBNLayer(
            self.full_name(),
            num_channels=256,
            num_filters=512,
            filter_size=3,
            stride = 2,
            act='relu', 
            is_3d=True)

        self.res5a_2 = Conv3D(
                num_channels=512,
                num_filters=512,
                filter_size=3,
                stride=1,
                padding=1,
                act=None,
                bias_attr=False)

        self.res5a_down = Conv3D(
                num_channels=256,
                num_filters=512,
                filter_size=3,
                stride=2,
                padding=1,
                act=None,
                bias_attr=False)

        self.res5a_bn = BatchNorm(512, act="relu")

        self.res5b_1 = ConvBNLayer(
            self.full_name(),
            num_channels=512,
            num_filters=512,
            filter_size=3,
            stride = 1,
            act='relu', 
            is_3d=True)

        self.res5b_2 = Conv3D(
                num_channels=512,
                num_filters=512,
                filter_size=3,
                stride=1,
                padding=1,
                act=None,
                bias_attr=False)

        self.res5b_bn = BatchNorm(512, act="relu")

    def forward(self, inputs):
        res3a_2 = self.res3a_2(inputs)
        res3b_1 = self.res3b_1(res3a_2)
        res3b_2 = self.res3b_2(res3b_1)

        res4a_1 = self.res4a_1(res3b_2)
        res4a_2 = self.res4a_2(res4a_1)
        res4a_down = self.res4a_down(res3b_2)

        #y+z
        add_a = fluid.layers.elementwise_add(res4a_2, res4a_down)
        res_4a_bn = self.res_4a_bn(add_a)
        res4b_1 = self.res4b_1(res_4a_bn)
        res4b_2 = self.res4b_2(res4b_1)

        #res4b_2 + add_a
        add_b = fluid.layers.elementwise_add(res4b_2, add_a)
        res_4b_bn = self.res_4b_bn(add_b) 
        res5a_1 = self.res5a_1(res_4b_bn)
        res5a_2 = self.res5a_2(res5a_1)
        res5a_down = self.res5a_down(res_4b_bn)
        res5a = fluid.layers.elementwise_add(res5a_2, res5a_down)

        res5a_bn = self.res5a_bn(res5a)
        res5b_1 = self.res5b_1(res5a_bn)
        res5b_2 = self.res5b_2(res5b_1)
        res5b = fluid.layers.elementwise_add(res5b_2, res5a)
        res5b_bn = self.res5b_bn(res5b)
        return res5b_bn

#主模型结构, 输入1*32*3*224*224
class ECOFull(fluid.dygraph.Layer):
    def __init__(self, name_scope, num_segments=32, class_dim=101):
        super(ECOFull, self).__init__(name_scope)
        self.num_segments = num_segments
        self.class_dim = class_dim

        self.conv1_x = ConvBNLayer(
            self.full_name(),
            num_channels=3,
            num_filters=64,
            filter_size=7,
            stride = 2,
            act='relu')
        
        #这里一定要加上ceil_mode
        self.pool1 = Pool2D(pool_size=3, pool_stride=2, pool_type='max', ceil_mode=True)

        #不清楚这个是否有用，待debug
        self.conv2_reduce = ConvBNLayer(
            self.full_name(),
            num_channels=64,
            num_filters=64,
            filter_size=1,
            stride = 1,
            act=None)

        self.conv2_x = ConvBNLayer(
            self.full_name(),
            num_channels=64,
            num_filters=192,
            filter_size=3,
            stride = 1,
            act='relu')
        self.pool2 = Pool2D(pool_size=3, pool_stride=2, pool_type='max', ceil_mode=True)

        # print("inception_3a")
        self.inception_3a = InceptionBasic(self.full_name(), 192, [64, 64, 64, 64, 96, 96, 32])
        # print("inception_3b")
        self.inception_3b = InceptionBasic(self.full_name(), 256, [64, 64, 96, 64, 96, 96, 64])
        
        #第3个inception
        inception_3c_filterlist = [128, 160, 64, 96, 96]
        self.inception_3c_1 = Inception3c_1(self.full_name(), 320, inception_3c_filterlist)
        self.inception_3c_2a = Inception3c_2a(self.full_name(), 320, inception_3c_filterlist)

        #3Net
        self.eco_3dnet = ECO3dNet(self.full_name())

        #2dNets
        self.inception_3c_2b = ConvBNLayer(
            self.full_name(),
            num_channels=inception_3c_filterlist[3],
            num_filters=inception_3c_filterlist[4],
            filter_size=3,
            stride = 2,
            act='relu')
        self.inception_3c_pool = Pool2D(pool_size=3, pool_stride=2, pool_type="max", ceil_mode=True) #没有convbn

        # print("inception_4a")
        self.inception_4a = InceptionBasic(self.full_name(), 576, [224, 64, 96, 96, 128, 128, 128])
        # print("inception_4b")
        self.inception_4b = InceptionBasic(self.full_name(), 576, [192, 96, 128, 96, 128, 128, 128])
        # print("inception_4c")
        self.inception_4c = InceptionBasic(self.full_name(), 576, [160, 128, 160, 128, 160, 160, 96])
        self.inception_4d = InceptionBasic(self.full_name(), 576, [96, 128, 192, 160, 192, 192, 96])
        self.inception_4e = Inception4e(self.full_name(), 576, [128, 192, 192, 256, 256]) #没有convbn
        self.inception_5a = InceptionBasic(self.full_name(), 1024, [352, 192, 320, 160, 224, 224, 128])
        # print("inception_5b")
        self.inception_5b = InceptionBasic(self.full_name(), 1024, [352, 192, 320, 192, 224, 224, 128], 'max')

        self.end_pool1 =  Pool2D(pool_size=7, pool_stride=1, pool_type="avg", ceil_mode=True)
        self.drop_out1 = Dropout(p=0.5)
        self.drop_out2 = Dropout(p=0.3)
        self.out = Linear(input_dim=1536, output_dim=self.class_dim, act='softmax')

    def forward(self, inputs, label=None):
        inputs = fluid.layers.reshape(inputs, [-1, inputs.shape[2], inputs.shape[3], inputs.shape[4]])
        # print("xshape", inputs.shape)
        x = self.conv1_x(inputs)
        # print("xshape", x.shape)
        x = self.pool1(x)
        # print("xshape", x.shape)
        x = self.conv2_reduce(x)
        x = self.conv2_x(x)
        # print("xshape", x.shape)
        x = self.pool2(x)
        # print("xshape", x.shape)
        x = self.inception_3a(x)
        x = self.inception_3b(x)

        inc_3c_1 = self.inception_3c_1(x)
        inc_3c_2a = self.inception_3c_2a(x)
        # print("****", inc_3c_2a.shape)

        #分支3dnet,
        #先reshape，再transpose
        inc_3c_2a_reshape = fluid.layers.reshape(
            x=inc_3c_2a,
            shape=[-1, self.num_segments, inc_3c_2a.shape[1], inc_3c_2a.shape[2], inc_3c_2a.shape[3]])#tchw ==> 1*tchw
        inc_3c_2a_perm = fluid.layers.transpose(inc_3c_2a_reshape, perm=[0, 2, 1, 3, 4])
        
        eco3dNet = self.eco_3dnet(inc_3c_2a_perm) 

        # print("---------->", self.num_segments, eco3dNet.shape)
        global_pool3D = fluid.layers.pool3d(
              input=eco3dNet,
              pool_size=[int(self.num_segments / 4), 7, 7],
              pool_type='avg',
              pool_stride=[1,1,1],
              ceil_mode=True)
        global_pool3D = self.drop_out2(global_pool3D)

        #分支2dNets
        inception_3c_2b = self.inception_3c_2b(inc_3c_2a)
        inception_3c_pool = self.inception_3c_pool(x)
        inception = fluid.layers.concat([inc_3c_1, inception_3c_2b, inception_3c_pool], axis=1)
        inception = self.inception_4a(inception)
        inception = self.inception_4b(inception)
        inception = self.inception_4c(inception)
        inception = self.inception_4d(inception)
        inception = self.inception_4e(inception) 
        inception = self.inception_5a(inception)
        inception = self.inception_5b(inception)
        
        # print("ck1", inception.shape)
        global_pool2D_pre = self.end_pool1(inception)
        # print("ck2", global_pool2D_pre.shape)
        global_pool2D_pre_drop = self.drop_out1(global_pool2D_pre)
        # print("ck3", global_pool2D_pre_drop.shape)

        #可能需要一个transpose
        tmp_shape = global_pool2D_pre_drop.shape
        global_pool2D_pre_drop_reshape = fluid.layers.reshape(
            x=global_pool2D_pre_drop,
            shape=[-1, self.num_segments, tmp_shape[1], tmp_shape[2], tmp_shape[3]])
        # print("aaaa===>", global_pool2D_pre_drop_reshape.shape)
        global_pool2D_pre_drop_perm = fluid.layers.transpose(global_pool2D_pre_drop_reshape, perm=[0, 2, 1, 3, 4])
        # print("bbbb===>", global_pool2D_pre_drop_perm.shape)

        global_pool2D_reshape_consensus = fluid.layers.pool3d(
              input=global_pool2D_pre_drop_perm,
              pool_size=[int(self.num_segments), 1, 1],
              pool_type='avg',
              pool_stride=1,
              ceil_mode=True)
        # print("cccc===>", global_pool2D_reshape_consensus.shape)

        #3d和2dnets融合
        # print("===>", global_pool2D_reshape_consensus.shape, global_pool3D.shape)
        global_pool = fluid.layers.concat(input=[global_pool2D_reshape_consensus, global_pool3D], axis=1)
        global_pool_reshape = fluid.layers.reshape(x=global_pool, shape=[-1, 1536])
        y = self.out(global_pool_reshape)

        if label is not None:
            acc = fluid.layers.accuracy(input=y, label=label)
            return y, acc
        else:
            return y

if __name__ == '__main__':
    with fluid.dygraph.guard():
        network = ECOFull("eco_full", 16, 101)
        img = np.zeros([1, 16, 3, 224, 224]).astype('float32')
        img = fluid.dygraph.to_variable(img)
        outs = network(img).numpy()
        print(outs)


      
