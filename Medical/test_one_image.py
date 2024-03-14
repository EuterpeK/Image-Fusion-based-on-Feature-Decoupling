# # # -*- coding: utf-8 -*-
# # import tensorflow as tf
# # import numpy as np
# # import scipy.misc
# # import time
# # import os
# # import glob
# # import cv2

# # #reader = tf.train.NewCheckpointReader("./checkpoint/CGAN_120/CGAN.model-9")


# # def imread(path, is_grayscale=True):
# #   """
# #   Read image using its path.
# #   Default value is gray-scale, and image is read by YCbCr format as the paper said.
# #   """
# #   if is_grayscale:
# #     #flatten=True 以灰度图的形式读�?
# #     return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
# #   else:
# #     return scipy.misc.imread(path, mode='YCbCr').astype(np.float)

# # def imsave(image, path):
# #   return scipy.misc.imsave(path, image)
  
  
# # def prepare_data(dataset):
# #     data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)))
# #     data = glob.glob(os.path.join(data_dir, "*.jpg"))
# #     data.extend(glob.glob(os.path.join(data_dir, "*.bmp")))
# #     data.sort(key=lambda x:int(x[len(data_dir)+1:-4]))
# #     return data

# # def lrelu(x, leak=0.2):
# #     return tf.maximum(x, leak * x)

# # def fusion_model(img_ir,img_vi):
# #     with tf.compat.v1.variable_scope('fusion_model'):
    
# # ####################  Layer1  ###########################
# #         with tf.compat.v1.variable_scope('layer1'):
# #             weights=tf.compat.v1.get_variable("w1",initializer=tf.constant(reader.get_tensor('fusion_model/layer1/w1')))
# #             bias=tf.compat.v1.get_variable("b1",initializer=tf.constant(reader.get_tensor('fusion_model/layer1/b1')))
# #             conv1_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(img_ir, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
# #             conv1_ir = lrelu(conv1_ir)
# #         with tf.compat.v1.variable_scope('layer1_vi'):
# #             weights=tf.compat.v1.get_variable("w1_vi",initializer=tf.constant(reader.get_tensor('fusion_model/layer1_vi/w1_vi')))
# #             bias=tf.compat.v1.get_variable("b1_vi",initializer=tf.constant(reader.get_tensor('fusion_model/layer1_vi/b1_vi')))
# #             conv1_vi= tf.contrib.layers.batch_norm(tf.nn.conv2d(img_vi, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
# #             conv1_vi = lrelu(conv1_vi)    
                    



            
# # ####################  Layer2  ###########################           
            
                      
# #         with tf.compat.v1.variable_scope('layer2'):
# #             weights=tf.compat.v1.get_variable("w2",initializer=tf.constant(reader.get_tensor('fusion_model/layer2/w2')))
# #             bias=tf.compat.v1.get_variable("b2",initializer=tf.constant(reader.get_tensor('fusion_model/layer2/b2')))
# #             conv2_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv1_ir, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
# #             conv2_ir = lrelu(conv2_ir)
# #         with tf.compat.v1.variable_scope('layer2_vi'):
# #             weights=tf.compat.v1.get_variable("w2_vi",initializer=tf.constant(reader.get_tensor('fusion_model/layer2_vi/w2_vi')))
# #             bias=tf.compat.v1.get_variable("b2_vi",initializer=tf.constant(reader.get_tensor('fusion_model/layer2_vi/b2_vi')))
# #             conv2_vi= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv1_vi, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
# #             conv2_vi = lrelu(conv2_vi)   
            
# #         conv_2_midle =tf.concat([conv2_ir,conv2_vi],axis=-1)      
        
# #         with tf.compat.v1.variable_scope('layer2_3'):
# #             weights=tf.compat.v1.get_variable("w2_3",initializer=tf.constant(reader.get_tensor('fusion_model/layer2_3/w2_3')))
# #             bias=tf.compat.v1.get_variable("b2_3",initializer=tf.constant(reader.get_tensor('fusion_model/layer2_3/b2_3')))
# #             conv2_3_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv_2_midle, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
# #             conv2_3_ir = lrelu(conv2_3_ir)
# #         with tf.compat.v1.variable_scope('layer2_3_vi'):
# #             weights=tf.compat.v1.get_variable("w2_3_vi",initializer=tf.constant(reader.get_tensor('fusion_model/layer2_3_vi/w2_3_vi')))
# #             bias=tf.compat.v1.get_variable("b2_3_vi",initializer=tf.constant(reader.get_tensor('fusion_model/layer2_3_vi/b2_3_vi')))
# #             conv2_3_vi= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv_2_midle, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
# #             conv2_3_vi = lrelu(conv2_3_vi)               
            
            
                     
# # ####################  Layer3  ###########################                 
# #         conv_12_ir=tf.concat([conv1_ir,conv2_ir,conv2_3_ir],axis=-1)
# #         conv_12_vi=tf.concat([conv1_vi,conv2_vi,conv2_3_vi],axis=-1)                   
         
# #         with tf.compat.v1.variable_scope('layer3'):
# #             weights=tf.compat.v1.get_variable("w3",initializer=tf.constant(reader.get_tensor('fusion_model/layer3/w3')))
# #             bias=tf.compat.v1.get_variable("b3",initializer=tf.constant(reader.get_tensor('fusion_model/layer3/b3')))
# #             conv3_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv_12_ir, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
# #             conv3_ir = lrelu(conv3_ir)            
# #         with tf.compat.v1.variable_scope('layer3_vi'):
# #             weights=tf.compat.v1.get_variable("w3_vi",initializer=tf.constant(reader.get_tensor('fusion_model/layer3_vi/w3_vi')))
# #             bias=tf.compat.v1.get_variable("b3_vi",initializer=tf.constant(reader.get_tensor('fusion_model/layer3_vi/b3_vi')))
# #             conv3_vi= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv_12_vi, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
# #             conv3_vi =lrelu(conv3_vi)            

# #         conv_3_midle =tf.concat([conv3_ir,conv3_vi],axis=-1)    
        
# #         with tf.compat.v1.variable_scope('layer3_4'):
# #             weights=tf.compat.v1.get_variable("w3_4",initializer=tf.constant(reader.get_tensor('fusion_model/layer3_4/w3_4')))
# #             bias=tf.compat.v1.get_variable("b3_4",initializer=tf.constant(reader.get_tensor('fusion_model/layer3_4/b3_4')))
# #             conv3_4_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv_3_midle, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
# #             conv3_4_ir = lrelu(conv3_4_ir)
# #         with tf.compat.v1.variable_scope('layer3_4_vi'):
# #             weights=tf.compat.v1.get_variable("w3_4_vi",initializer=tf.constant(reader.get_tensor('fusion_model/layer3_4_vi/w3_4_vi')))
# #             bias=tf.compat.v1.get_variable("b3_4_vi",initializer=tf.constant(reader.get_tensor('fusion_model/layer3_4_vi/b3_4_vi')))
# #             conv3_4_vi= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv_3_midle, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
# #             conv3_4_vi = lrelu(conv3_4_vi)  



# # ####################  Layer4  ###########################                 
# #         conv_123_ir=tf.concat([conv1_ir,conv2_ir,conv3_ir,conv3_4_ir],axis=-1)
# #         conv_123_vi=tf.concat([conv1_vi,conv2_vi,conv3_vi,conv3_4_vi],axis=-1)               
            
          
# #         with tf.compat.v1.variable_scope('layer4'):
# #             weights=tf.compat.v1.get_variable("w4",initializer=tf.constant(reader.get_tensor('fusion_model/layer4/w4')))
# #             bias=tf.compat.v1.get_variable("b4",initializer=tf.constant(reader.get_tensor('fusion_model/layer4/b4')))
# #             conv4_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv_123_ir, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
# #             conv4_ir = lrelu(conv4_ir)
            
# #         with tf.compat.v1.variable_scope('layer4_vi'):
# #             weights=tf.compat.v1.get_variable("w4_vi",initializer=tf.constant(reader.get_tensor('fusion_model/layer4_vi/w4_vi')))
# #             bias=tf.compat.v1.get_variable("b4_vi",initializer=tf.constant(reader.get_tensor('fusion_model/layer4_vi/b4_vi')))
# #             conv4_vi= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv_123_vi, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
# #             conv4_vi = lrelu(conv4_vi)            
            
# #         conv_ir_vi =tf.concat([conv1_ir,conv1_vi,conv2_ir,conv2_vi,conv3_ir,conv3_vi,conv4_ir,conv4_vi],axis=-1)
        
        
           
# # ####################  Layer5  ###########################                          
# #         with tf.compat.v1.variable_scope('layer5'):
# #             weights=tf.compat.v1.get_variable("w5",initializer=tf.constant(reader.get_tensor('fusion_model/layer5/w5')))
# #             bias=tf.compat.v1.get_variable("b5",initializer=tf.constant(reader.get_tensor('fusion_model/layer5/b5')))
# #             conv5_ir= tf.nn.conv2d(conv_ir_vi, weights, strides=[1,1,1,1], padding='SAME') + bias
# #             conv5_ir=tf.nn.tanh(conv5_ir)
# #     return conv5_ir




# # def input_setup(index):
# #     padding=0
# #     sub_ir_sequence = []
# #     sub_vi_sequence = []
# #     input_ir=(imread(data_ir[index])-127.5)/127.5
# #     input_ir=np.lib.pad(input_ir,((padding,padding),(padding,padding)),'edge')
# #     w,h=input_ir.shape
# #     input_ir=input_ir.reshape([w,h,1])
# #     input_vi=(imread(data_vi[index])-127.5)/127.5
# #     input_vi=np.lib.pad(input_vi,((padding,padding),(padding,padding)),'edge')
# #     w,h=input_vi.shape
# #     input_vi=input_vi.reshape([w,h,1])
# #     sub_ir_sequence.append(input_ir)
# #     sub_vi_sequence.append(input_vi)
# #     train_data_ir= np.asarray(sub_ir_sequence)
# #     train_data_vi= np.asarray(sub_vi_sequence)
# #     return train_data_ir,train_data_vi

# # for idx_num in range(35):
# #   num_epoch=29
# #   while(num_epoch==idx_num):

# #       reader = tf.train.NewCheckpointReader('./checkpoint/CGAN_120/CGAN.model-'+ str(num_epoch))

# #       with tf.name_scope('IR_input'):
# #           #红外图像patch
# #           images_ir = tf.placeholder(tf.float32, [1,None,None,None], name='images_ir')
# #       with tf.name_scope('VI_input'):
# #           #可见光图像patch
# #           images_vi = tf.placeholder(tf.float32, [1,None,None,None], name='images_vi')
# #           #self.labels_vi_gradient=gradient(self.labels_vi)
# #       #将红外和可见光图像在通道方向连起来，第一通道是红外图像，第二通道是可见光图像
# #       with tf.name_scope('input'):
# #           #resize_ir=tf.image.resize_images(images_ir, (512, 512), method=2)
# #           input_image_ir =tf.concat([images_ir,images_ir,images_vi],axis=-1)
# #           input_image_vi =tf.concat([images_vi,images_vi,images_ir],axis=-1)

# #       with tf.name_scope('fusion'):
# #           fusion_image=fusion_model(input_image_ir,input_image_vi)


# #       with tf.Session() as sess:
# #           init_op=tf.global_variables_initializer()
# #           sess.run(init_op)
# #           data_ir=prepare_data('Test_ir')
# #           data_vi=prepare_data('Test_vi')
# #           for i in range(len(data_ir)):
# #               start=time.time()
# #               train_data_ir,train_data_vi=input_setup(i)
# #               result =sess.run(fusion_image,feed_dict={images_ir: train_data_ir,images_vi: train_data_vi})
# #               result=result*127.5+127.5
# #               result = result.squeeze()
# #               image_path = os.path.join(os.getcwd(), 'result','epoch'+str(num_epoch))
# #               if not os.path.exists(image_path):
# #                   os.makedirs(image_path)
# #               if i<=9:
# #                   image_path = os.path.join(image_path,'F9_0'+str(i)+".bmp")
# #               else:
# #                   image_path = os.path.join(image_path,'F9_'+str(i)+".bmp")
# #               end=time.time()
# #               # print(out.shape)
# #               imsave(result, image_path)
# #               print("Testing [%d] success,Testing time is [%f]"%(i,end-start))
# #       tf.reset_default_graph()
# #       num_epoch=num_epoch+1


# -*- coding: utf-8 -*-
from pprint import pprint
import tensorflow as tf
import numpy as np
import scipy.misc
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import glob
import cv2

#reader = tf.compat.v1.train.NewCheckpointReader("./checkpoint/CGAN_120/CGAN.model-9")


def imread(path, is_grayscale=True):
  """
  Read image using its path.
  Default value is gray-scale, and image is read by YCbCr format as the paper said.
  """
  if is_grayscale:
    #flatten=True 以灰度图的形式读�?
    return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
  else:
    return scipy.misc.imread(path, mode='YCbCr').astype(np.float)

def imsave(image, path):
  return scipy.misc.imsave(path, image)
  
  
def prepare_data(dataset):
    data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)))
    print("*********",data_dir)
    data = glob.glob(os.path.join(data_dir, "*.jpg"))
    data.extend(glob.glob(os.path.join(data_dir, "*.bmp")))
    data.sort(key=lambda x:int(x[len(data_dir)+1:-4]))
    return data

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)

def ir_encoder(img_ir):
        # with tf.compat.v1.variable_scope('ir_encoder'):
    with tf.compat.v1.variable_scope('ir_block1'):
      with tf.compat.v1.variable_scope('ir_block1_stride2'):
        weighs = tf.compat.v1.get_variable('block1_w1_ir', initializer=tf.constant(reader.get_tensor('ir_block1/ir_block1_stride2/block1_w1_ir')))
        bias = tf.compat.v1.get_variable('block1_b1_ir',initializer=tf.constant(reader.get_tensor('ir_block1/ir_block1_stride2/block1_b1_ir')))
        block1_stride = tf.contrib.layers.batch_norm(tf.nn.conv2d(img_ir,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block1_stride = lrelu(block1_stride)
      with tf.compat.v1.variable_scope('ir_block1_layer2'):
        weighs = tf.compat.v1.get_variable('block1_w2_ir', initializer=tf.constant(reader.get_tensor('ir_block1/ir_block1_layer2/block1_w2_ir')))
        bias = tf.compat.v1.get_variable('block2_b2_ir',initializer=tf.constant(reader.get_tensor('ir_block1/ir_block1_layer2/block2_b2_ir')))
        block1_conv2 = tf.contrib.layers.batch_norm(tf.nn.conv2d(block1_stride,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block1_conv2 = lrelu(block1_conv2)
      with tf.compat.v1.variable_scope('ir_block1_layer3'):
        weighs = tf.compat.v1.get_variable('block1_w3_ir', initializer=tf.constant(reader.get_tensor('ir_block1/ir_block1_layer3/block1_w3_ir')))
        bias = tf.compat.v1.get_variable('block1_b3_ir', initializer=tf.constant(reader.get_tensor('ir_block1/ir_block1_layer3/block1_b3_ir')))
        block1_conv3 = tf.contrib.layers.batch_norm(tf.nn.conv2d(block1_conv2,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block1_conv3 = lrelu(block1_conv3)

    with tf.compat.v1.variable_scope('ir_block2'):
      with tf.compat.v1.variable_scope('ir_block2_stride2'):
        weighs = tf.compat.v1.get_variable('block2_w1_ir', initializer=tf.constant(reader.get_tensor('ir_block2/ir_block2_stride2/block2_w1_ir')))
        bias = tf.compat.v1.get_variable('block2_b1_ir', initializer=tf.constant(reader.get_tensor('ir_block2/ir_block2_stride2/block2_b1_ir')))
        block2_stride = tf.contrib.layers.batch_norm(tf.nn.conv2d(block1_conv3,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block2_stride = lrelu(block2_stride)
      with tf.compat.v1.variable_scope('ir_block2_layer2'):
        weighs = tf.compat.v1.get_variable('block2_w2_ir', initializer=tf.constant(reader.get_tensor('ir_block2/ir_block2_layer2/block2_w2_ir')))
        bias = tf.compat.v1.get_variable('block2_b2_ir',initializer=tf.constant(reader.get_tensor('ir_block2/ir_block2_layer2/block2_b2_ir')))
        block2_conv2 = tf.contrib.layers.batch_norm(tf.nn.conv2d(block2_stride,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block2_conv2 = lrelu(block2_conv2)
      with tf.compat.v1.variable_scope('ir_block2_layer3'):
        weighs = tf.compat.v1.get_variable('block2_w3_ir', initializer=tf.constant(reader.get_tensor('ir_block2/ir_block2_layer3/block2_w3_ir')))
        bias = tf.compat.v1.get_variable('block2_b3_ir',initializer=tf.constant(reader.get_tensor('ir_block2/ir_block2_layer3/block2_b3_ir')))
        block2_conv3 = tf.contrib.layers.batch_norm(tf.nn.conv2d(block2_conv2,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block2_conv3 = lrelu(block2_conv3)
      with tf.compat.v1.variable_scope('ir_block2_layer4'):
        weighs = tf.compat.v1.get_variable('block2_w4_ir', initializer=tf.constant(reader.get_tensor('ir_block2/ir_block2_layer4/block2_w4_ir')))
        bias = tf.compat.v1.get_variable('block2_b4_ir',initializer=tf.constant(reader.get_tensor('ir_block2/ir_block2_layer4/block2_b4_ir')))
        block2_conv4 = tf.contrib.layers.batch_norm(tf.nn.conv2d(block2_conv3,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block2_conv4 = lrelu(block2_conv4)
      with tf.compat.v1.variable_scope('ir_block2_layer5'):
        weighs = tf.compat.v1.get_variable('block2_w5_ir', initializer=tf.constant(reader.get_tensor('ir_block2/ir_block2_layer5/block2_w5_ir')))
        bias = tf.compat.v1.get_variable('block2_b5_ir',initializer=tf.constant(reader.get_tensor('ir_block2/ir_block2_layer5/block2_b5_ir')))
        block2_conv5 = tf.contrib.layers.batch_norm(tf.nn.conv2d(block2_conv4,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block2_conv5 = lrelu(block2_conv5)
  
    with tf.compat.v1.variable_scope('ir_block3'):
      with tf.compat.v1.variable_scope('ir_block3_stride2'):
        weighs = tf.compat.v1.get_variable('block3_w1_ir', initializer=tf.constant(reader.get_tensor('ir_block3/ir_block3_stride2/block3_w1_ir')))
        bias = tf.compat.v1.get_variable('block3_b1_ir', initializer=tf.constant(reader.get_tensor('ir_block3/ir_block3_stride2/block3_b1_ir')))
        block3_stride = tf.contrib.layers.batch_norm(tf.nn.conv2d(block2_conv5,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block3_stride = lrelu(block3_stride)
      with tf.compat.v1.variable_scope('ir_block3_layer2'):
        weighs = tf.compat.v1.get_variable('block3_w2_ir', initializer=tf.constant(reader.get_tensor('ir_block3/ir_block3_layer2/block3_w2_ir')))
        bias = tf.compat.v1.get_variable('block3_b2_ir', initializer=tf.constant(reader.get_tensor('ir_block3/ir_block3_layer2/block3_b2_ir')))
        block3_conv2 = tf.contrib.layers.batch_norm(tf.nn.conv2d(block3_stride,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block3_conv2 = lrelu(block3_conv2)
      with tf.compat.v1.variable_scope('ir_block3_layer3'):
        weighs = tf.compat.v1.get_variable('block3_w3_ir', initializer=tf.constant(reader.get_tensor('ir_block3/ir_block3_layer3/block3_w3_ir')))
        bias = tf.compat.v1.get_variable('block3_b3_ir',initializer=tf.constant(reader.get_tensor('ir_block3/ir_block3_layer3/block3_b3_ir')))
        block3_conv3 = tf.contrib.layers.batch_norm(tf.nn.conv2d(block3_conv2,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block3_conv3 = lrelu(block3_conv3)
      with tf.compat.v1.variable_scope('ir_block3_layer4'):
        weighs = tf.compat.v1.get_variable('block3_w4_ir', initializer=tf.constant(reader.get_tensor('ir_block3/ir_block3_layer4/block3_w4_ir')))
        bias = tf.compat.v1.get_variable('b4_ir',initializer=tf.constant(reader.get_tensor('ir_block3/ir_block3_layer4/b4_ir')))
        block3_conv4 = tf.contrib.layers.batch_norm(tf.nn.conv2d(block3_conv3,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block3_conv4 = lrelu(block3_conv4)
      with tf.compat.v1.variable_scope('ir_block3_layer5'):
        weighs = tf.compat.v1.get_variable('block3_w5_ir', initializer=tf.constant(reader.get_tensor('ir_block3/ir_block3_layer5/block3_w5_ir')))
        bias = tf.compat.v1.get_variable('block3_b5_ir',initializer=tf.constant(reader.get_tensor('ir_block3/ir_block3_layer5/block3_b5_ir')))
        block3_conv5 = tf.contrib.layers.batch_norm(tf.nn.conv2d(block3_conv4,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block3_conv5 = lrelu(block3_conv5)
      with tf.compat.v1.variable_scope('ir_block3_layer6'):
        weighs = tf.compat.v1.get_variable('block3_w6_ir', initializer=tf.constant(reader.get_tensor('ir_block3/ir_block3_layer6/block3_w6_ir')))
        bias = tf.compat.v1.get_variable('block3_b6_ir',initializer=tf.constant(reader.get_tensor('ir_block3/ir_block3_layer6/block3_b6_ir')))
        block3_conv6 = tf.contrib.layers.batch_norm(tf.nn.conv2d(block3_conv5,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block3_conv6 = lrelu(block3_conv6)
      with tf.compat.v1.variable_scope('ir_block3_layer7'):
        weighs = tf.compat.v1.get_variable('block3_w7_ir', initializer=tf.constant(reader.get_tensor('ir_block3/ir_block3_layer7/block3_w7_ir')))
        bias = tf.compat.v1.get_variable('block3_b7_ir',initializer=tf.constant(reader.get_tensor('ir_block3/ir_block3_layer7/block3_b7_ir')))
        block3_conv7 = tf.contrib.layers.batch_norm(tf.nn.conv2d(block3_conv6,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block3_conv7 = lrelu(block3_conv7)
      with tf.compat.v1.variable_scope('ir_block3_layer8'):
        weighs = tf.compat.v1.get_variable('block3_w8_ir',initializer=tf.constant(reader.get_tensor('ir_block3/ir_block3_layer8/block3_w8_ir')))
        bias = tf.compat.v1.get_variable('block3_b8_ir',initializer=tf.constant(reader.get_tensor('ir_block3/ir_block3_layer8/block3_b8_ir')))
        block3_conv8 = tf.contrib.layers.batch_norm(tf.nn.conv2d(block3_conv7,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block3_conv8 = lrelu(block3_conv8)
      with tf.compat.v1.variable_scope('ir_block3_layer9'):
        weighs = tf.compat.v1.get_variable('block3_w9_ir', initializer=tf.constant(reader.get_tensor('ir_block3/ir_block3_layer9/block3_w9_ir')))
        bias = tf.compat.v1.get_variable('block3_b9_ir',initializer=tf.constant(reader.get_tensor('ir_block3/ir_block3_layer9/block3_b9_ir')))
        block3_conv9 = tf.contrib.layers.batch_norm(tf.nn.conv2d(block3_conv8,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block3_conv9 = lrelu(block3_conv9)
    with tf.compat.v1.variable_scope('ir_block4'):
      with tf.compat.v1.variable_scope('ir_block4_stride2'):
        weighs = tf.compat.v1.get_variable('block4_w1_ir', initializer=tf.constant(reader.get_tensor('ir_block4/ir_block4_stride2/block4_w1_ir')))
        bias = tf.compat.v1.get_variable('block4_b1_ir', initializer=tf.constant(reader.get_tensor('ir_block4/ir_block4_stride2/block4_b1_ir')))
        block4_stride = tf.contrib.layers.batch_norm(tf.nn.conv2d(block3_conv9,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block4_stride = lrelu(block4_stride)
      with tf.compat.v1.variable_scope('ir_block4_layer2'):
        weighs = tf.compat.v1.get_variable('block4_w2_ir', initializer=tf.constant(reader.get_tensor('ir_block4/ir_block4_layer2/block4_w2_ir')))
        bias = tf.compat.v1.get_variable('block4_b2_ir', initializer=tf.constant(reader.get_tensor('ir_block4/ir_block4_layer2/block4_b2_ir')))
        block4_conv2 = tf.contrib.layers.batch_norm(tf.nn.conv2d(block4_stride,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block4_conv2 = lrelu(block4_conv2)
      with tf.compat.v1.variable_scope('ir_block4_layer3'):
        weighs = tf.compat.v1.get_variable('block4_w3_ir',  initializer=tf.constant(reader.get_tensor('ir_block4/ir_block4_layer3/block4_w3_ir')))
        bias = tf.compat.v1.get_variable('block4_b3_ir', initializer=tf.constant(reader.get_tensor('ir_block4/ir_block4_layer3/block4_b3_ir')))
        block4_conv3 = tf.contrib.layers.batch_norm(tf.nn.conv2d(block4_conv2,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block4_conv3 = lrelu(block4_conv3)

    with tf.compat.v1.variable_scope('ir_intensity'):
      intensity_concat = tf.concat([block1_conv3, block2_conv5, block3_conv9, block4_conv3],axis=-1)
      with tf.compat.v1.variable_scope('ir_abstract_intensity'):
        weighs = tf.compat.v1.get_variable('ir_layer_intensity_w', initializer=tf.constant(reader.get_tensor('ir_intensity/ir_abstract_intensity/ir_layer_intensity_w')))
        bias = tf.compat.v1.get_variable('ir_layer_intensity_b',initializer=tf.constant(reader.get_tensor('ir_intensity/ir_abstract_intensity/ir_layer_intensity_b')))
        intensity1 = tf.contrib.layers.batch_norm(tf.nn.conv2d(intensity_concat,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        intensity1 = lrelu(intensity1)
      with tf.compat.v1.variable_scope('ir_intensity_single'):
        weighs = tf.compat.v1.get_variable('ir_intensity_w', initializer=tf.constant(reader.get_tensor('ir_intensity/ir_intensity_single/ir_intensity_w')))
        bias = tf.compat.v1.get_variable('ir_intensity_b',initializer=tf.constant(reader.get_tensor('ir_intensity/ir_intensity_single/ir_intensity_b')))
        intensity = tf.contrib.layers.batch_norm(tf.nn.conv2d(intensity1,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        intensity = lrelu(intensity)

    with tf.compat.v1.variable_scope('ir_gradient'):
      gradient_concat = tf.concat([block1_conv3, block2_conv5, block3_conv9, block4_conv3], axis=-1)
      with tf.compat.v1.variable_scope('ir_abstract_gradient'):
        weighs = tf.compat.v1.get_variable('ir_layer_gradient_w', initializer=tf.constant(reader.get_tensor('ir_gradient/ir_abstract_gradient/ir_layer_gradient_w')))
        bias = tf.compat.v1.get_variable('ir_layer_gradient_b',initializer=tf.constant(reader.get_tensor('ir_gradient/ir_abstract_gradient/ir_layer_gradient_b')))
        gradient1 = tf.contrib.layers.batch_norm(tf.nn.conv2d(gradient_concat,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        gradient1 = lrelu(gradient1)
      with tf.compat.v1.variable_scope('ir_gradient_single'):
        weighs = tf.compat.v1.get_variable('ir_gradient_w', initializer=tf.constant(reader.get_tensor('ir_gradient/ir_gradient_single/ir_gradient_w')))
        bias = tf.compat.v1.get_variable('ir_gradient_b', initializer=tf.constant(reader.get_tensor('ir_gradient/ir_gradient_single/ir_gradient_b')))
        gradient = tf.contrib.layers.batch_norm(tf.nn.conv2d(gradient1,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        gradient = lrelu(gradient)
    
    return intensity, gradient

def vi_encoder(img_vi):
    # with tf.compat.v1.variable_scope('vi_encoder'):
    with tf.compat.v1.variable_scope('vi_block1'):
      with tf.compat.v1.variable_scope('vi_block1_stride2'):
        weighs = tf.compat.v1.get_variable('block1_w1_vi', initializer=tf.constant(reader.get_tensor('vi_block1/vi_block1_stride2/block1_w1_vi')))
        bias = tf.compat.v1.get_variable('block1_b1_vi', initializer=tf.constant(reader.get_tensor('vi_block1/vi_block1_stride2/block1_b1_vi')))
        block1_stride = tf.contrib.layers.batch_norm(tf.nn.conv2d(img_vi,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block1_stride = lrelu(block1_stride)
      with tf.compat.v1.variable_scope('vi_block1_layer2'):
        weighs = tf.compat.v1.get_variable('block1_w2_vi', initializer=tf.constant(reader.get_tensor('vi_block1/vi_block1_layer2/block1_w2_vi')))
        bias = tf.compat.v1.get_variable('block2_b2_vi', initializer=tf.constant(reader.get_tensor('vi_block1/vi_block1_layer2/block2_b2_vi')))
        block1_conv2 = tf.contrib.layers.batch_norm(tf.nn.conv2d(block1_stride,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block1_conv2 = lrelu(block1_conv2)
      with tf.compat.v1.variable_scope('vi_block1_layer3'):
        weighs = tf.compat.v1.get_variable('block1_w3_vi', initializer=tf.constant(reader.get_tensor('vi_block1/vi_block1_layer3/block1_w3_vi')))
        bias = tf.compat.v1.get_variable('block1_b3_vi', initializer=tf.constant(reader.get_tensor('vi_block1/vi_block1_layer3/block1_b3_vi')))
        block1_conv3 = tf.contrib.layers.batch_norm(tf.nn.conv2d(block1_conv2,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block1_conv3 = lrelu(block1_conv3)

    with tf.compat.v1.variable_scope('vi_block2'):
      with tf.compat.v1.variable_scope('vi_block2_stride2'):
        weighs = tf.compat.v1.get_variable('block2_w1_vi',  initializer=tf.constant(reader.get_tensor('vi_block2/vi_block2_stride2/block2_w1_vi')))
        bias = tf.compat.v1.get_variable('block2_b1_vi', initializer=tf.constant(reader.get_tensor('vi_block2/vi_block2_stride2/block2_b1_vi')))
        block2_stride = tf.contrib.layers.batch_norm(tf.nn.conv2d(block1_conv3,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block2_stride = lrelu(block2_stride)
      with tf.compat.v1.variable_scope('vi_block2_layer2'):
        weighs = tf.compat.v1.get_variable('block2_w2_vi', initializer=tf.constant(reader.get_tensor('vi_block2/vi_block2_layer2/block2_w2_vi')))
        bias = tf.compat.v1.get_variable('block2_b2_vi', initializer=tf.constant(reader.get_tensor('vi_block2/vi_block2_layer2/block2_b2_vi')))
        block2_conv2 = tf.contrib.layers.batch_norm(tf.nn.conv2d(block2_stride,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block2_conv2 = lrelu(block2_conv2)
      with tf.compat.v1.variable_scope('vi_block2_layer3'):
        weighs = tf.compat.v1.get_variable('block2_w3_vi',  initializer=tf.constant(reader.get_tensor('vi_block2/vi_block2_layer3/block2_w3_vi')))
        bias = tf.compat.v1.get_variable('block2_b3_vi', initializer=tf.constant(reader.get_tensor('vi_block2/vi_block2_layer3/block2_b3_vi')))
        block2_conv3 = tf.contrib.layers.batch_norm(tf.nn.conv2d(block2_conv2,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block2_conv3 = lrelu(block2_conv3)
      with tf.compat.v1.variable_scope('vi_block2_layer4'):
        weighs = tf.compat.v1.get_variable('block2_w4_vi',  initializer=tf.constant(reader.get_tensor('vi_block2/vi_block2_layer4/block2_w4_vi')))
        bias = tf.compat.v1.get_variable('block2_b4_vi', initializer=tf.constant(reader.get_tensor('vi_block2/vi_block2_layer4/block2_b4_vi')))
        block2_conv4 = tf.contrib.layers.batch_norm(tf.nn.conv2d(block2_conv3,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block2_conv4 = lrelu(block2_conv4)
      with tf.compat.v1.variable_scope('vi_block2_layer5'):
        weighs = tf.compat.v1.get_variable('block2_w5_vi',  initializer=tf.constant(reader.get_tensor('vi_block2/vi_block2_layer5/block2_w5_vi')))
        bias = tf.compat.v1.get_variable('block2_b5_vi', initializer=tf.constant(reader.get_tensor('vi_block2/vi_block2_layer5/block2_b5_vi')))
        block2_conv5 = tf.contrib.layers.batch_norm(tf.nn.conv2d(block2_conv4,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block2_conv5 = lrelu(block2_conv5)

    with tf.compat.v1.variable_scope('vi_block3'):
      with tf.compat.v1.variable_scope('vi_block3_stride2'):
        weighs = tf.compat.v1.get_variable('block3_w1_vi', initializer=tf.constant(reader.get_tensor('vi_block3/vi_block3_stride2/block3_w1_vi')))
        bias = tf.compat.v1.get_variable('block3_b1_vi',initializer=tf.constant(reader.get_tensor('vi_block3/vi_block3_stride2/block3_b1_vi')))
        block3_stride = tf.contrib.layers.batch_norm(tf.nn.conv2d(block2_conv5,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block3_stride = lrelu(block3_stride)
      with tf.compat.v1.variable_scope('vi_block3_layer2'):
        weighs = tf.compat.v1.get_variable('block3_w2_vi',initializer=tf.constant(reader.get_tensor('vi_block3/vi_block3_layer2/block3_w2_vi')))
        bias = tf.compat.v1.get_variable('block3_b2_vi',initializer=tf.constant(reader.get_tensor('vi_block3/vi_block3_layer2/block3_b2_vi')))
        block3_conv2 = tf.contrib.layers.batch_norm(tf.nn.conv2d(block3_stride,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block3_conv2 = lrelu(block3_conv2)
      with tf.compat.v1.variable_scope('vi_block3_layer3'):
        weighs = tf.compat.v1.get_variable('block3_w3_vi', initializer=tf.constant(reader.get_tensor('vi_block3/vi_block3_layer3/block3_w3_vi')))
        bias = tf.compat.v1.get_variable('block3_b3_vi',initializer=tf.constant(reader.get_tensor('vi_block3/vi_block3_layer3/block3_b3_vi')))
        block3_conv3 = tf.contrib.layers.batch_norm(tf.nn.conv2d(block3_conv2,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block3_conv3 = lrelu(block3_conv3)
      with tf.compat.v1.variable_scope('vi_block3_layer4'):
        weighs = tf.compat.v1.get_variable('block3_w4_vi', initializer=tf.constant(reader.get_tensor('vi_block3/vi_block3_layer4/block3_w4_vi')))
        bias = tf.compat.v1.get_variable('b4_vi',initializer=tf.constant(reader.get_tensor('vi_block3/vi_block3_layer4/b4_vi')))
        block3_conv4 = tf.contrib.layers.batch_norm(tf.nn.conv2d(block3_conv3,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block3_conv4 = lrelu(block3_conv4)
      with tf.compat.v1.variable_scope('vi_block3_layer5'):
        weighs = tf.compat.v1.get_variable('block3_w5_vi', initializer=tf.constant(reader.get_tensor('vi_block3/vi_block3_layer5/block3_w5_vi')))
        bias = tf.compat.v1.get_variable('block3_b5_vi',initializer=tf.constant(reader.get_tensor('vi_block3/vi_block3_layer5/block3_b5_vi')))
        block3_conv5 = tf.contrib.layers.batch_norm(tf.nn.conv2d(block3_conv4,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block3_conv5 = lrelu(block3_conv5)
      with tf.compat.v1.variable_scope('vi_block3_layer6'):
        weighs = tf.compat.v1.get_variable('block3_w6_vi', initializer=tf.constant(reader.get_tensor('vi_block3/vi_block3_layer6/block3_w6_vi')))
        bias = tf.compat.v1.get_variable('block3_b6_vi',initializer=tf.constant(reader.get_tensor('vi_block3/vi_block3_layer6/block3_b6_vi')))
        block3_conv6 = tf.contrib.layers.batch_norm(tf.nn.conv2d(block3_conv5,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block3_conv6 = lrelu(block3_conv6)
      with tf.compat.v1.variable_scope('vi_block3_layer7'):
        weighs = tf.compat.v1.get_variable('block3_w7_vi',initializer=tf.constant(reader.get_tensor('vi_block3/vi_block3_layer7/block3_w7_vi')))
        bias = tf.compat.v1.get_variable('block3_b7_vi',initializer=tf.constant(reader.get_tensor('vi_block3/vi_block3_layer7/block3_b7_vi')))
        block3_conv7 = tf.contrib.layers.batch_norm(tf.nn.conv2d(block3_conv6,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block3_conv7 = lrelu(block3_conv7)
      with tf.compat.v1.variable_scope('vi_block3_layer8'):
        weighs = tf.compat.v1.get_variable('block3_w8_vi', initializer=tf.constant(reader.get_tensor('vi_block3/vi_block3_layer8/block3_w8_vi')))
        bias = tf.compat.v1.get_variable('block3_b8_vi',initializer=tf.constant(reader.get_tensor('vi_block3/vi_block3_layer8/block3_b8_vi')))
        block3_conv8 = tf.contrib.layers.batch_norm(tf.nn.conv2d(block3_conv7,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block3_conv8 = lrelu(block3_conv8)
      with tf.compat.v1.variable_scope('vi_block3_layer9'):
        weighs = tf.compat.v1.get_variable('block3_w9_vi', initializer=tf.constant(reader.get_tensor('vi_block3/vi_block3_layer9/block3_w9_vi')))
        bias = tf.compat.v1.get_variable('block3_b9_vi', initializer=tf.constant(reader.get_tensor('vi_block3/vi_block3_layer9/block3_b9_vi')))
        block3_conv9 = tf.contrib.layers.batch_norm(tf.nn.conv2d(block3_conv8,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block3_conv9 = lrelu(block3_conv9)

    with tf.compat.v1.variable_scope('vi_block4'):
      with tf.compat.v1.variable_scope('vi_block4_stride2'):
        weighs = tf.compat.v1.get_variable('block4_w1_vi', initializer=tf.constant(reader.get_tensor('vi_block4/vi_block4_stride2/block4_w1_vi')))
        bias = tf.compat.v1.get_variable('block4_b1_vi', initializer=tf.constant(reader.get_tensor('vi_block4/vi_block4_stride2/block4_b1_vi')))
        block4_stride = tf.contrib.layers.batch_norm(tf.nn.conv2d(block3_conv9,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block4_stride = lrelu(block4_stride)
      with tf.compat.v1.variable_scope('vi_block4_layer2'):
        weighs = tf.compat.v1.get_variable('block4_w2_vi', initializer=tf.constant(reader.get_tensor('vi_block4/vi_block4_layer2/block4_w2_vi')))
        bias = tf.compat.v1.get_variable('block4_b2_vi', initializer=tf.constant(reader.get_tensor('vi_block4/vi_block4_layer2/block4_b2_vi')))
        block4_conv2 = tf.contrib.layers.batch_norm(tf.nn.conv2d(block4_stride,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block4_conv2 = lrelu(block4_conv2)
      with tf.compat.v1.variable_scope('vi_block4_layer3'):
        weighs = tf.compat.v1.get_variable('block4_w3_vi',  initializer=tf.constant(reader.get_tensor('vi_block4/vi_block4_layer3/block4_w3_vi')))
        bias = tf.compat.v1.get_variable('block4_b3_vi', initializer=tf.constant(reader.get_tensor('vi_block4/vi_block4_layer3/block4_b3_vi')))
        block4_conv3 = tf.contrib.layers.batch_norm(tf.nn.conv2d(block4_conv2,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block4_conv3 = lrelu(block4_conv3)

    with tf.compat.v1.variable_scope('vi_intensity'):
      intensity_concat = tf.concat([block1_conv3, block2_conv5, block3_conv9, block4_conv3],axis=-1)
      with tf.compat.v1.variable_scope('vi_abstract_intensity'):
        weighs = tf.compat.v1.get_variable('vi_layer_intensity_w',  initializer=tf.constant(reader.get_tensor('vi_intensity/vi_abstract_intensity/vi_layer_intensity_w')))
        bias = tf.compat.v1.get_variable('vi_layer_intensity_b', initializer=tf.constant(reader.get_tensor('vi_intensity/vi_abstract_intensity/vi_layer_intensity_b')))
        intensity1 = tf.contrib.layers.batch_norm(tf.nn.conv2d(intensity_concat,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        intensity1 = lrelu(intensity1)
      with tf.compat.v1.variable_scope('vi_intensity_single'):
        weighs = tf.compat.v1.get_variable('vi_intensity_w',  initializer=tf.constant(reader.get_tensor('vi_intensity/vi_intensity_single/vi_intensity_w')))
        bias = tf.compat.v1.get_variable('vi_intensity_b', initializer=tf.constant(reader.get_tensor('vi_intensity/vi_intensity_single/vi_intensity_b')))
        intensity = tf.contrib.layers.batch_norm(tf.nn.conv2d(intensity1,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        intensity = lrelu(intensity)

    with tf.compat.v1.variable_scope('vi_gradient'):
      gradient_concat = tf.concat([block1_conv3, block2_conv5, block3_conv9, block4_conv3], axis=-1)
      with tf.compat.v1.variable_scope('vi_abstract_gradient'):
        weighs = tf.compat.v1.get_variable('vi_layer_gradient_w',  initializer=tf.constant(reader.get_tensor('vi_gradient/vi_abstract_gradient/vi_layer_gradient_w')))
        bias = tf.compat.v1.get_variable('vi_layer_gradient_b', initializer=tf.constant(reader.get_tensor('vi_gradient/vi_abstract_gradient/vi_layer_gradient_b')))
        gradient1 = tf.contrib.layers.batch_norm(tf.nn.conv2d(gradient_concat,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        gradient1 = lrelu(gradient1)
      with tf.compat.v1.variable_scope('vi_gradient_single'):
        weights = tf.compat.v1.get_variable('vi_gradient_w',  initializer=tf.constant(reader.get_tensor('vi_gradient/vi_gradient_single/vi_gradient_w')))
        bias = tf.compat.v1.get_variable('vi_gradient_b', initializer=tf.constant(reader.get_tensor('vi_gradient/vi_gradient_single/vi_gradient_b')))
        gradient = tf.contrib.layers.batch_norm(tf.nn.conv2d(gradient1,weights,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        gradient = lrelu(gradient)
    
    return intensity, gradient

def ir_decoder(ir_concat):
    with tf.compat.v1.variable_scope('ir_decoder'):
      with tf.compat.v1.variable_scope('ir_layer1'):
        weights = tf.compat.v1.get_variable('ir_layer1_w', initializer=tf.constant(reader.get_tensor('ir_decoder/ir_layer1/ir_layer1_w')))
        bias = tf.compat.v1.get_variable('ir_layer1_b', initializer=tf.constant(reader.get_tensor('ir_decoder/ir_layer1/ir_layer1_b')))
        conv1 = tf.contrib.layers.batch_norm(tf.nn.conv2d(ir_concat,weights,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        conv1 = lrelu(conv1)
      with tf.compat.v1.variable_scope('ir_layer2'):
        weights = tf.compat.v1.get_variable('ir_layer2_w', initializer=tf.constant(reader.get_tensor('ir_decoder/ir_layer2/ir_layer2_w')))
        bias = tf.compat.v1.get_variable('ir_layer2_b', initializer=tf.constant(reader.get_tensor('ir_decoder/ir_layer2/ir_layer2_b')))
        conv2 = tf.contrib.layers.batch_norm(tf.nn.conv2d(conv1,weights,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        conv2 = lrelu(conv2)
      with tf.compat.v1.variable_scope('ir_fusion'):
        weights = tf.compat.v1.get_variable('ir_fusion_w', initializer=tf.constant(reader.get_tensor('ir_decoder/ir_fusion/ir_fusion_w')))
        bias = tf.compat.v1.get_variable('ir_fusion_b', initializer=tf.constant(reader.get_tensor('ir_decoder/ir_fusion/ir_fusion_b')))
        ir_fusion = tf.contrib.layers.batch_norm(tf.nn.conv2d(conv2,weights,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        ir_fusion = lrelu(ir_fusion)
    
    return ir_fusion

def vi_decoder(vi_concat):
    with tf.compat.v1.variable_scope('vi_decoder'):
      with tf.compat.v1.variable_scope('vi_layer1'):
        weights = tf.compat.v1.get_variable('vi_layer1_w', initializer=tf.constant(reader.get_tensor('vi_decoder/vi_layer1/vi_layer1_w')))
        bias = tf.compat.v1.get_variable('vi_layer1_b', initializer=tf.constant(reader.get_tensor('vi_decoder/vi_layer1/vi_layer1_b')))
        conv1 = tf.contrib.layers.batch_norm(tf.nn.conv2d(vi_concat,weights,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        conv1 = lrelu(conv1)
      with tf.compat.v1.variable_scope('vi_layer2'):
        weights = tf.compat.v1.get_variable('vi_layer2_w', initializer=tf.constant(reader.get_tensor('vi_decoder/vi_layer2/vi_layer2_w')))
        bias = tf.compat.v1.get_variable('vi_layer2_b', initializer=tf.constant(reader.get_tensor('vi_decoder/vi_layer2/vi_layer2_b')))
        conv2 = tf.contrib.layers.batch_norm(tf.nn.conv2d(conv1,weights,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        conv2 = lrelu(conv2)
      with tf.compat.v1.variable_scope('vi_fusion'):
        weights = tf.compat.v1.get_variable('vi_fusion_w', initializer=tf.constant(reader.get_tensor('vi_decoder/vi_fusion/vi_fusion_w')))
        bias = tf.compat.v1.get_variable('vi_fusion_b', initializer=tf.constant(reader.get_tensor('vi_decoder/vi_fusion/vi_fusion_b')))
        vi_fusion = tf.contrib.layers.batch_norm(tf.nn.conv2d(conv2,weights,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        vi_fusion = lrelu(vi_fusion)
    
    return vi_fusion



def fusion_model(img_ir,img_vi):
    with tf.compat.v1.variable_scope('fusion_model'):
####################  Layer1  ###########################
        with tf.compat.v1.variable_scope('layer1'):
            weights=tf.compat.v1.get_variable("w1",initializer=tf.constant(reader.get_tensor('fusion_model/layer1/w1')))
            bias=tf.compat.v1.get_variable("b1",initializer=tf.constant(reader.get_tensor('fusion_model/layer1/b1')))
            conv1_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(img_ir, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv1_ir = lrelu(conv1_ir)
        with tf.compat.v1.variable_scope('layer1_vi'):
            weights=tf.compat.v1.get_variable("w1_vi",initializer=tf.constant(reader.get_tensor('fusion_model/layer1_vi/w1_vi')))
            bias=tf.compat.v1.get_variable("b1_vi",initializer=tf.constant(reader.get_tensor('fusion_model/layer1_vi/b1_vi')))
            conv1_vi= tf.contrib.layers.batch_norm(tf.nn.conv2d(img_vi, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv1_vi = lrelu(conv1_vi)    
                            
####################  Layer2  ###########################           
            
                      
        with tf.compat.v1.variable_scope('layer2'):
            weights=tf.compat.v1.get_variable("w2",initializer=tf.constant(reader.get_tensor('fusion_model/layer2/w2')))
            bias=tf.compat.v1.get_variable("b2",initializer=tf.constant(reader.get_tensor('fusion_model/layer2/b2')))
            conv2_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv1_ir, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv2_ir = lrelu(conv2_ir)
        with tf.compat.v1.variable_scope('layer2_vi'):
            weights=tf.compat.v1.get_variable("w2_vi",initializer=tf.constant(reader.get_tensor('fusion_model/layer2_vi/w2_vi')))
            bias=tf.compat.v1.get_variable("b2_vi",initializer=tf.constant(reader.get_tensor('fusion_model/layer2_vi/b2_vi')))
            conv2_vi= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv1_vi, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv2_vi = lrelu(conv2_vi)   
            
        conv_2_midle =tf.concat([conv2_ir,conv2_vi],axis=-1)      
        
        with tf.compat.v1.variable_scope('layer2_3'):
            weights=tf.compat.v1.get_variable("w2_3",initializer=tf.constant(reader.get_tensor('fusion_model/layer2_3/w2_3')))
            bias=tf.compat.v1.get_variable("b2_3",initializer=tf.constant(reader.get_tensor('fusion_model/layer2_3/b2_3')))
            conv2_3_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv_2_midle, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv2_3_ir = lrelu(conv2_3_ir)
        with tf.compat.v1.variable_scope('layer2_3_vi'):
            weights=tf.compat.v1.get_variable("w2_3_vi",initializer=tf.constant(reader.get_tensor('fusion_model/layer2_3_vi/w2_3_vi')))
            bias=tf.compat.v1.get_variable("b2_3_vi",initializer=tf.constant(reader.get_tensor('fusion_model/layer2_3_vi/b2_3_vi')))
            conv2_3_vi= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv_2_midle, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv2_3_vi = lrelu(conv2_3_vi)               
                                      
####################  Layer3  ###########################                 
        conv_12_ir=tf.concat([conv1_ir,conv2_ir,conv2_3_ir],axis=-1)
        conv_12_vi=tf.concat([conv1_vi,conv2_vi,conv2_3_vi],axis=-1)                   
         
        with tf.compat.v1.variable_scope('layer3'):
            weights=tf.compat.v1.get_variable("w3",initializer=tf.constant(reader.get_tensor('fusion_model/layer3/w3')))
            bias=tf.compat.v1.get_variable("b3",initializer=tf.constant(reader.get_tensor('fusion_model/layer3/b3')))
            conv3_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv_12_ir, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv3_ir = lrelu(conv3_ir)            
        with tf.compat.v1.variable_scope('layer3_vi'):
            weights=tf.compat.v1.get_variable("w3_vi",initializer=tf.constant(reader.get_tensor('fusion_model/layer3_vi/w3_vi')))
            bias=tf.compat.v1.get_variable("b3_vi",initializer=tf.constant(reader.get_tensor('fusion_model/layer3_vi/b3_vi')))
            conv3_vi= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv_12_vi, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv3_vi =lrelu(conv3_vi)            

        conv_3_midle =tf.concat([conv3_ir,conv3_vi],axis=-1)    
        
        with tf.compat.v1.variable_scope('layer3_4'):
            weights=tf.compat.v1.get_variable("w3_4",initializer=tf.constant(reader.get_tensor('fusion_model/layer3_4/w3_4')))
            bias=tf.compat.v1.get_variable("b3_4",initializer=tf.constant(reader.get_tensor('fusion_model/layer3_4/b3_4')))
            conv3_4_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv_3_midle, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv3_4_ir = lrelu(conv3_4_ir)
        with tf.compat.v1.variable_scope('layer3_4_vi'):
            weights=tf.compat.v1.get_variable("w3_4_vi",initializer=tf.constant(reader.get_tensor('fusion_model/layer3_4_vi/w3_4_vi')))
            bias=tf.compat.v1.get_variable("b3_4_vi",initializer=tf.constant(reader.get_tensor('fusion_model/layer3_4_vi/b3_4_vi')))
            conv3_4_vi= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv_3_midle, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv3_4_vi = lrelu(conv3_4_vi)  

####################  Layer4  ###########################                 
        conv_123_ir=tf.concat([conv1_ir,conv2_ir,conv3_ir,conv3_4_ir],axis=-1)
        conv_123_vi=tf.concat([conv1_vi,conv2_vi,conv3_vi,conv3_4_vi],axis=-1)               
            
          
        with tf.compat.v1.variable_scope('layer4'):
            weights=tf.compat.v1.get_variable("w4",initializer=tf.constant(reader.get_tensor('fusion_model/layer4/w4')))
            bias=tf.compat.v1.get_variable("b4",initializer=tf.constant(reader.get_tensor('fusion_model/layer4/b4')))
            conv4_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv_123_ir, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv4_ir = lrelu(conv4_ir)
            
        with tf.compat.v1.variable_scope('layer4_vi'):
            weights=tf.compat.v1.get_variable("w4_vi",initializer=tf.constant(reader.get_tensor('fusion_model/layer4_vi/w4_vi')))
            bias=tf.compat.v1.get_variable("b4_vi",initializer=tf.constant(reader.get_tensor('fusion_model/layer4_vi/b4_vi')))
            conv4_vi= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv_123_vi, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv4_vi = lrelu(conv4_vi)            
            
        conv_ir_vi =tf.concat([conv1_ir,conv1_vi,conv2_ir,conv2_vi,conv3_ir,conv3_vi,conv4_ir,conv4_vi],axis=-1)
                 
####################  Layer5  ###########################                          
        with tf.compat.v1.variable_scope('layer5'):
            weights=tf.compat.v1.get_variable("w5",initializer=tf.constant(reader.get_tensor('fusion_model/layer5/w5')))
            bias=tf.compat.v1.get_variable("b5",initializer=tf.constant(reader.get_tensor('fusion_model/layer5/b5')))
            conv5_ir= tf.nn.conv2d(conv_ir_vi, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv5_ir=tf.nn.tanh(conv5_ir)
    return conv5_ir




def input_setup(index):
    padding=0
    sub_ir_sequence = []
    sub_vi_sequence = []
    input_ir=(imread(data_ir[index])-127.5)/127.5    
    input_ir=np.lib.pad(input_ir,((padding,padding),(padding,padding)),'edge')
    w,h=input_ir.shape
    input_ir=input_ir.reshape([w,h,1])
    input_vi=(imread(data_vi[index])-127.5)/127.5
    input_vi=np.lib.pad(input_vi,((padding,padding),(padding,padding)),'edge')
    w,h=input_vi.shape
    input_vi=input_vi.reshape([w,h,1])
    sub_ir_sequence.append(input_ir)
    sub_vi_sequence.append(input_vi)
    train_data_ir= np.asarray(sub_ir_sequence)
    train_data_vi= np.asarray(sub_vi_sequence)
    return train_data_ir,train_data_vi

for idx_num in range(15):
  
  num_epoch=1
  while(num_epoch==idx_num):
      model_dir = '0303_1'
      reader = tf.compat.v1.train.NewCheckpointReader('./checkpoint/'+ model_dir +'/CGAN.model-250')
  
      with tf.name_scope('IR_input'):
          #红外图像patch
          images_ir = tf.compat.v1.placeholder(tf.float32, [1,None,None,None], name='images_ir')
      with tf.name_scope('VI_input'):
          #可见光图像patch
          images_vi = tf.compat.v1.placeholder(tf.float32, [1,None,None,None], name='images_vi')
          #self.labels_vi_gradient=gradient(self.labels_vi)
      #将红外和可见光图像在通道方向连起来，第一通道是红外图像，第二通道是可见光图像
      with tf.name_scope('input'):
          #resize_ir=tf.image.resize_images(images_ir, (512, 512), method=2)
          ir_intensity, ir_gradient = ir_encoder(images_ir)
          vi_intensity, vi_gradient = vi_encoder(images_vi)
          ir_concat = tf.concat([ir_intensity, ir_gradient], axis=-1)
          vi_concat = tf.concat([vi_intensity, vi_gradient], axis=-1)
          
          input_intensity = tf.concat([ir_intensity, ir_intensity, vi_intensity],axis=-1)
          input_gradient =tf.concat([vi_gradient, vi_gradient, ir_gradient],axis=-1)
  
      with tf.name_scope('fusion'):
          fusion_image=fusion_model(input_intensity,input_gradient)
  
  
      with tf.compat.v1.Session() as sess:
        #   print()
          init_op=tf.compat.v1.global_variables_initializer()
          sess.run(init_op)


        #   data_ir=prepare_data('../input/d/euterpek/medical/Medical/Test_ir')
        #   data_vi=prepare_data('../input/d/euterpek/medical/Medical/Test_vi')
          data_ir=prepare_data('Test_ir')
          data_vi=prepare_data('Test_vi')
          a=time.time()
          for i in range(len(data_ir)):
              start=time.time()
              train_data_ir,train_data_vi=input_setup(i)
              result =sess.run(fusion_image,feed_dict={images_ir: train_data_ir,images_vi: train_data_vi})
              result=result*127.5+127.5
              result = result.squeeze()
              image_path = os.path.join(os.getcwd(), 'result',model_dir)
              if not os.path.exists(image_path):
                  os.makedirs(image_path)
              if i<=9:
                  image_path = os.path.join(image_path,'F9_0'+str(i)+".bmp")
              else:
                  image_path = os.path.join(image_path,'F9_'+str(i)+".bmp")
              end=time.time()
              # print(out.shape)
              imsave(result, image_path)
              print("Testing [%d] success,Testing time is [%f]"%(i,end-start))
          b=time.time()
          print(b-a)
      tf.compat.v1.reset_default_graph()
      num_epoch=num_epoch+1

