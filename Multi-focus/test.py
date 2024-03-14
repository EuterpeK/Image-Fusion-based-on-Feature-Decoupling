from pickletools import long4
from pprint import pprint
from re import A
from matplotlib.pyplot import axis
import tensorflow as tf
import numpy as np
import scipy.misc
import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import glob
import cv2
import argparse

def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dir', type=str, default='auto')
  parser.add_argument('--epochs', type=int, default=10)

  args, _ = parser.parse_known_args()
  return args

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
    data = glob.glob(os.path.join(data_dir, "*.jpg"))
    data.extend(glob.glob(os.path.join(data_dir, "*.bmp")))
    data.sort(key=lambda x:int(x[len(data_dir)+1:-4]))
    return data

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)

def ir_encoder(img_ir):
        # with tf.compat.v1.variable_scope('ir_encoder'):
    with tf.compat.v1.variable_scope('ir_encoder'):
      with tf.compat.v1.variable_scope('ir_layer1'):
        weighs = tf.compat.v1.get_variable('ir_w1', initializer=tf.constant(reader.get_tensor('ir_encoder/ir_layer1/ir_w1')))
        bias = tf.compat.v1.get_variable('ir_b1',initializer=tf.constant(reader.get_tensor('ir_encoder/ir_layer1/ir_b1')))
        layer1 = tf.contrib.layers.batch_norm(tf.nn.conv2d(img_ir,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        layer1 = lrelu(layer1)

      # concat1 = tf.concat([img_ir, layer1], axis=-1)

      with tf.compat.v1.variable_scope('ir_layer2'):
        weighs = tf.compat.v1.get_variable('ir_w2', initializer=tf.constant(reader.get_tensor('ir_encoder/ir_layer2/ir_w2')))
        bias = tf.compat.v1.get_variable('ir_b2',initializer=tf.constant(reader.get_tensor('ir_encoder/ir_layer2/ir_b2')))
        layer1 = tf.contrib.layers.batch_norm(tf.nn.conv2d(layer1,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        layer2 = lrelu(layer1)
      
      concat1 = tf.concat([ layer1, layer2], axis=-1)

      with tf.compat.v1.variable_scope('ir_layer3'):
        weighs = tf.compat.v1.get_variable('ir_w3', initializer=tf.constant(reader.get_tensor('ir_encoder/ir_layer3/ir_w3')))
        bias = tf.compat.v1.get_variable('ir_b3', initializer=tf.constant(reader.get_tensor('ir_encoder/ir_layer3/ir_b3')))
        layer3 = tf.contrib.layers.batch_norm(tf.nn.conv2d(concat1,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        layer3 = lrelu(layer3)
      
      concat2 = tf.concat([layer1, layer2, layer3], axis=-1)

      with tf.compat.v1.variable_scope('ir_layer4'):
        weighs = tf.compat.v1.get_variable('ir_w4', initializer=tf.constant(reader.get_tensor('ir_encoder/ir_layer4/ir_w4')))
        bias = tf.compat.v1.get_variable('ir_b4', initializer=tf.constant(reader.get_tensor('ir_encoder/ir_layer4/ir_b4')))
        layer4 = tf.contrib.layers.batch_norm(tf.nn.conv2d(concat2,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        layer4 = lrelu(layer3)

      concat3 = tf.concat([layer1, layer2, layer3, layer4], axis=-1)

      with tf.compat.v1.variable_scope('ir_layer5'):
        weighs = tf.compat.v1.get_variable('ir_w5', initializer=tf.constant(reader.get_tensor('ir_encoder/ir_layer5/ir_w5')))
        bias = tf.compat.v1.get_variable('ir_b5', initializer=tf.constant(reader.get_tensor('ir_encoder/ir_layer5/ir_b5')))
        layer5 = tf.contrib.layers.batch_norm(tf.nn.conv2d(concat3,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        layer5 = lrelu(layer5)
      
      concat4 = tf.concat([layer1, layer2, layer3, layer4, layer5], axis=-1)

      with tf.compat.v1.variable_scope('ir_intensity'):
        weighs = tf.compat.v1.get_variable('ir_intensity_w', initializer=tf.constant(reader.get_tensor('ir_encoder/ir_intensity/ir_intensity_w')))
        bias = tf.compat.v1.get_variable('ir_intensity_b', initializer=tf.constant(reader.get_tensor('ir_encoder/ir_intensity/ir_intensity_b')))
        intensity = tf.contrib.layers.batch_norm(tf.nn.conv2d(concat4,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        intensity = lrelu(intensity)
      with tf.compat.v1.variable_scope('ir_res'):
        weighs = tf.compat.v1.get_variable('ir_res_w', initializer=tf.constant(reader.get_tensor('ir_encoder/ir_res/ir_res_w')))
        bias = tf.compat.v1.get_variable('ir_res_b',initializer=tf.constant(reader.get_tensor('ir_encoder/ir_res/ir_res_b')))
        res = tf.contrib.layers.batch_norm(tf.nn.conv2d(concat4,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        res = lrelu(res)
    
    return intensity, res 

def vi_encoder(img_vi):
        # with tf.compat.v1.variable_scope('vi_encoder'):
    with tf.compat.v1.variable_scope('vi_encoder'):
      with tf.compat.v1.variable_scope('vi_layer1'):
        weighs = tf.compat.v1.get_variable('vi_w1', initializer=tf.constant(reader.get_tensor('vi_encoder/vi_layer1/vi_w1')))
        bias = tf.compat.v1.get_variable('vi_b1',initializer=tf.constant(reader.get_tensor('vi_encoder/vi_layer1/vi_b1')))
        layer1 = tf.contrib.layers.batch_norm(tf.nn.conv2d(img_vi,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        layer1 = lrelu(layer1)

      # concat1 = tf.concat([img_vi, layer1], axis=-1)

      with tf.compat.v1.variable_scope('vi_layer2'):
        weighs = tf.compat.v1.get_variable('vi_w2', initializer=tf.constant(reader.get_tensor('vi_encoder/vi_layer2/vi_w2')))
        bias = tf.compat.v1.get_variable('vi_b2',initializer=tf.constant(reader.get_tensor('vi_encoder/vi_layer2/vi_b2')))
        layer1 = tf.contrib.layers.batch_norm(tf.nn.conv2d(layer1,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        layer2 = lrelu(layer1)
      
      concat1 = tf.concat([ layer1, layer2], axis=-1)

      with tf.compat.v1.variable_scope('vi_layer3'):
        weighs = tf.compat.v1.get_variable('vi_w3', initializer=tf.constant(reader.get_tensor('vi_encoder/vi_layer3/vi_w3')))
        bias = tf.compat.v1.get_variable('vi_b3', initializer=tf.constant(reader.get_tensor('vi_encoder/vi_layer3/vi_b3')))
        layer3 = tf.contrib.layers.batch_norm(tf.nn.conv2d(concat1,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        layer3 = lrelu(layer3)
      
      concat2 = tf.concat([layer1, layer2, layer3], axis=-1)

      with tf.compat.v1.variable_scope('vi_layer4'):
        weighs = tf.compat.v1.get_variable('vi_w4', initializer=tf.constant(reader.get_tensor('vi_encoder/vi_layer4/vi_w4')))
        bias = tf.compat.v1.get_variable('vi_b4', initializer=tf.constant(reader.get_tensor('vi_encoder/vi_layer4/vi_b4')))
        layer4 = tf.contrib.layers.batch_norm(tf.nn.conv2d(concat2,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        layer4 = lrelu(layer3)

      concat3 = tf.concat([layer1, layer2, layer3, layer4], axis=-1)

      with tf.compat.v1.variable_scope('vi_layer5'):
        weighs = tf.compat.v1.get_variable('vi_w5', initializer=tf.constant(reader.get_tensor('vi_encoder/vi_layer5/vi_w5')))
        bias = tf.compat.v1.get_variable('vi_b5', initializer=tf.constant(reader.get_tensor('vi_encoder/vi_layer5/vi_b5')))
        layer5 = tf.contrib.layers.batch_norm(tf.nn.conv2d(concat3,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        layer5 = lrelu(layer5)
      
      concat4 = tf.concat([layer1, layer2, layer3, layer4, layer5], axis=-1)

      with tf.compat.v1.variable_scope('vi_gradient'):
        weighs = tf.compat.v1.get_variable('vi_gradient_w', initializer=tf.constant(reader.get_tensor('vi_encoder/vi_gradient/vi_gradient_w')))
        bias = tf.compat.v1.get_variable('vi_gradient_b', initializer=tf.constant(reader.get_tensor('vi_encoder/vi_gradient/vi_gradient_b')))
        gradient = tf.contrib.layers.batch_norm(tf.nn.conv2d(concat4,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        gradient = lrelu(gradient)
      with tf.compat.v1.variable_scope('vi_res'):
        weighs = tf.compat.v1.get_variable('vi_res_w', initializer=tf.constant(reader.get_tensor('vi_encoder/vi_res/vi_res_w')))
        bias = tf.compat.v1.get_variable('vi_res_b',initializer=tf.constant(reader.get_tensor('vi_encoder/vi_res/vi_res_b')))
        res = tf.contrib.layers.batch_norm(tf.nn.conv2d(concat4,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        res = lrelu(res)
    
    return gradient, res 

def ir_decoder(ir_concat):
    with tf.compat.v1.variable_scope('ir_decoder'):
      with tf.compat.v1.variable_scope('ir_layer1'):
        weights = tf.compat.v1.get_variable('ir_w1', initializer=tf.constant(reader.get_tensor('ir_decoder/ir_layer1/ir_w1')))
        bias = tf.compat.v1.get_variable('ir_b1', initializer=tf.constant(reader.get_tensor('ir_decoder/ir_layer1/ir_b1')))
        layer1 = tf.contrib.layers.batch_norm(tf.nn.conv2d(ir_concat,weights,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        layer1 = lrelu(layer1)
      with tf.compat.v1.variable_scope('ir_layer2'):
        weights = tf.compat.v1.get_variable('ir_w2', initializer=tf.constant(reader.get_tensor('ir_decoder/ir_layer2/ir_w2')))
        bias = tf.compat.v1.get_variable('ir_b2', initializer=tf.constant(reader.get_tensor('ir_decoder/ir_layer2/ir_b2')))
        ir_fusion = tf.contrib.layers.batch_norm(tf.nn.conv2d(layer1,weights,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        ir_fusion = tf.nn.tanh(ir_fusion)
    
    return ir_fusion

def vi_decoder(vi_concat):
    with tf.compat.v1.variable_scope('vi_decoder'):
      with tf.compat.v1.variable_scope('vi_layer1'):
        weights = tf.compat.v1.get_variable('vi_w1', initializer=tf.constant(reader.get_tensor('vi_decoder/vi_layer1/vi_w1')))
        bias = tf.compat.v1.get_variable('vi_b1', initializer=tf.constant(reader.get_tensor('vi_decoder/vi_layer1/vi_b1')))
        layer1 = tf.contrib.layers.batch_norm(tf.nn.conv2d(vi_concat,weights,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        layer1 = lrelu(layer1)
      with tf.compat.v1.variable_scope('vi_layer2'):
        weights = tf.compat.v1.get_variable('vi_w2', initializer=tf.constant(reader.get_tensor('vi_decoder/vi_layer2/vi_w2')))
        bias = tf.compat.v1.get_variable('vi_b2', initializer=tf.constant(reader.get_tensor('vi_decoder/vi_layer2/vi_b2')))
        vi_fusion = tf.contrib.layers.batch_norm(tf.nn.conv2d(layer1,weights,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        vi_fusion = tf.nn.tanh(vi_fusion)
    
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

args = get_args()

model_dir = args.dir

for epoch in range(args.epochs):
  reader = tf.compat.v1.train.NewCheckpointReader('./checkpoint/'+model_dir+'/CGAN.model-'+str(epoch))
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
      
      ir_fusion = ir_decoder(ir_concat)
      vi_fusion = vi_decoder(vi_concat)

      input_intensity = tf.concat([ir_intensity, vi_intensity, images_ir],axis=-1)
      input_gradient =tf.concat([vi_gradient, ir_gradient, images_vi],axis=-1)

  with tf.name_scope('fusion'):
      fusion_image=fusion_model(input_intensity,input_gradient)


  with tf.compat.v1.Session() as sess:
    #   print()
      init_op=tf.compat.v1.global_variables_initializer()
      sess.run(init_op)
      data_ir=prepare_data('../Multi-focus/Test_ir')
      data_vi=prepare_data('../Multi-focus/Test_vi')
      for i in range(len(data_ir)):
          # if i==1:
          #   break
          start=time.time()
          train_data_ir,train_data_vi=input_setup(i)
          result, ir_result, vi_result =sess.run([fusion_image, ir_fusion, vi_fusion],feed_dict={images_ir: train_data_ir,images_vi: train_data_vi})
          # ir_result, vi_result =sess.run([ir_fusion, vi_fusion],feed_dict={images_ir: train_data_ir,images_vi: train_data_vi})

          result=result*127.5+127.5

          # ir_result = ir_result * 127.5 + 127.5
          # vi_result = vi_result * 127.5 + 127.5
          # ir_result = ir_result.squeeze()
          # vi_result = vi_result.squeeze()
          
          result = result.squeeze()
          image_path = os.path.join(os.getcwd(), 'result',model_dir)
          if not os.path.exists(image_path):
              os.makedirs(image_path)
          auto_path = os.path.join('eval', model_dir)
          if not os.path.exists(auto_path):
            os.makedirs(auto_path)
          fusion_path = os.path.join(image_path,'F' + str(epoch+1) + '_'+str(i)+".bmp")
          end=time.time()
          if i==0:
            imsave(result, fusion_path)
          imsave(result, os.path.join(auto_path,'F' + str(epoch+1) + '_'+str(i)+".bmp" ))
          print("Testing [%d] success,Testing time is [%f]"%(i,end-start))
  tf.compat.v1.reset_default_graph()


