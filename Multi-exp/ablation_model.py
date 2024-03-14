# -*- coding: utf-8 -*-
from matplotlib.pyplot import axis
from utils import (
  read_data, 
  input_setup, 
  imsave,
  merge,
  gradient,
  lrelu,
)

import time
import os
os.environ['CUDA_VISIBLE_DEVIDES']='0'

import tensorflow as tf

class CGAN(object):

  def __init__(self, 
               sess, 
               image_size=132,
               label_size=120,
               batch_size=32,
               c_dim=1, 
               checkpoint_dir=None, 
               sample_dir=None,
               vi_gradient=1.0,
               vi_res=1.0,
               vi_fusion_gradient=1.0,
               vi_fusion_intensity=1.0,
               ir_intensity=1.0,
               ir_res=1.0,
               ir_fusion_intensity=1.0,
               ir_fusion_gradient=1.0,
               vi_loss=1.0,
               ir_loss=1.0,
               fusion_vi_gradient=7.0,
               fusion_vi_intensity=1.01,
               fusion_ir_gradient=1.27,
               fusion_ir_intensity=2.2,
               dir_name='auto'):

    self.sess = sess
    self.is_grayscale = (c_dim == 1)
    self.image_size = image_size
    self.label_size = label_size
    self.batch_size = batch_size

    self.c_dim = c_dim

    self.checkpoint_dir = checkpoint_dir
    self.sample_dir = sample_dir

    self.vi_gradient_weight = vi_gradient
    self.vi_res_weight = vi_res
    self.vi_fusion_gradient_weight = vi_fusion_gradient
    self.vi_fusion_intensity_weight = vi_fusion_intensity
    self.ir_intensity_weight = ir_intensity
    self.ir_res_weight = ir_res
    self.ir_fusion_intensity_weight = ir_fusion_intensity
    self.ir_fusion_gradient_weight = ir_fusion_gradient
    self.vi_loss_weight = vi_loss
    self.ir_loss_weight = ir_loss
    self.fusion_vi_gradient_weight = fusion_vi_gradient
    self.fusion_vi_intensity_weight = fusion_vi_intensity
    self.fusion_ir_gradient_weight = fusion_ir_gradient
    self.fusion_ir_intensity_weight = fusion_ir_intensity
    self.store_dir = dir_name

    self.build_model()

  def build_model(self):
    with tf.name_scope('IR_input'):
        #红外图像patch
        self.images_ir = tf.compat.v1.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images_ir')
        self.labels_ir = tf.compat.v1.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name='labels_ir')
    with tf.name_scope('VI_input'):
        #可见光图像patch
        self.images_vi = tf.compat.v1.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images_vi')
        self.labels_vi = tf.compat.v1.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name='labels_vi')
        #self.labels_vi_gradient=gradient(self.labels_vi)
    #将红外和可见光图像在通道方向连起来，第一通道是红外图像，第二通道是可见光图像
    with tf.name_scope('input'):
        #self.resize_ir=tf.image.resize_images(self.images_ir, (self.image_size, self.image_size), method=2)
        self.ir_intensity, self.ir_res = self.ir_encoder(self.labels_ir)
        self.vi_gradient, self.vi_res = self.vi_encoder(self.labels_vi)
        # ir_concat = tf.concat([self.ir_intensity, self.ir_res], axis=-1)
        # vi_concat = tf.concat([self.vi_gradient, self.vi_res], axis=-1)
        # self.ir_fusion = self.ir_decoder(ir_concat)
        # self.vi_fusion = self.vi_decoder(vi_concat)
        self.input_intensity =tf.concat([self.ir_intensity,self.vi_res,self.labels_ir],axis=-1)
        self.input_gradient =tf.concat([self.vi_gradient,self.ir_res,self.labels_vi],axis=-1)
    #融合图像
    with tf.name_scope('fusion'): 
        self.fusion_image=self.fusion_model(self.input_intensity,self.input_gradient)

    with tf.name_scope('g_loss'):
        self.vi_loss = 1.8*tf.reduce_mean(1-tf.image.ssim(self.labels_vi,self.vi_gradient,1.0)) + 1.5* tf.reduce_mean(tf.square(self.labels_vi - self.vi_res)) 

        self.ir_loss = 1.8*tf.reduce_mean(1-tf.image.ssim(self.labels_ir, self.vi_res, 1.0)) + 1.5* tf.reduce_mean(tf.square(self.labels_ir-self.ir_intensity))
        
        self.g_loss_2= 1 * self.vi_loss + 1*self.ir_loss + \
                       1.6*tf.reduce_mean(tf.square(self.fusion_image - self.labels_ir))+ 1.5*tf.reduce_mean(tf.square(self.fusion_image - self.labels_vi))+ \
                       17.6*tf.reduce_mean(1-tf.image.ssim(self.labels_vi, self.fusion_image, 1.0))+ 16*tf.reduce_mean(1-tf.image.ssim(self.labels_ir, self.fusion_image, 1.0))
        
        ## auto tune arguments
        # self.vi_loss = self.vi_res_weight*tf.reduce_mean(tf.square(self.vi_fusion - self.labels_vi)) + self.vi_gradient_weight*(1- tf.reduce_mean(tf.image.ssim(self.labels_vi, self.vi_fusion, 1.0))) + \
        #                self.vi_fusion_gradient_weight*tf.reduce_mean(1-tf.image.ssim(self.labels_vi,self.vi_gradient,1.0)) + self.vi_fusion_intensity_weight* tf.reduce_mean(tf.square(self.labels_vi - self.vi_res)) 
        #                 # 10 * tf.reduce_mean(tf.square(gradient(self.vi_fusion) - gradient(self.labels_vi)))
        #                 # 1*tf.reduce_mean(1-tf.image.ssim(self.labels_vi,self.vi_gradient,1.0) 
        # self.ir_loss = self.ir_intensity_weight*tf.reduce_mean(tf.square(self.ir_fusion - self.labels_ir)) + self.ir_res_weight*tf.reduce_mean(1-tf.image.ssim(self.labels_ir, self.ir_fusion, 1.0)) + \
        #                self.ir_fusion_gradient_weight*tf.reduce_mean(1-tf.image.ssim(self.labels_ir, self.vi_res, 1.0)) + self.ir_fusion_intensity_weight* tf.reduce_mean(tf.square(self.labels_ir-self.ir_intensity))
        # self.g_loss_2= self.vi_loss_weight * self.vi_loss + self.ir_loss_weight*self.ir_loss + \
        #                self.fusion_ir_intensity_weight*tf.reduce_mean(tf.square(self.fusion_image - self.labels_ir))+ self.fusion_vi_intensity_weight*tf.reduce_mean(tf.square(self.fusion_image - self.labels_vi))+ \
        #                self.fusion_vi_gradient_weight*tf.reduce_mean(1-tf.image.ssim(self.labels_vi, self.fusion_image, 1.0))+ self.fusion_ir_gradient_weight*tf.reduce_mean(1-tf.image.ssim(self.labels_ir, self.fusion_image, 1.0))
        
        tf.summary.scalar('g_loss_2', self.g_loss_2)
        self.vi_loss = 100* self.vi_loss
        self.ir_loss = 100 * self.ir_loss
        self.g_loss_total=100*self.g_loss_2
        
        # tf.summary.scalar('int_ir',tf.reduce_mean(tf.square(self.fusion_image - self.labels_ir)))
        # tf.summary.scalar('int_vi',tf.reduce_mean(tf.square(self.fusion_image - self.labels_vi)))
        # tf.summary.scalar('gra_vi',tf.reduce_mean(tf.square(gradient(self.fusion_image) -gradient (self.labels_vi))))

        tf.summary.scalar('loss_g',self.g_loss_total)
    self.saver = tf.train.Saver(max_to_keep=100)

    # with tf.name_scope('image'):
    #     tf.summary.image('input_ir',tf.expand_dims(self.images_ir[1,:,:,:],0))  
    #     tf.summary.image('input_vi',tf.expand_dims(self.images_vi[1,:,:,:],0))  
    #     tf.summary.image('fusion_image',tf.expand_dims(self.fusion_image[1,:,:,:],0))   
    
  def train(self, config):
    if config.is_train:
      input_setup(self.sess, config,"../Multi-exp/Train_ir")
      input_setup(self.sess,config,"../Multi-exp/Train_vi")
    else:
      nx_ir, ny_ir = input_setup(self.sess, config,"../Multi-exp/Test_ir")
      nx_vi,ny_vi=input_setup(self.sess, config,"../Multi-exp/Test_vi")

    if config.is_train:     
      data_dir_ir = os.path.join('./{}'.format(config.checkpoint_dir), "../Multi-exp/Train_ir","train.h5")
      data_dir_vi = os.path.join('./{}'.format(config.checkpoint_dir), "../Multi-exp/Train_vi","train.h5")
    else:
      data_dir_ir = os.path.join('./{}'.format(config.checkpoint_dir),"../Multi-exp/Test_ir", "test.h5")
      data_dir_vi = os.path.join('./{}'.format(config.checkpoint_dir),"../Multi-exp/Test_vi", "test.h5")

    train_data_ir, train_label_ir = read_data(data_dir_ir)
    train_data_vi, train_label_vi = read_data(data_dir_vi)
    #找训练时更新的变量组（判决器和生成器是分开训练的，所以要找到对应的变量）
    t_vars = tf.compat.v1.trainable_variables()
    self.ir_encoder_vars = [var for var in t_vars if 'ir_encoder' in var.name]
    self.ir_decoder_vars = [var for var in t_vars if 'ir_decoder' in var.name]
    self.vi_encoder_vars = [var for var in t_vars if 'vi_encoder' in var.name]
    self.vi_decoder_vars = [var for var in t_vars if 'vi_decoder' in var.name]
    self.g_vars = [var for var in t_vars if 'fusion_model' in var.name]
    # print(self.g_vars)
    # clip_ops = []
    # for var in self.d_vars:
        # clip_bounds = [-.01, .01]
        # clip_ops.append(
            # tf.assign(
                # var, 
                # tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
            # )
        # )
    # self.clip_disc_weights = tf.group(*clip_ops)
    # Stochastic gradient descent with the standard backpropagation
    with tf.name_scope('train_step'):
        self.train_fusion_op = tf.compat.v1.train.AdamOptimizer(config.learning_rate).minimize(self.g_loss_total,var_list=[self.g_vars, self.ir_encoder_vars, self.ir_decoder_vars,self.vi_encoder_vars,self.vi_decoder_vars])
        # self.train_fusion_op = tf.compat.v1.train.AdamOptimizer(config.learning_rate).minimize(self.g_loss_total,var_list=[self.ir_encoder_vars, self.ir_decoder_vars,self.vi_encoder_vars,self.vi_decoder_vars])
        #self.train_discriminator_op=tf.train.AdamOptimizer(config.learning_rate).minimize(self.d_loss,var_list=self.d_vars)
    #将所有统计的量合起来
    self.summary_op = tf.summary.merge_all()
    #生成日志文件
    self.train_writer = tf.summary.FileWriter(config.summary_dir + '/train',self.sess.graph,flush_secs=60)
    
    # tf.initialize_all_variables().run()
    tf.global_variables_initializer().run()
    
    counter = 0
    start_time = time.time()

    # if self.load(self.checkpoint_dir):
      # print(" [*] Load SUCCESS")
    # else:
      # print(" [!] Load failed...")

    if config.is_train:
      print("Training...")

      for ep in range(config.epoch):
        # Run by batch images
        batch_idxs = len(train_data_ir) // config.batch_size
        for idx in range(0, batch_idxs):
          batch_images_ir = train_data_ir[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_labels_ir = train_label_ir[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_images_vi = train_data_vi[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_labels_vi = train_label_vi[idx*config.batch_size : (idx+1)*config.batch_size]

          counter += 1
          #for i in range(2):
           # _, err_d= self.sess.run([self.train_discriminator_op, self.d_loss], feed_dict={self.images_ir: batch_images_ir, self.images_vi: batch_images_vi, self.labels_vi: batch_labels_vi,self.labels_ir:batch_labels_ir})
            # self.sess.run(self.clip_disc_weights)
          _, err_g, err_ir, err_vi,summary_str= self.sess.run([self.train_fusion_op, self.g_loss_total,self.ir_loss, self.vi_loss,self.summary_op], feed_dict={self.images_ir: batch_images_ir, self.images_vi: batch_images_vi, self.labels_ir: batch_labels_ir,self.labels_vi:batch_labels_vi})
          #将统计的量写到日志文件里
          self.train_writer.add_summary(summary_str,counter)

          if counter % 10 == 0:
            print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss_g:[%.8f], loss_ir:[%.8f], loss_vi:[%0.8f]" \
              % ((ep+1), counter, time.time()-start_time, err_g, err_ir, err_vi))
            #print(a)

        self.save(config.checkpoint_dir, ep)

    else:
      print("Testing...")

      result = self.fusion_image.eval(feed_dict={self.images_ir: train_data_ir, self.labels_ir: train_label_ir,self.images_vi: train_data_vi, self.labels_vi: train_label_vi})
      result=result*127.5+127.5
      result = merge(result, [nx_ir, ny_ir])
      result = result.squeeze()
      image_path = os.path.join(os.getcwd(), config.sample_dir)
      image_path = os.path.join(image_path, "test_image.png")
      imsave(result, image_path)

  def ir_encoder(self, img_ir):
    # with tf.variable_scope('ir_encoder'):
    with tf.variable_scope('ir_encoder'):
      with tf.variable_scope('ir_layer1'):
        weighs = tf.compat.v1.get_variable('ir_w1',[5,5,1,16], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.compat.v1.get_variable('ir_b1',[16],initializer=tf.constant_initializer(0.0))
        layer1 = tf.contrib.layers.batch_norm(tf.nn.conv2d(img_ir,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
        layer1 = lrelu(layer1)
      
      # concat1 = tf.concat([img_ir, layer1],axis=-1)

      with tf.variable_scope('ir_layer2'):
        weighs = tf.get_variable('ir_w2',[3,3,16,16], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('ir_b2',[16],initializer=tf.constant_initializer(0.0))
        layer2 = tf.contrib.layers.batch_norm(tf.nn.conv2d(layer1,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
        layer2 = lrelu(layer2)
      
      concat1 = tf.concat([layer1,layer2],axis=-1)

      with tf.variable_scope('ir_layer3'):
        weighs = tf.get_variable('ir_w3',[3,3,32,16], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('ir_b3',[16],initializer=tf.constant_initializer(0.0))
        layer3 = tf.contrib.layers.batch_norm(tf.nn.conv2d(concat1,weighs,strides=[1,1,1,1],padding='SAME')+bias,decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
        layer3 = lrelu(layer3)

      concat2 = tf.concat([layer1, layer2, layer3],axis=-1)

      with tf.variable_scope('ir_layer4'):
        weighs = tf.get_variable('ir_w4',[3,3,48,16], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('ir_b4',[16],initializer=tf.constant_initializer(0.0))
        layer4 = tf.contrib.layers.batch_norm(tf.nn.conv2d(concat2,weighs,strides=[1,1,1,1],padding='SAME')+bias,decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
        layer4 = lrelu(layer4)

      concat3 = tf.concat([layer1, layer2, layer3, layer4], axis=-1)

      with tf.variable_scope('ir_layer5'):
        weighs = tf.get_variable('ir_w5',[3,3,64,16], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('ir_b5',[16],initializer=tf.constant_initializer(0.0))
        layer5 = tf.contrib.layers.batch_norm(tf.nn.conv2d(concat3,weighs,strides=[1,1,1,1],padding='SAME')+bias,decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
        layer5 = lrelu(layer5)

      concat4 = tf.concat([layer1, layer2, layer3, layer4, layer5], axis=-1)

      with tf.variable_scope('ir_intensity'):
        weights = tf.get_variable('ir_intensity_w',[1,1,80,1], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('ir_intensity_b',[1],initializer=tf.constant_initializer(0.0))
        intensity = tf.nn.conv2d(concat4,weights,strides=[1,1,1,1],padding='SAME')+bias
        intensity = lrelu(intensity)
      
      with tf.variable_scope('ir_res'):
        weights = tf.get_variable('ir_res_w',[1,1,80,1], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('ir_res_b',[1],initializer=tf.constant_initializer(0.0))
        res = tf.nn.conv2d(concat4,weights,strides=[1,1,1,1],padding='SAME')+bias
        res = lrelu(res)

    return intensity, res

  def vi_encoder(self, img_vi):
    # with tf.variable_scope('vi_encoder'):
    with tf.variable_scope('vi_encoder'):
      with tf.variable_scope('vi_layer1'):
        weighs = tf.compat.v1.get_variable('vi_w1',[5,5,1,16], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.compat.v1.get_variable('vi_b1',[16],initializer=tf.constant_initializer(0.0))
        layer1 = tf.contrib.layers.batch_norm(tf.nn.conv2d(img_vi,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
        layer1 = lrelu(layer1)
      
      # concat1 = tf.concat([img_vi, layer1],axis=-1)

      with tf.variable_scope('vi_layer2'):
        weighs = tf.get_variable('vi_w2',[3,3,16,16], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('vi_b2',[16],initializer=tf.constant_initializer(0.0))
        layer2 = tf.contrib.layers.batch_norm(tf.nn.conv2d(layer1,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
        layer2 = lrelu(layer2)
      
      concat1 = tf.concat([layer1,layer2],axis=-1)

      with tf.variable_scope('vi_layer3'):
        weighs = tf.get_variable('vi_w3',[3,3,32,16], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('vi_b3',[16],initializer=tf.constant_initializer(0.0))
        layer3 = tf.contrib.layers.batch_norm(tf.nn.conv2d(concat1,weighs,strides=[1,1,1,1],padding='SAME')+bias,decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
        layer3 = lrelu(layer3)

      concat2 = tf.concat([layer1, layer2, layer3],axis=-1)

      with tf.variable_scope('vi_layer4'):
        weighs = tf.get_variable('vi_w4',[3,3,48,16], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('vi_b4',[16],initializer=tf.constant_initializer(0.0))
        layer4 = tf.contrib.layers.batch_norm(tf.nn.conv2d(concat2,weighs,strides=[1,1,1,1],padding='SAME')+bias,decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
        layer4 = lrelu(layer4)

      concat3 = tf.concat([layer1, layer2, layer3, layer4], axis=-1)

      with tf.variable_scope('vi_layer5'):
        weighs = tf.get_variable('vi_w5',[3,3,64,16], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('vi_b5',[16],initializer=tf.constant_initializer(0.0))
        layer5 = tf.contrib.layers.batch_norm(tf.nn.conv2d(concat3,weighs,strides=[1,1,1,1],padding='SAME')+bias,decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
        layer5 = lrelu(layer5)

      concat4 = tf.concat([layer1, layer2, layer3, layer4, layer5], axis=-1)

      with tf.variable_scope('vi_gradient'):
        weights = tf.get_variable('vi_gradient_w',[1,1,80,1], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('vi_gradient_b',[1],initializer=tf.constant_initializer(0.0))
        gradient = tf.nn.conv2d(concat4,weights,strides=[1,1,1,1],padding='SAME')+bias
        gradient = lrelu(gradient)
      
      with tf.variable_scope('vi_res'):
        weights = tf.get_variable('vi_res_w',[1,1,80,1], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('vi_res_b',[1],initializer=tf.constant_initializer(0.0))
        res = tf.nn.conv2d(concat4,weights,strides=[1,1,1,1],padding='SAME')+bias
        res = lrelu(res)

    return gradient, res


  def fusion_model(self,img_ir,img_vi):
####################  Layer1  ###########################
    with tf.variable_scope('fusion_model'):
        with tf.variable_scope('layer1'):
            weights=tf.get_variable("w1",[5,5,3,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b1",[16],initializer=tf.constant_initializer(0.0))
            conv1_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(img_ir, weights, strides=[1,1,1,1], padding='SAME') + bias,decay=0.9, updates_collections=None, epsilon=1e-5, scale=True )
            conv1_ir = lrelu(conv1_ir)   
        with tf.variable_scope('layer1_vi'):
            weights=tf.get_variable("w1_vi",[5,5,3,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b1_vi",[16],initializer=tf.constant_initializer(0.0))
            conv1_vi= tf.contrib.layers.batch_norm(tf.nn.conv2d(img_vi, weights, strides=[1,1,1,1], padding='SAME') + bias,decay=0.9, updates_collections=None, epsilon=1e-5, scale=True )
            conv1_vi = lrelu(conv1_vi)           
            

####################  Layer2  ###########################            
        with tf.variable_scope('layer2'):
            weights=tf.get_variable("w2",[3,3,16,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b2",[16],initializer=tf.constant_initializer(0.0))
            conv2_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv1_ir, weights, strides=[1,1,1,1], padding='SAME') + bias,decay=0.9, updates_collections=None, epsilon=1e-5, scale=True )
            conv2_ir = lrelu(conv2_ir)         
            
        with tf.variable_scope('layer2_vi'):
            weights=tf.get_variable("w2_vi",[3,3,16,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b2_vi",[16],initializer=tf.constant_initializer(0.0))
            conv2_vi= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv1_vi, weights, strides=[1,1,1,1], padding='SAME') + bias,decay=0.9, updates_collections=None, epsilon=1e-5, scale=True )
            conv2_vi = lrelu(conv2_vi)            
            

        conv_2_midle =tf.concat([conv2_ir,conv2_vi],axis=-1)    
       
  
        with tf.variable_scope('layer2_3'):
            weights=tf.get_variable("w2_3",[1,1,32,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b2_3",[16],initializer=tf.constant_initializer(0.0))
            conv2_3_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv_2_midle, weights, strides=[1,1,1,1], padding='SAME') + bias,decay=0.9, updates_collections=None, epsilon=1e-5, scale=True )
            conv2_3_ir = lrelu(conv2_3_ir)   
                    
                       
        with tf.variable_scope('layer2_3_vi'):
            weights=tf.get_variable("w2_3_vi",[1,1,32,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b2_3_vi",[16],initializer=tf.constant_initializer(0.0))
            conv2_3_vi= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv_2_midle, weights, strides=[1,1,1,1], padding='SAME') + bias,decay=0.9, updates_collections=None, epsilon=1e-5, scale=True )
            conv2_3_vi = lrelu(conv2_3_vi)       
            
            
####################  Layer3  ###########################               
        conv_12_ir=tf.concat([conv1_ir,conv2_ir,conv2_3_ir],axis=-1)
        conv_12_vi=tf.concat([conv1_vi,conv2_vi,conv2_3_vi],axis=-1)        
            
        with tf.variable_scope('layer3'):
            weights=tf.get_variable("w3",[3,3,48,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b3",[16],initializer=tf.constant_initializer(0.0))
            conv3_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv_12_ir, weights, strides=[1,1,1,1], padding='SAME') + bias,decay=0.9, updates_collections=None, epsilon=1e-5, scale=True )
            conv3_ir =lrelu(conv3_ir)
        with tf.variable_scope('layer3_vi'):
            weights=tf.get_variable("w3_vi",[3,3,48,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b3_vi",[16],initializer=tf.constant_initializer(0.0))
            conv3_vi= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv_12_vi, weights, strides=[1,1,1,1], padding='SAME') + bias,decay=0.9, updates_collections=None, epsilon=1e-5, scale=True )
            conv3_vi = lrelu(conv3_vi)
            

        conv_3_midle =tf.concat([conv3_ir,conv3_vi],axis=-1)    
       
  
        with tf.variable_scope('layer3_4'):
            weights=tf.get_variable("w3_4",[1,1,32,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b3_4",[16],initializer=tf.constant_initializer(0.0))
            conv3_4_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv_3_midle, weights, strides=[1,1,1,1], padding='SAME') + bias,decay=0.9, updates_collections=None, epsilon=1e-5, scale=True )
            conv3_4_ir = lrelu(conv3_4_ir)   
                    
                       
        with tf.variable_scope('layer3_4_vi'):
            weights=tf.get_variable("w3_4_vi",[1,1,32,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b3_4_vi",[16],initializer=tf.constant_initializer(0.0))
            conv3_4_vi= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv_3_midle, weights, strides=[1,1,1,1], padding='SAME') + bias,decay=0.9, updates_collections=None, epsilon=1e-5, scale=True )
            conv3_4_vi = lrelu(conv3_4_vi)  


            
####################  Layer4  ########################### 
        conv_123_ir=tf.concat([conv1_ir,conv2_ir,conv3_ir,conv3_4_ir],axis=-1)
        conv_123_vi=tf.concat([conv1_vi,conv2_vi,conv3_vi,conv3_4_vi],axis=-1)                   
            
        with tf.variable_scope('layer4'):
            weights=tf.get_variable("w4",[3,3,64,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b4",[16],initializer=tf.constant_initializer(0.0))
            conv4_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv_123_ir, weights, strides=[1,1,1,1], padding='SAME') + bias,decay=0.9, updates_collections=None, epsilon=1e-5, scale=True )
            conv4_ir = lrelu(conv4_ir)
        with tf.variable_scope('layer4_vi'):
            weights=tf.get_variable("w4_vi",[3,3,64,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b4_vi",[16],initializer=tf.constant_initializer(0.0))
            conv4_vi= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv_123_vi, weights, strides=[1,1,1,1], padding='SAME') + bias,decay=0.9, updates_collections=None, epsilon=1e-5, scale=True )
            conv4_vi = lrelu(conv4_vi)
            
 
        conv_ir_vi =tf.concat([conv1_ir,conv1_vi,conv2_ir,conv2_vi,conv3_ir,conv3_vi,conv4_ir,conv4_vi],axis=-1)
 
        
        with tf.variable_scope('layer5'):
            weights=tf.get_variable("w5",[1,1,128,1],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b5",[1],initializer=tf.constant_initializer(0.0))
            conv5_ir= tf.nn.conv2d(conv_ir_vi, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv5_ir=tf.nn.tanh(conv5_ir)
    return conv5_ir
    
  def save(self, checkpoint_dir, step):
    model_name = "CGAN.model"
    # model_dir = "%s_%s" % ("0414", "1")
    model_dir = self.store_dir
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
                    os.path.join(checkpoint_dir, model_name),
                    global_step=step)

  def load(self, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    model_dir = self.store_dir
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        print(ckpt_name)
        self.saver.restore(self.sess, os.path.join(checkpoint_dir,ckpt_name))
        return True
    else:
        return False

  

  
