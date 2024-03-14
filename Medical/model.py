# # -*- coding: utf-8 -*-

# from tensorflow.python.ops.init_ops_v2 import Initializer
# from utils import (
#   read_data, 
#   input_setup, 
#   imsave,
#   merge,
#   gradient,
#   lrelu,
#   weights_spectral_norm,
#   l2_norm
# )

# import time
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
# # import matplotlib.pyplot as plt

# # import numpy as np
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
# tf.disable_eager_execution()

# class CGAN(object):

#   def __init__(self, 
#                sess, 
#                image_size=132,
#                label_size=120,
#                batch_size=32,
#                c_dim=1, 
#                checkpoint_dir=None, 
#                sample_dir=None):

#     self.sess = sess
#     self.is_grayscale = (c_dim == 1)
#     self.image_size = image_size
#     self.label_size = label_size
#     self.batch_size = batch_size

#     self.c_dim = c_dim

#     self.checkpoint_dir = checkpoint_dir
#     self.sample_dir = sample_dir
#     self.build_model()

#   def build_model(self):
#     with tf.name_scope('IR_input'):
#         #红外图像patch
#         self.images_ir = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images_ir')
#         self.labels_ir = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name='labels_ir')
#     with tf.name_scope('VI_input'):
#         #可见光图像patch
#         self.images_vi = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images_vi')
#         self.labels_vi = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name='labels_vi')
#         #self.labels_vi_gradient=gradient(self.labels_vi)
#     #将红外和可见光图像在通道方向连起来，第一通道是红外图像，第二通道是可见光图像
#     with tf.name_scope('input'):
#         #self.resize_ir=tf.image.resize_images(self.images_ir, (self.image_size, self.image_size), method=2)
#         self.ir_intensity, self.ir_gradient = self.ir_encoder(self.labels_ir)
#         self.vi_intensity, self.vi_gradient = self.vi_encoder(self.labels_vi)
#         ir_concat = tf.concat([self.ir_intensity, self.ir_gradient], axis=-1)
#         vi_concat = tf.concat([self.vi_intensity, self.vi_gradient], axis=-1)
#         self.ir_fusion = self.ir_decoder(ir_concat)
#         self.vi_fusion = self.vi_decoder(vi_concat)
#         self.input_intensity =tf.concat([self.ir_intensity,self.ir_intensity,self.vi_intensity],axis=-1)
#         self.input_gradient =tf.concat([self.vi_gradient,self.vi_gradient,self.ir_gradient],axis=-1)
#         # self.input_image_ir =tf.concat([self.labels_ir,self.labels_ir,self.labels_vi],axis=-1)
#         # self.input_image_vi =tf.concat([self.labels_vi,self.labels_vi,self.labels_ir],axis=-1)
#     #self.pred=tf.clip_by_value(tf.sign(self.pred_ir-self.pred_vi),0,1)
#     #融合图像
#     with tf.name_scope('fusion'): 
#         self.fusion_image=self.fusion_model(self.input_intensity,self.input_gradient)

#     with tf.name_scope('g_loss'):
#         #self.g_loss_1=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=neg, labels=tf.ones_like(neg)))
#         #self.g_loss_1=tf.reduce_mean(tf.square(neg-tf.ones_like(pos)))
#         #self.g_loss_1=tf.reduce_mean(tf.square(neg-tf.random_uniform(shape=[self.batch_size,1],minval=0.7,maxval=1.2,dtype=tf.float32)))
#         #tf.summary.scalar('g_loss_1',self.g_loss_1)
#         #self.g_loss_2=tf.reduce_mean(tf.square(self.fusion_image - self.labels_ir))
#         # print(self.fusion_image.shape, self.labels_ir.shape)
#         vi_loss = 10*tf.reduce_mean(tf.square(self.vi_fusion - self.labels_vi)) + 50 * tf.reduce_mean(tf.square(gradient(self.vi_fusion)-gradient(self.labels_vi))) + 50 * tf.reduce_mean(tf.square(gradient(self.labels_vi)-self.vi_gradient)) + 10*tf.reduce_mean(tf.square(self.vi_intensity) - self.labels_vi)
#         ir_loss = 10*tf.reduce_mean(tf.square(self.ir_fusion - self.labels_ir)) + 1 * tf.reduce_mean(tf.square(gradient(self.ir_fusion)-gradient(self.labels_ir))) + 10 * tf.reduce_mean(tf.square(self.labels_ir-self.ir_intensity))
#         self.g_loss_2=vi_loss + ir_loss + tf.reduce_mean(tf.square(self.fusion_image - self.labels_ir))+1*tf.reduce_mean(tf.square(self.fusion_image - self.labels_vi))+300*tf.reduce_mean(tf.square(gradient(self.fusion_image) - gradient(self.labels_vi)))+0*tf.reduce_mean(tf.square(gradient(self.fusion_image) - self.ir_gradient))
#         tf.summary.scalar('g_loss_2',self.g_loss_2)

#         self.g_loss_total=100*self.g_loss_2
        
        
#         tf.summary.scalar('int_ir',tf.reduce_mean(tf.square(self.fusion_image - self.labels_ir)))
#         tf.summary.scalar('int_vi',tf.reduce_mean(tf.square(self.fusion_image - self.labels_vi)))
#         tf.summary.scalar('gra_vi',tf.reduce_mean(tf.square(gradient(self.fusion_image) -gradient (self.labels_vi))))

#         tf.summary.scalar('loss_g',self.g_loss_total)
#     self.saver = tf.train.Saver(max_to_keep=50)

#     with tf.name_scope('image'):
#         tf.summary.image('input_ir',tf.expand_dims(self.images_ir[1,:,:,:],0))  
#         tf.summary.image('input_vi',tf.expand_dims(self.images_vi[1,:,:,:],0))  
#         tf.summary.image('fusion_image',tf.expand_dims(self.fusion_image[1,:,:,:],0))   
    
#   def train(self, config):
#     if config.is_train:
#       input_setup(self.sess, config,"./Train_ir")
#       input_setup(self.sess,config,"./Train_vi")
#     else:
#       nx_ir, ny_ir = input_setup(self.sess, config,"./Test_ir")
#       nx_vi,ny_vi=input_setup(self.sess, config,"./Test_vi")

#     if config.is_train:     
#       data_dir_ir = os.path.join('./{}'.format(config.checkpoint_dir), "./Train_ir","train.h5")
#       data_dir_vi = os.path.join('./{}'.format(config.checkpoint_dir), "./Train_vi","train.h5")
#     else:
#       data_dir_ir = os.path.join('./{}'.format(config.checkpoint_dir),"./Test_ir", "test.h5")
#       data_dir_vi = os.path.join('./{}'.format(config.checkpoint_dir),"./Test_vi", "test.h5")

#     train_data_ir, train_label_ir = read_data(data_dir_ir)
#     train_data_vi, train_label_vi = read_data(data_dir_vi)
#     #找训练时更新的变量组（判决器和生成器是分开训练的，所以要找到对应的变量）
#     t_vars = tf.trainable_variables()
#     self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
#     print(self.d_vars)
#     self.g_vars = [var for var in t_vars if 'fusion_model' in var.name]
#     print(self.g_vars)
#     # clip_ops = []
#     # for var in self.d_vars:
#         # clip_bounds = [-.01, .01]
#         # clip_ops.append(
#             # tf.assign(
#                 # var, 
#                 # tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
#             # )
#         # )
#     # self.clip_disc_weights = tf.group(*clip_ops)
#     # Stochastic gradient descent with the standard backpropagation
#     with tf.name_scope('train_step'):
#         self.train_fusion_op = tf.train.AdamOptimizer(config.learning_rate).minimize(self.g_loss_total,var_list=self.g_vars)
#         #self.train_discriminator_op=tf.train.AdamOptimizer(config.learning_rate).minimize(self.d_loss,var_list=self.d_vars)
#     #将所有统计的量合起来
#     self.summary_op = tf.summary.merge_all()
#     #生成日志文件
#     self.train_writer = tf.summary.FileWriter(config.summary_dir + '/train',self.sess.graph,flush_secs=60)
    
#     tf.initialize_all_variables().run()
    
#     counter = 0
#     start_time = time.time()

#     # if self.load(self.checkpoint_dir):
#       # print(" [*] Load SUCCESS")
#     # else:
#       # print(" [!] Load failed...")

#     if config.is_train:
#       print("Training...")

#       for ep in range(config.epoch):
#         # Run by batch images
#         batch_idxs = len(train_data_ir) // config.batch_size
#         for idx in range(0, batch_idxs):
#           batch_images_ir = train_data_ir[idx*config.batch_size : (idx+1)*config.batch_size]
#           batch_labels_ir = train_label_ir[idx*config.batch_size : (idx+1)*config.batch_size]
#           batch_images_vi = train_data_vi[idx*config.batch_size : (idx+1)*config.batch_size]
#           batch_labels_vi = train_label_vi[idx*config.batch_size : (idx+1)*config.batch_size]

#           counter += 1
#           #for i in range(2):
#            # _, err_d= self.sess.run([self.train_discriminator_op, self.d_loss], feed_dict={self.images_ir: batch_images_ir, self.images_vi: batch_images_vi, self.labels_vi: batch_labels_vi,self.labels_ir:batch_labels_ir})
#             # self.sess.run(self.clip_disc_weights)
#           _, err_g,summary_str= self.sess.run([self.train_fusion_op, self.g_loss_total,self.summary_op], feed_dict={self.images_ir: batch_images_ir, self.images_vi: batch_images_vi, self.labels_ir: batch_labels_ir,self.labels_vi:batch_labels_vi})
#           #将统计的量写到日志文件里
#           self.train_writer.add_summary(summary_str,counter)

#           if counter % 10 == 0:
#             print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss_g:[%.8f]" \
#               % ((ep+1), counter, time.time()-start_time, err_g))
#             #print(a)

#         self.save(config.checkpoint_dir, ep)

#     else:
#       print("Testing...")

#       result = self.fusion_image.eval(feed_dict={self.images_ir: train_data_ir, self.labels_ir: train_label_ir,self.images_vi: train_data_vi, self.labels_vi: train_label_vi})
#       result=result*127.5+127.5
#       result = merge(result, [nx_ir, ny_ir])
#       result = result.squeeze()
#       image_path = os.path.join(os.getcwd(), config.sample_dir)
#       image_path = os.path.join(image_path, "test_image.png")
#       imsave(result, image_path)

#   def ir_encoder(self, img_ir):
#     # with tf.variable_scope('ir_encoder'):
#     with tf.variable_scope('ir_block1'):
#       with tf.variable_scope('ir_block1_stride2'):
#         weighs = tf.get_variable('block1_w1_ir',[3,3,1,64], initializer=tf.truncated_normal_initializer(stddev=1e-3))
#         bias = tf.get_variable('block1_b1_ir',[64],initializer=tf.constant_initializer(0.0))
#         block1_stride = tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(img_ir,weighs,strides=[1,1,1,1],padding='SAME')+bias,        )
#         block1_stride = lrelu(block1_stride)
#       with tf.variable_scope('ir_block1_layer2'):
#         weighs = tf.get_variable('block1_w2_ir',[3,3,64,64], initializer=tf.truncated_normal_initializer(stddev=1e-3))
#         bias = tf.get_variable('block2_b2_ir',[64],initializer=tf.constant_initializer(0.0))
#         block1_conv2 = tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(block1_stride,weighs,strides=[1,1,1,1],padding='SAME')+bias,        )
#         block1_conv2 = lrelu(block1_conv2)
#       with tf.variable_scope('ir_block1_layer3'):
#         weighs = tf.get_variable('block1_w3_ir',[3,3,64,64], initializer=tf.truncated_normal_initializer(stddev=1e-3))
#         bias = tf.get_variable('block1_b3_ir',[64],initializer=tf.constant_initializer(0.0))
#         block1_conv3 = tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(block1_conv2,weighs,strides=[1,1,1,1],padding='SAME')+bias,        )
#         block1_conv3 = lrelu(block1_conv3)

#     with tf.variable_scope('ir_block2'):
#       with tf.variable_scope('ir_block2_stride2'):
#         weighs = tf.get_variable('block2_w1_ir',[3,3,64,128], initializer=tf.truncated_normal_initializer(stddev=1e-3))
#         bias = tf.get_variable('block2_b1_ir',[128],initializer=tf.constant_initializer(0.0))
#         block2_stride = tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(block1_conv3,weighs,strides=[1,1,1,1],padding='SAME')+bias,        )
#         block2_stride = lrelu(block2_stride)
#       with tf.variable_scope('ir_block2_layer2'):
#         weighs = tf.get_variable('block2_w2_ir',[3,3,128,128], initializer=tf.truncated_normal_initializer(stddev=1e-3))
#         bias = tf.get_variable('block2_b2_ir',[128],initializer=tf.constant_initializer(0.0))
#         block2_conv2 = tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(block2_stride,weighs,strides=[1,1,1,1],padding='SAME')+bias,        )
#         block2_conv2 = lrelu(block2_conv2)
#       with tf.variable_scope('ir_block2_layer3'):
#         weighs = tf.get_variable('block2_w3_ir',[3,3,128,128], initializer=tf.truncated_normal_initializer(stddev=1e-3))
#         bias = tf.get_variable('block2_b3_ir',[128],initializer=tf.constant_initializer(0.0))
#         block2_conv3 = tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(block2_conv2,weighs,strides=[1,1,1,1],padding='SAME')+bias,        )
#         block2_conv3 = lrelu(block2_conv3)
#       with tf.variable_scope('ir_block2_layer4'):
#         weighs = tf.get_variable('block2_w4_ir',[3,3,128,128], initializer=tf.truncated_normal_initializer(stddev=1e-3))
#         bias = tf.get_variable('block2_b4_ir',[128],initializer=tf.constant_initializer(0.0))
#         block2_conv4 = tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(block2_conv3,weighs,strides=[1,1,1,1],padding='SAME')+bias,        )
#         block2_conv4 = lrelu(block2_conv4)
#       with tf.variable_scope('ir_block2_layer5'):
#         weighs = tf.get_variable('block2_w5_ir',[3,3,128,128], initializer=tf.truncated_normal_initializer(stddev=1e-3))
#         bias = tf.get_variable('block2_b5_ir',[128],initializer=tf.constant_initializer(0.0))
#         block2_conv5 = tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(block2_conv4,weighs,strides=[1,1,1,1],padding='SAME')+bias,        )
#         block2_conv5 = lrelu(block2_conv5)

#     with tf.variable_scope('ir_block3'):
#       with tf.variable_scope('ir_block3_stride2'):
#         weighs = tf.get_variable('block3_w1_ir',[3,3,128,256], initializer=tf.truncated_normal_initializer(stddev=1e-3))
#         bias = tf.get_variable('block3_b1_ir',[256],initializer=tf.constant_initializer(0.0))
#         block3_stride = tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(block2_conv5,weighs,strides=[1,1,1,1],padding='SAME')+bias,        )
#         block3_stride = lrelu(block3_stride)
#       with tf.variable_scope('ir_block3_layer2'):
#         weighs = tf.get_variable('block3_w2_ir',[3,3,256,256], initializer=tf.truncated_normal_initializer(stddev=1e-3))
#         bias = tf.get_variable('block3_b2_ir',[256],initializer=tf.constant_initializer(0.0))
#         block3_conv2 = tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(block3_stride,weighs,strides=[1,1,1,1],padding='SAME')+bias,        )
#         block3_conv2 = lrelu(block3_conv2)
#       with tf.variable_scope('ir_block3_layer3'):
#         weighs = tf.get_variable('block3_w3_ir',[3,3,256,256], initializer=tf.truncated_normal_initializer(stddev=1e-3))
#         bias = tf.get_variable('block3_b3_ir',[256],initializer=tf.constant_initializer(0.0))
#         block3_conv3 = tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(block3_conv2,weighs,strides=[1,1,1,1],padding='SAME')+bias,        )
#         block3_conv3 = lrelu(block3_conv3)
#       with tf.variable_scope('ir_block3_layer4'):
#         weighs = tf.get_variable('block3_w4_ir',[3,3,256,256], initializer=tf.truncated_normal_initializer(stddev=1e-3))
#         bias = tf.get_variable('b4_ir',[256],initializer=tf.constant_initializer(0.0))
#         block3_conv4 = tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(block3_conv3,weighs,strides=[1,1,1,1],padding='SAME')+bias,        )
#         block3_conv4 = lrelu(block3_conv4)
#       with tf.variable_scope('ir_block3_layer5'):
#         weighs = tf.get_variable('block3_w5_ir',[3,3,256,256], initializer=tf.truncated_normal_initializer(stddev=1e-3))
#         bias = tf.get_variable('block3_b5_ir',[256],initializer=tf.constant_initializer(0.0))
#         block3_conv5 = tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(block3_conv4,weighs,strides=[1,1,1,1],padding='SAME')+bias,        )
#         block3_conv5 = lrelu(block3_conv5)
#       with tf.variable_scope('ir_block3_layer6'):
#         weighs = tf.get_variable('block3_w6_ir',[3,3,256,256], initializer=tf.truncated_normal_initializer(stddev=1e-3))
#         bias = tf.get_variable('block3_b6_ir',[256],initializer=tf.constant_initializer(0.0))
#         block3_conv6 = tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(block3_conv5,weighs,strides=[1,1,1,1],padding='SAME')+bias,        )
#         block3_conv6 = lrelu(block3_conv6)
#       with tf.variable_scope('ir_block3_layer7'):
#         weighs = tf.get_variable('block3_w7_ir',[3,3,256,256], initializer=tf.truncated_normal_initializer(stddev=1e-3))
#         bias = tf.get_variable('block3_b7_ir',[256],initializer=tf.constant_initializer(0.0))
#         block3_conv7 = tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(block3_conv6,weighs,strides=[1,1,1,1],padding='SAME')+bias,        )
#         block3_conv7 = lrelu(block3_conv7)
#       with tf.variable_scope('ir_block3_layer8'):
#         weighs = tf.get_variable('block3_w8_ir',[3,3,256,256], initializer=tf.truncated_normal_initializer(stddev=1e-3))
#         bias = tf.get_variable('block3_b8_ir',[256],initializer=tf.constant_initializer(0.0))
#         block3_conv8 = tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(block3_conv7,weighs,strides=[1,1,1,1],padding='SAME')+bias,        )
#         block3_conv8 = lrelu(block3_conv8)
#       with tf.variable_scope('ir_block3_layer9'):
#         weighs = tf.get_variable('block3_w9_ir',[3,3,256,256], initializer=tf.truncated_normal_initializer(stddev=1e-3))
#         bias = tf.get_variable('block3_b9_ir',[256],initializer=tf.constant_initializer(0.0))
#         block3_conv9 = tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(block3_conv8,weighs,strides=[1,1,1,1],padding='SAME')+bias,        )
#         block3_conv9 = lrelu(block3_conv9)

#     with tf.variable_scope('ir_block4'):
#       with tf.variable_scope('ir_block4_stride2'):
#         weighs = tf.get_variable('block4_w1_ir',[3,3,256,512], initializer=tf.truncated_normal_initializer(stddev=1e-3))
#         bias = tf.get_variable('block4_b1_ir',[512],initializer=tf.constant_initializer(0.0))
#         block4_stride = tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(block3_conv9,weighs,strides=[1,1,1,1],padding='SAME')+bias,        )
#         block4_stride = lrelu(block4_stride)
#       with tf.variable_scope('ir_block4_layer2'):
#         weighs = tf.get_variable('block4_w2_ir',[3,3,512,512], initializer=tf.truncated_normal_initializer(stddev=1e-3))
#         bias = tf.get_variable('block4_b2_ir',[512],initializer=tf.constant_initializer(0.0))
#         block4_conv2 = tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(block4_stride,weighs,strides=[1,1,1,1],padding='SAME')+bias,        )
#         block4_conv2 = lrelu(block4_conv2)
#       with tf.variable_scope('ir_block4_layer3'):
#         weighs = tf.get_variable('block4_w3_ir',[3,3,512,512], initializer=tf.truncated_normal_initializer(stddev=1e-3))
#         bias = tf.get_variable('block4_b3_ir',[512],initializer=tf.constant_initializer(0.0))
#         block4_conv3 = tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(block4_conv2,weighs,strides=[1,1,1,1],padding='SAME')+bias,        )
#         block4_conv3 = lrelu(block4_conv3)

#     with tf.variable_scope('ir_intensity'):
#       intensity_concat = tf.concat([block1_conv3, block2_conv5, block3_conv9, block4_conv3],axis=-1)
#       with tf.variable_scope('ir_abstract_intensity'):
#         weighs = tf.get_variable('ir_layer_intensity_w',[3,3,960,64], initializer=tf.truncated_normal_initializer(stddev=1e-3))
#         bias = tf.get_variable('ir_layer_intensity_b',[64],initializer=tf.constant_initializer(0.0))
#         intensity1 = tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(intensity_concat,weighs,strides=[1,1,1,1],padding='SAME')+bias,        )
#         intensity1 = lrelu(intensity1)
      
#       with tf.variable_scope('ir_intensity_single'):
#         weighs = tf.get_variable('ir_intensity_w',[3,3,64,1], initializer=tf.truncated_normal_initializer(stddev=1e-3))
#         bias = tf.get_variable('ir_intensity_b',[1],initializer=tf.constant_initializer(0.0))
#         intensity = tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(intensity1,weighs,strides=[1,1,1,1],padding='SAME')+bias,        )
#         intensity = lrelu(intensity)

#     with tf.variable_scope('ir_gradient'):
#       gradient_concat = tf.concat([block1_conv3, block2_conv5, block3_conv9, block4_conv3], axis=-1)
#       with tf.variable_scope('ir_abstract_gradient'):
#         weighs = tf.get_variable('ir_layer_gradient_w',[3,3,960,64], initializer=tf.truncated_normal_initializer(stddev=1e-3))
#         bias = tf.get_variable('ir_layer_gradient_b',[64],initializer=tf.constant_initializer(0.0))
#         gradient1 = tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(gradient_concat,weighs,strides=[1,1,1,1],padding='SAME')+bias,        )
#         gradient1 = lrelu(gradient1)
#       with tf.variable_scope('ir_gradient_single'):
#         weighs = tf.get_variable('ir_gradient_w',[3,3,64,1], initializer=tf.truncated_normal_initializer(stddev=1e-3))
#         bias = tf.get_variable('ir_gradient_b',[1],initializer=tf.constant_initializer(0.0))
#         gradient = tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(gradient1,weighs,strides=[1,1,1,1],padding='SAME')+bias,        )
#         gradient = lrelu(gradient)
    
#     return intensity, gradient

#   def vi_encoder(self, img_vi):
#     # with tf.variable_scope('vi_encoder'):
#     with tf.variable_scope('vi_block1'):
#       with tf.variable_scope('vi_block1_stride2'):
#         weighs = tf.get_variable('block1_w1_vi',[3,3,1,64], initializer=tf.truncated_normal_initializer(stddev=1e-3))
#         bias = tf.get_variable('block1_b1_vi',[64],initializer=tf.constant_initializer(0.0))
#         block1_stride = tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(img_vi,weighs,strides=[1,1,1,1],padding='SAME')+bias,        )
#         block1_stride = lrelu(block1_stride)
#       with tf.variable_scope('vi_block1_layer2'):
#         weighs = tf.get_variable('block1_w2_vi',[3,3,64,64], initializer=tf.truncated_normal_initializer(stddev=1e-3))
#         bias = tf.get_variable('block2_b2_vi',[64],initializer=tf.constant_initializer(0.0))
#         block1_conv2 = tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(block1_stride,weighs,strides=[1,1,1,1],padding='SAME')+bias,        )
#         block1_conv2 = lrelu(block1_conv2)
#       with tf.variable_scope('vi_block1_layer3'):
#         weighs = tf.get_variable('block1_w3_vi',[3,3,64,64], initializer=tf.truncated_normal_initializer(stddev=1e-3))
#         bias = tf.get_variable('block1_b3_vi',[64],initializer=tf.constant_initializer(0.0))
#         block1_conv3 = tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(block1_conv2,weighs,strides=[1,1,1,1],padding='SAME')+bias,        )
#         block1_conv3 = lrelu(block1_conv3)

#     with tf.variable_scope('vi_block2'):
#       with tf.variable_scope('vi_block2_stride2'):
#         weighs = tf.get_variable('block2_w1_vi',[3,3,64,128], initializer=tf.truncated_normal_initializer(stddev=1e-3))
#         bias = tf.get_variable('block2_b1_vi',[128],initializer=tf.constant_initializer(0.0))
#         block2_stride = tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(block1_conv3,weighs,strides=[1,1,1,1],padding='SAME')+bias,        )
#         block2_stride = lrelu(block2_stride)
#       with tf.variable_scope('vi_block2_layer2'):
#         weighs = tf.get_variable('block2_w2_vi',[3,3,128,128], initializer=tf.truncated_normal_initializer(stddev=1e-3))
#         bias = tf.get_variable('block2_b2_vi',[128],initializer=tf.constant_initializer(0.0))
#         block2_conv2 = tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(block2_stride,weighs,strides=[1,1,1,1],padding='SAME')+bias,        )
#         block2_conv2 = lrelu(block2_conv2)
#       with tf.variable_scope('vi_block2_layer3'):
#         weighs = tf.get_variable('block2_w3_vi',[3,3,128,128], initializer=tf.truncated_normal_initializer(stddev=1e-3))
#         bias = tf.get_variable('block2_b3_vi',[128],initializer=tf.constant_initializer(0.0))
#         block2_conv3 = tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(block2_conv2,weighs,strides=[1,1,1,1],padding='SAME')+bias,        )
#         block2_conv3 = lrelu(block2_conv3)
#       with tf.variable_scope('vi_block2_layer4'):
#         weighs = tf.get_variable('block2_w4_vi',[3,3,128,128], initializer=tf.truncated_normal_initializer(stddev=1e-3))
#         bias = tf.get_variable('block2_b4_vi',[128],initializer=tf.constant_initializer(0.0))
#         block2_conv4 = tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(block2_conv3,weighs,strides=[1,1,1,1],padding='SAME')+bias,        )
#         block2_conv4 = lrelu(block2_conv4)
#       with tf.variable_scope('vi_block2_layer5'):
#         weighs = tf.get_variable('block2_w5_vi',[3,3,128,128], initializer=tf.truncated_normal_initializer(stddev=1e-3))
#         bias = tf.get_variable('block2_b5_vi',[128],initializer=tf.constant_initializer(0.0))
#         block2_conv5 = tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(block2_conv4,weighs,strides=[1,1,1,1],padding='SAME')+bias,        )
#         block2_conv5 = lrelu(block2_conv5)

#     with tf.variable_scope('vi_block3'):
#       with tf.variable_scope('vi_block3_stride2'):
#         weighs = tf.get_variable('block3_w1_vi',[3,3,128,256], initializer=tf.truncated_normal_initializer(stddev=1e-3))
#         bias = tf.get_variable('block3_b1_vi',[256],initializer=tf.constant_initializer(0.0))
#         block3_stride = tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(block2_conv5,weighs,strides=[1,1,1,1],padding='SAME')+bias,        )
#         block3_stride = lrelu(block3_stride)
#       with tf.variable_scope('vi_block3_layer2'):
#         weighs = tf.get_variable('block3_w2_vi',[3,3,256,256], initializer=tf.truncated_normal_initializer(stddev=1e-3))
#         bias = tf.get_variable('block3_b2_vi',[256],initializer=tf.constant_initializer(0.0))
#         block3_conv2 = tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(block3_stride,weighs,strides=[1,1,1,1],padding='SAME')+bias,        )
#         block3_conv2 = lrelu(block3_conv2)
#       with tf.variable_scope('vi_block3_layer3'):
#         weighs = tf.get_variable('block3_w3_vi',[3,3,256,256], initializer=tf.truncated_normal_initializer(stddev=1e-3))
#         bias = tf.get_variable('block3_b3_vi',[256],initializer=tf.constant_initializer(0.0))
#         block3_conv3 = tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(block3_conv2,weighs,strides=[1,1,1,1],padding='SAME')+bias,        )
#         block3_conv3 = lrelu(block3_conv3)
#       with tf.variable_scope('vi_block3_layer4'):
#         weighs = tf.get_variable('block3_w4_vi',[3,3,256,256], initializer=tf.truncated_normal_initializer(stddev=1e-3))
#         bias = tf.get_variable('b4_vi',[256],initializer=tf.constant_initializer(0.0))
#         block3_conv4 = tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(block3_conv3,weighs,strides=[1,1,1,1],padding='SAME')+bias,        )
#         block3_conv4 = lrelu(block3_conv4)
#       with tf.variable_scope('vi_block3_layer5'):
#         weighs = tf.get_variable('block3_w5_vi',[3,3,256,256], initializer=tf.truncated_normal_initializer(stddev=1e-3))
#         bias = tf.get_variable('block3_b5_vi',[256],initializer=tf.constant_initializer(0.0))
#         block3_conv5 = tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(block3_conv4,weighs,strides=[1,1,1,1],padding='SAME')+bias,        )
#         block3_conv5 = lrelu(block3_conv5)
#       with tf.variable_scope('vi_block3_layer6'):
#         weighs = tf.get_variable('block3_w6_vi',[3,3,256,256], initializer=tf.truncated_normal_initializer(stddev=1e-3))
#         bias = tf.get_variable('block3_b6_vi',[256],initializer=tf.constant_initializer(0.0))
#         block3_conv6 = tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(block3_conv5,weighs,strides=[1,1,1,1],padding='SAME')+bias,        )
#         block3_conv6 = lrelu(block3_conv6)
#       with tf.variable_scope('vi_block3_layer7'):
#         weighs = tf.get_variable('block3_w7_vi',[3,3,256,256], initializer=tf.truncated_normal_initializer(stddev=1e-3))
#         bias = tf.get_variable('block3_b7_vi',[256],initializer=tf.constant_initializer(0.0))
#         block3_conv7 = tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(block3_conv6,weighs,strides=[1,1,1,1],padding='SAME')+bias,        )
#         block3_conv7 = lrelu(block3_conv7)
#       with tf.variable_scope('vi_block3_layer8'):
#         weighs = tf.get_variable('block3_w8_vi',[3,3,256,256], initializer=tf.truncated_normal_initializer(stddev=1e-3))
#         bias = tf.get_variable('block3_b8_vi',[256],initializer=tf.constant_initializer(0.0))
#         block3_conv8 = tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(block3_conv7,weighs,strides=[1,1,1,1],padding='SAME')+bias,        )
#         block3_conv8 = lrelu(block3_conv8)
#       with tf.variable_scope('vi_block3_layer9'):
#         weighs = tf.get_variable('block3_w9_vi',[3,3,256,256], initializer=tf.truncated_normal_initializer(stddev=1e-3))
#         bias = tf.get_variable('block3_b9_vi',[256],initializer=tf.constant_initializer(0.0))
#         block3_conv9 = tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(block3_conv8,weighs,strides=[1,1,1,1],padding='SAME')+bias,        )
#         block3_conv9 = lrelu(block3_conv9)

#     with tf.variable_scope('vi_block4'):
#       with tf.variable_scope('vi_block4_stride2'):
#         weighs = tf.get_variable('block4_w1_vi',[3,3,256,512], initializer=tf.truncated_normal_initializer(stddev=1e-3))
#         bias = tf.get_variable('block4_b1_vi',[512],initializer=tf.constant_initializer(0.0))
#         block4_stride = tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(block3_conv9,weighs,strides=[1,1,1,1],padding='SAME')+bias,        )
#         block4_stride = lrelu(block4_stride)
#       with tf.variable_scope('vi_block4_layer2'):
#         weighs = tf.get_variable('block4_w2_vi',[3,3,512,512], initializer=tf.truncated_normal_initializer(stddev=1e-3))
#         bias = tf.get_variable('block4_b2_vi',[512],initializer=tf.constant_initializer(0.0))
#         block4_conv2 = tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(block4_stride,weighs,strides=[1,1,1,1],padding='SAME')+bias,        )
#         block4_conv2 = lrelu(block4_conv2)
#       with tf.variable_scope('vi_block4_layer3'):
#         weighs = tf.get_variable('block4_w3_vi',[3,3,512,512], initializer=tf.truncated_normal_initializer(stddev=1e-3))
#         bias = tf.get_variable('block4_b3_vi',[512],initializer=tf.constant_initializer(0.0))
#         block4_conv3 = tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(block4_conv2,weighs,strides=[1,1,1,1],padding='SAME')+bias,        )
#         block4_conv3 = lrelu(block4_conv3)

#     with tf.variable_scope('vi_intensity'):
#       intensity_concat = tf.concat([block1_conv3, block2_conv5, block3_conv9, block4_conv3],axis=-1)
#       with tf.variable_scope('vi_abstract_intensity'):
#         weighs = tf.get_variable('vi_layer_intensity_w',[3,3,960,64], initializer=tf.truncated_normal_initializer(stddev=1e-3))
#         bias = tf.get_variable('vi_layer_intensity_b',[64],initializer=tf.constant_initializer(0.0))
#         intensity1 = tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(intensity_concat,weighs,strides=[1,1,1,1],padding='SAME')+bias,        )
#         intensity1 = lrelu(intensity1)
#       with tf.variable_scope('vi_intensity_single'):
#         weighs = tf.get_variable('vi_intensity_w',[3,3,64,1], initializer=tf.truncated_normal_initializer(stddev=1e-3))
#         bias = tf.get_variable('vi_intensity_b',[1],initializer=tf.constant_initializer(0.0))
#         intensity = tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(intensity1,weighs,strides=[1,1,1,1],padding='SAME')+bias,        )
#         intensity = lrelu(intensity)

#     with tf.variable_scope('vi_gradient'):
#       gradient_concat = tf.concat([block1_conv3, block2_conv5, block3_conv9, block4_conv3], axis=-1)
#       with tf.variable_scope('vi_abstract_gradient'):
#         weighs = tf.get_variable('vi_layer_gradient_w',[3,3,960,64], initializer=tf.truncated_normal_initializer(stddev=1e-3))
#         bias = tf.get_variable('vi_layer_gradient_b',[64],initializer=tf.constant_initializer(0.0))
#         gradient1 = tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(gradient_concat,weighs,strides=[1,1,1,1],padding='SAME')+bias,        )
#         gradient1 = lrelu(gradient1)
#       with tf.variable_scope('vi_gradient_single'):
#         weights = tf.get_variable('vi_gradient_w',[3,3,64,1], initializer=tf.truncated_normal_initializer(stddev=1e-3))
#         bias = tf.get_variable('vi_gradient_b',[1],initializer=tf.constant_initializer(0.0))
#         gradient = tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(gradient1,weights,strides=[1,1,1,1],padding='SAME')+bias,        )
#         gradient = lrelu(gradient)
    
#     return intensity, gradient

#   def ir_decoder(self, ir_concat):
#     with tf.variable_scope('ir_decoder'):
#       with tf.variable_scope('ir_layer1'):
#         weights = tf.get_variable('ir_layer1_w', [2,2,2,64], initializer=tf.truncated_normal_initializer(stddev=1e-3))
#         bias = tf.get_variable('ir_layer1_b',[64],initializer=tf.constant_initializer(0.0))
#         conv1 = tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(ir_concat,weights,strides=[1,1,1,1],padding='SAME')+bias,        )
#         conv1 = lrelu(conv1)
#       with tf.variable_scope('ir_layer2'):
#         weights = tf.get_variable('ir_layer2_w', [2,2,64,16], initializer=tf.truncated_normal_initializer(stddev=1e-3))
#         bias = tf.get_variable('ir_layer2_b',[16],initializer=tf.constant_initializer(0.0))
#         conv2 = tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(conv1,weights,strides=[1,1,1,1],padding='SAME')+bias,        )
#         conv2 = lrelu(conv2)
#       with tf.variable_scope('ir_fusion'):
#         weights = tf.get_variable('ir_fusion_w', [2,2,16,1], initializer=tf.truncated_normal_initializer(stddev=1e-3))
#         bias = tf.get_variable('ir_fusion_b',[1],initializer=tf.constant_initializer(0.0))
#         ir_fusion = tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(conv2,weights,strides=[1,1,1,1],padding='SAME')+bias,        )
#         ir_fusion = lrelu(ir_fusion)
    
#     return ir_fusion

#   def vi_decoder(self, vi_concat):
#     with tf.variable_scope('vi_decoder'):
#       with tf.variable_scope('vi_layer1'):
#         weights = tf.get_variable('vi_layer1_w', [2,2,2,64], initializer=tf.truncated_normal_initializer(stddev=1e-3))
#         bias = tf.get_variable('vi_layer1_b',[64],initializer=tf.constant_initializer(0.0))
#         conv1 = tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(vi_concat,weights,strides=[1,1,1,1],padding='SAME')+bias,        )
#         conv1 = lrelu(conv1)
#       with tf.variable_scope('vi_layer2'):
#         weights = tf.get_variable('vi_layer2_w', [2,2,64,16], initializer=tf.truncated_normal_initializer(stddev=1e-3))
#         bias = tf.get_variable('vi_layer2_b',[16],initializer=tf.constant_initializer(0.0))
#         conv2 = tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(conv1,weights,strides=[1,1,1,1],padding='SAME')+bias,        )
#         conv2 = lrelu(conv2)
#       with tf.variable_scope('vi_fusion'):
#         weights = tf.get_variable('vi_fusion_w', [2,2,16,1], initializer=tf.truncated_normal_initializer(stddev=1e-3))
#         bias = tf.get_variable('vi_fusion_b',[1],initializer=tf.constant_initializer(0.0))
#         vi_fusion = tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(conv2,weights,strides=[1,1,1,1],padding='SAME')+bias,        )
#         vi_fusion = lrelu(vi_fusion)
    
#     return vi_fusion

#   def fusion_model(self,img_ir,img_vi):
# ####################  Layer1  ###########################
#     with tf.variable_scope('fusion_model'):
#         with tf.variable_scope('layer1'):
#             weights=tf.get_variable("w1",[5,5,3,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
#             bias=tf.get_variable("b1",[16],initializer=tf.constant_initializer(0.0))
#             conv1_ir= tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(img_ir, weights, strides=[1,1,1,1], padding='SAME') + bias,         )
#             conv1_ir = lrelu(conv1_ir)   
#         with tf.variable_scope('layer1_vi'):
#             weights=tf.get_variable("w1_vi",[5,5,3,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
#             bias=tf.get_variable("b1_vi",[16],initializer=tf.constant_initializer(0.0))
#             conv1_vi= tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(img_vi, weights, strides=[1,1,1,1], padding='SAME') + bias,         )
#             conv1_vi = lrelu(conv1_vi)           
            

# ####################  Layer2  ###########################            
#         with tf.variable_scope('layer2'):
#             weights=tf.get_variable("w2",[3,3,16,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
#             bias=tf.get_variable("b2",[16],initializer=tf.constant_initializer(0.0))
#             conv2_ir= tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(conv1_ir, weights, strides=[1,1,1,1], padding='SAME') + bias,         )
#             conv2_ir = lrelu(conv2_ir)         
            
#         with tf.variable_scope('layer2_vi'):
#             weights=tf.get_variable("w2_vi",[3,3,16,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
#             bias=tf.get_variable("b2_vi",[16],initializer=tf.constant_initializer(0.0))
#             conv2_vi= tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(conv1_vi, weights, strides=[1,1,1,1], padding='SAME') + bias,         )
#             conv2_vi = lrelu(conv2_vi)            
            

#         conv_2_midle =tf.concat([conv2_ir,conv2_vi],axis=-1)    
       
  
#         with tf.variable_scope('layer2_3'):
#             weights=tf.get_variable("w2_3",[1,1,32,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
#             bias=tf.get_variable("b2_3",[16],initializer=tf.constant_initializer(0.0))
#             conv2_3_ir= tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(conv_2_midle, weights, strides=[1,1,1,1], padding='SAME') + bias,         )
#             conv2_3_ir = lrelu(conv2_3_ir)   
                    
                       
#         with tf.variable_scope('layer2_3_vi'):
#             weights=tf.get_variable("w2_3_vi",[1,1,32,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
#             bias=tf.get_variable("b2_3_vi",[16],initializer=tf.constant_initializer(0.0))
#             conv2_3_vi= tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(conv_2_midle, weights, strides=[1,1,1,1], padding='SAME') + bias,         )
#             conv2_3_vi = lrelu(conv2_3_vi)       
            
            
# ####################  Layer3  ###########################               
#         conv_12_ir=tf.concat([conv1_ir,conv2_ir,conv2_3_ir],axis=-1)
#         conv_12_vi=tf.concat([conv1_vi,conv2_vi,conv2_3_vi],axis=-1)        
            
#         with tf.variable_scope('layer3'):
#             weights=tf.get_variable("w3",[3,3,48,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
#             bias=tf.get_variable("b3",[16],initializer=tf.constant_initializer(0.0))
#             conv3_ir= tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(conv_12_ir, weights, strides=[1,1,1,1], padding='SAME') + bias,         )
#             conv3_ir =lrelu(conv3_ir)
#         with tf.variable_scope('layer3_vi'):
#             weights=tf.get_variable("w3_vi",[3,3,48,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
#             bias=tf.get_variable("b3_vi",[16],initializer=tf.constant_initializer(0.0))
#             conv3_vi= tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(conv_12_vi, weights, strides=[1,1,1,1], padding='SAME') + bias,         )
#             conv3_vi = lrelu(conv3_vi)
            

#         conv_3_midle =tf.concat([conv3_ir,conv3_vi],axis=-1)    
       
  
#         with tf.variable_scope('layer3_4'):
#             weights=tf.get_variable("w3_4",[1,1,32,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
#             bias=tf.get_variable("b3_4",[16],initializer=tf.constant_initializer(0.0))
#             conv3_4_ir= tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(conv_3_midle, weights, strides=[1,1,1,1], padding='SAME') + bias,         )
#             conv3_4_ir = lrelu(conv3_4_ir)   
                    
                       
#         with tf.variable_scope('layer3_4_vi'):
#             weights=tf.get_variable("w3_4_vi",[1,1,32,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
#             bias=tf.get_variable("b3_4_vi",[16],initializer=tf.constant_initializer(0.0))
#             conv3_4_vi= tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(conv_3_midle, weights, strides=[1,1,1,1], padding='SAME') + bias,         )
#             conv3_4_vi = lrelu(conv3_4_vi)  


            
# ####################  Layer4  ########################### 
#         conv_123_ir=tf.concat([conv1_ir,conv2_ir,conv3_ir,conv3_4_ir],axis=-1)
#         conv_123_vi=tf.concat([conv1_vi,conv2_vi,conv3_vi,conv3_4_vi],axis=-1)                   
            
#         with tf.variable_scope('layer4'):
#             weights=tf.get_variable("w4",[3,3,64,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
#             bias=tf.get_variable("b4",[16],initializer=tf.constant_initializer(0.0))
#             conv4_ir= tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(conv_123_ir, weights, strides=[1,1,1,1], padding='SAME') + bias,         )
#             conv4_ir = lrelu(conv4_ir)
#         with tf.variable_scope('layer4_vi'):
#             weights=tf.get_variable("w4_vi",[3,3,64,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
#             bias=tf.get_variable("b4_vi",[16],initializer=tf.constant_initializer(0.0))
#             conv4_vi= tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(conv_123_vi, weights, strides=[1,1,1,1], padding='SAME') + bias,         )
#             conv4_vi = lrelu(conv4_vi)
            
 
#         conv_ir_vi =tf.concat([conv1_ir,conv1_vi,conv2_ir,conv2_vi,conv3_ir,conv3_vi,conv4_ir,conv4_vi],axis=-1)
 
        
#         with tf.variable_scope('layer5'):
#             weights=tf.get_variable("w5",[1,1,128,1],initializer=tf.truncated_normal_initializer(stddev=1e-3))
#             bias=tf.get_variable("b5",[1],initializer=tf.constant_initializer(0.0))
#             conv5_ir= tf.nn.conv2d(conv_ir_vi, weights, strides=[1,1,1,1], padding='SAME') + bias
#             conv5_ir=tf.nn.tanh(conv5_ir)
#     return conv5_ir
    
#   '''  
#   def discriminator(self,img,reuse,update_collection=None):
#     with tf.variable_scope('discriminator',reuse=reuse):
#         print(img.shape)
#         with tf.variable_scope('layer_1'):
#             weights=tf.get_variable("w_1",[3,3,1,32],initializer=tf.truncated_normal_initializer(stddev=1e-3))
#             weights=weights_spectral_norm(weights,update_collection=update_collection)
#             bias=tf.get_variable("b_1",[32],initializer=tf.constant_initializer(0.0))
#             conv1_vi=tf.nn.conv2d(img, weights, strides=[1,2,2,1], padding='VALID') + bias
#             conv1_vi = lrelu(conv1_vi)
#             #print(conv1_vi.shape)
#         with tf.variable_scope('layer_2'):
#             weights=tf.get_variable("w_2",[3,3,32,64],initializer=tf.truncated_normal_initializer(stddev=1e-3))
#             weights=weights_spectral_norm(weights,update_collection=update_collection)
#             bias=tf.get_variable("b_2",[64],initializer=tf.constant_initializer(0.0))
#             conv2_vi= tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(conv1_vi, weights, strides=[1,2,2,1], padding='VALID') + bias,         )
#             conv2_vi = lrelu(conv2_vi)
#             #print(conv2_vi.shape)
#         with tf.variable_scope('layer_3'):
#             weights=tf.get_variable("w_3",[3,3,64,128],initializer=tf.truncated_normal_initializer(stddev=1e-3))
#             weights=weights_spectral_norm(weights,update_collection=update_collection)
#             bias=tf.get_variable("b_3",[128],initializer=tf.constant_initializer(0.0))
#             conv3_vi= tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(conv2_vi, weights, strides=[1,2,2,1], padding='VALID') + bias,         )
#             conv3_vi=lrelu(conv3_vi)
#             #print(conv3_vi.shape)
#         with tf.variable_scope('layer_4'):
#             weights=tf.get_variable("w_4",[3,3,128,256],initializer=tf.truncated_normal_initializer(stddev=1e-3))
#             weights=weights_spectral_norm(weights,update_collection=update_collection)
#             bias=tf.get_variable("b_4",[256],initializer=tf.constant_initializer(0.0))
#             conv4_vi= tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(conv3_vi, weights, strides=[1,2,2,1], padding='VALID') + bias,         )
#             conv4_vi=lrelu(conv4_vi)
#             conv4_vi = tf.reshape(conv4_vi,[self.batch_size,6*6*256])
#         with tf.variable_scope('line_5'):
#             weights=tf.get_variable("w_5",[6*6*256,2],initializer=tf.truncated_normal_initializer(stddev=1e-3))
#             weights=weights_spectral_norm(weights,update_collection=update_collection)
#             bias=tf.get_variable("b_5",[2],initializer=tf.constant_initializer(0.0))
#             line_5=tf.matmul(conv4_vi, weights) + bias
#             #conv3_vi= tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(conv3_vi,         )
#     return line_5
#     '''
#   def save(self, checkpoint_dir, step):
#     model_name = "CGAN.model"
#     model_dir = "%s_%s" % ("CGAN", self.label_size)
#     checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

#     if not os.path.exists(checkpoint_dir):
#         os.makedirs(checkpoint_dir)

#     self.saver.save(self.sess,
#                     os.path.join(checkpoint_dir, model_name),
#                     global_step=step)

#   def load(self, checkpoint_dir):
#     print(" [*] Reading checkpoints...")
#     model_dir = "%s_%s" % ("CGAN", self.label_size)
#     checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

#     ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
#     if ckpt and ckpt.model_checkpoint_path:
#         ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
#         print(ckpt_name)
#         self.saver.restore(self.sess, os.path.join(checkpoint_dir,ckpt_name))
#         return True
#     else:
#         return False

# -*- coding: utf-8 -*-

# from tensorflow.python.ops.init_ops_v2 import Initializer
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
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
# import matplotlib.pyplot as plt

# import numpy as np
import tensorflow as tf
# tf.disable_v2_behavior()
# tf.disable_eager_execution()

class CGAN(object):

  def __init__(self, 
               sess, 
               image_size=132,
               label_size=120,
               batch_size=32,
               c_dim=1, 
               checkpoint_dir=None, 
               sample_dir=None):

    self.sess = sess
    self.is_grayscale = (c_dim == 1)
    self.image_size = image_size
    self.label_size = label_size
    self.batch_size = batch_size

    self.c_dim = c_dim

    self.checkpoint_dir = checkpoint_dir
    self.sample_dir = sample_dir
    self.build_model()

  def build_model(self):
    with tf.name_scope('IR_input'):
        #红外图像patch
        self.images_ir = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images_ir')
        self.labels_ir = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name='labels_ir')
    with tf.name_scope('VI_input'):
        #可见光图像patch
        self.images_vi = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images_vi')
        self.labels_vi = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name='labels_vi')
        #self.labels_vi_gradient=gradient(self.labels_vi)
    #将红外和可见光图像在通道方向连起来，第一通道是红外图像，第二通道是可见光图像
    with tf.name_scope('input'):
        #self.resize_ir=tf.image.resize_images(self.images_ir, (self.image_size, self.image_size), method=2)
        self.ir_intensity, self.ir_gradient = self.ir_encoder(self.labels_ir)
        self.vi_intensity, self.vi_gradient = self.vi_encoder(self.labels_vi)
        ir_concat = tf.concat([self.ir_intensity, self.ir_gradient], axis=-1)
        vi_concat = tf.concat([self.vi_intensity, self.vi_gradient], axis=-1)
        self.ir_fusion = self.ir_decoder(ir_concat)
        self.vi_fusion = self.vi_decoder(vi_concat)
        self.input_intensity =tf.concat([self.ir_intensity,self.ir_intensity,self.vi_intensity],axis=-1)
        self.input_gradient =tf.concat([self.vi_gradient,self.vi_gradient,self.ir_gradient],axis=-1)
        # self.input_image_ir =tf.concat([self.labels_ir,self.labels_ir,self.labels_vi],axis=-1)
        # self.input_image_vi =tf.concat([self.labels_vi,self.labels_vi,self.labels_ir],axis=-1)
    #self.pred=tf.clip_by_value(tf.sign(self.pred_ir-self.pred_vi),0,1)
    #融合图像
    with tf.name_scope('fusion'): 
        self.fusion_image=self.fusion_model(self.input_intensity,self.input_gradient)

    with tf.name_scope('g_loss'):
        #self.g_loss_1=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=neg, labels=tf.ones_like(neg)))
        #self.g_loss_1=tf.reduce_mean(tf.square(neg-tf.ones_like(pos)))
        #self.g_loss_1=tf.reduce_mean(tf.square(neg-tf.random_uniform(shape=[self.batch_size,1],minval=0.7,maxval=1.2,dtype=tf.float32)))
        #tf.summary.scalar('g_loss_1',self.g_loss_1)
        #self.g_loss_2=tf.reduce_mean(tf.square(self.fusion_image - self.labels_ir))
        # print(self.fusion_image.shape, self.labels_ir.shape)
        self.vi_loss = 10*tf.reduce_mean(tf.square(self.vi_fusion - self.labels_vi)) + 50 * tf.reduce_mean(tf.square(gradient(self.vi_fusion)-gradient(self.labels_vi))) + 50 * tf.reduce_mean(tf.square(gradient(self.labels_vi)-self.vi_gradient)) + 10*tf.reduce_mean(tf.square(self.vi_intensity) - self.labels_vi)
        self.ir_loss = 10*tf.reduce_mean(tf.square(self.ir_fusion - self.labels_ir)) + 1 * tf.reduce_mean(tf.square(gradient(self.ir_fusion)-gradient(self.labels_ir))) + 10 * tf.reduce_mean(tf.square(self.labels_ir-self.ir_intensity))
        self.g_loss_2=self.vi_loss + self.ir_loss + tf.reduce_mean(tf.square(self.fusion_image - self.labels_ir))+1*tf.reduce_mean(tf.square(self.fusion_image - self.labels_vi))+300*tf.reduce_mean(tf.square(gradient(self.fusion_image) - gradient(self.labels_vi)))+0*tf.reduce_mean(tf.square(gradient(self.fusion_image) - self.ir_gradient))
        tf.summary.scalar('g_loss_2',self.g_loss_2)

        self.g_loss_total=100*self.g_loss_2
        
        
        tf.summary.scalar('int_ir',tf.reduce_mean(tf.square(self.fusion_image - self.labels_ir)))
        tf.summary.scalar('int_vi',tf.reduce_mean(tf.square(self.fusion_image - self.labels_vi)))
        tf.summary.scalar('gra_vi',tf.reduce_mean(tf.square(gradient(self.fusion_image) -gradient (self.labels_vi))))

        tf.summary.scalar('loss_g',self.g_loss_total)
    self.saver = tf.train.Saver(max_to_keep=300)

    with tf.name_scope('image'):
        tf.summary.image('input_ir',tf.expand_dims(self.images_ir[1,:,:,:],0))  
        tf.summary.image('input_vi',tf.expand_dims(self.images_vi[1,:,:,:],0))  
        tf.summary.image('fusion_image',tf.expand_dims(self.fusion_image[1,:,:,:],0))   
    
    
  def train(self, config):
    if config.is_train:
      input_setup(self.sess, config,"Train_ir")
      input_setup(self.sess,config,"Train_vi")
    else:
      nx_ir, ny_ir = input_setup(self.sess, config,"Test_ir")
      nx_vi,ny_vi=input_setup(self.sess, config,"Test_vi")

    if config.is_train:     
      data_dir_ir = os.path.join('./{}'.format(config.checkpoint_dir), "Train_ir","train.h5")
      data_dir_vi = os.path.join('./{}'.format(config.checkpoint_dir), "Train_vi","train.h5")
    else:
      data_dir_ir = os.path.join('./{}'.format(config.checkpoint_dir),"Test_ir", "test.h5")
      data_dir_vi = os.path.join('./{}'.format(config.checkpoint_dir),"Test_vi", "test.h5")

    train_data_ir, train_label_ir = read_data(data_dir_ir)
    train_data_vi, train_label_vi = read_data(data_dir_vi)
    #找训练时更新的变量组（判决器和生成器是分开训练的，所以要找到对应的变量）
    t_vars = tf.trainable_variables()
    self.ir_encoder_vars = [var for var in t_vars if 'ir_encoder' in var.name]
    self.ir_decoder_vars = [var for var in t_vars if 'ir_decoder' in var.name]
    self.vi_encoder_vars = [var for var in t_vars if 'vi_encoder' in var.name]
    self.vi_decoder_vars = [var for var in t_vars if 'vi_decoder' in var.name]
    self.g_vars = [var for var in t_vars if 'fusion_model' in var.name]
    print(self.g_vars)
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
        #self.train_discriminator_op=tf.train.AdamOptimizer(config.learning_rate).minimize(self.d_loss,var_list=self.d_vars)
    #将所有统计的量合起来
    self.summary_op = tf.summary.merge_all()
    #生成日志文件
    self.train_writer = tf.summary.FileWriter(config.summary_dir + '/train',self.sess.graph,flush_secs=60)
    
    tf.initialize_all_variables().run()
    
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
    with tf.variable_scope('ir_block1'):
      with tf.variable_scope('ir_block1_stride2'):
        weights = tf.get_variable('block1_w1_ir',[3,3,1,64], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('block1_b1_ir',[64],initializer=tf.constant_initializer(0.0))
        block1_stride = tf.contrib.layers.batch_norm(tf.nn.conv2d(img_ir, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
        block1_stride = lrelu(block1_stride)
      with tf.variable_scope('ir_block1_layer2'):
        weighs = tf.get_variable('block1_w2_ir',[3,3,64,64], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('block2_b2_ir',[64],initializer=tf.constant_initializer(0.0))
        block1_conv2 = tf.contrib.layers.batch_norm(tf.nn.conv2d(block1_stride, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
        block1_conv2 = lrelu(block1_conv2)
      with tf.variable_scope('ir_block1_layer3'):
        weighs = tf.get_variable('block1_w3_ir',[3,3,64,64], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('block1_b3_ir',[64],initializer=tf.constant_initializer(0.0))
        block1_conv3 = tf.contrib.layers.batch_norm(tf.nn.conv2d(block1_conv2,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block1_conv3 = lrelu(block1_conv3)

    with tf.variable_scope('ir_block2'):
      with tf.variable_scope('ir_block2_stride2'):
        weighs = tf.get_variable('block2_w1_ir',[3,3,64,128], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('block2_b1_ir',[128],initializer=tf.constant_initializer(0.0))
        block2_stride = tf.contrib.layers.batch_norm(tf.nn.conv2d(block1_conv3,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block2_stride = lrelu(block2_stride)
      with tf.variable_scope('ir_block2_layer2'):
        weighs = tf.get_variable('block2_w2_ir',[3,3,128,128], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('block2_b2_ir',[128],initializer=tf.constant_initializer(0.0))
        block2_conv2 = tf.contrib.layers.batch_norm(tf.nn.conv2d(block2_stride,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block2_conv2 = lrelu(block2_conv2)
      with tf.variable_scope('ir_block2_layer3'):
        weighs = tf.get_variable('block2_w3_ir',[3,3,128,128], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('block2_b3_ir',[128],initializer=tf.constant_initializer(0.0))
        block2_conv3 = tf.contrib.layers.batch_norm(tf.nn.conv2d(block2_conv2,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block2_conv3 = lrelu(block2_conv3)
      with tf.variable_scope('ir_block2_layer4'):
        weighs = tf.get_variable('block2_w4_ir',[3,3,128,128], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('block2_b4_ir',[128],initializer=tf.constant_initializer(0.0))
        block2_conv4 = tf.contrib.layers.batch_norm(tf.nn.conv2d(block2_conv3,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block2_conv4 = lrelu(block2_conv4)
      with tf.variable_scope('ir_block2_layer5'):
        weighs = tf.get_variable('block2_w5_ir',[3,3,128,128], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('block2_b5_ir',[128],initializer=tf.constant_initializer(0.0))
        block2_conv5 = tf.contrib.layers.batch_norm(tf.nn.conv2d(block2_conv4,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block2_conv5 = lrelu(block2_conv5)

    with tf.variable_scope('ir_block3'):
      with tf.variable_scope('ir_block3_stride2'):
        weighs = tf.get_variable('block3_w1_ir',[3,3,128,256], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('block3_b1_ir',[256],initializer=tf.constant_initializer(0.0))
        block3_stride = tf.contrib.layers.batch_norm(tf.nn.conv2d(block2_conv5,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block3_stride = lrelu(block3_stride)
      with tf.variable_scope('ir_block3_layer2'):
        weighs = tf.get_variable('block3_w2_ir',[3,3,256,256], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('block3_b2_ir',[256],initializer=tf.constant_initializer(0.0))
        block3_conv2 = tf.contrib.layers.batch_norm(tf.nn.conv2d(block3_stride,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block3_conv2 = lrelu(block3_conv2)
      with tf.variable_scope('ir_block3_layer3'):
        weighs = tf.get_variable('block3_w3_ir',[3,3,256,256], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('block3_b3_ir',[256],initializer=tf.constant_initializer(0.0))
        block3_conv3 = tf.contrib.layers.batch_norm(tf.nn.conv2d(block3_conv2,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block3_conv3 = lrelu(block3_conv3)
      with tf.variable_scope('ir_block3_layer4'):
        weighs = tf.get_variable('block3_w4_ir',[3,3,256,256], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('b4_ir',[256],initializer=tf.constant_initializer(0.0))
        block3_conv4 = tf.contrib.layers.batch_norm(tf.nn.conv2d(block3_conv3,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block3_conv4 = lrelu(block3_conv4)
      with tf.variable_scope('ir_block3_layer5'):
        weighs = tf.get_variable('block3_w5_ir',[3,3,256,256], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('block3_b5_ir',[256],initializer=tf.constant_initializer(0.0))
        block3_conv5 = tf.contrib.layers.batch_norm(tf.nn.conv2d(block3_conv4,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block3_conv5 = lrelu(block3_conv5)
      with tf.variable_scope('ir_block3_layer6'):
        weighs = tf.get_variable('block3_w6_ir',[3,3,256,256], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('block3_b6_ir',[256],initializer=tf.constant_initializer(0.0))
        block3_conv6 = tf.contrib.layers.batch_norm(tf.nn.conv2d(block3_conv5,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block3_conv6 = lrelu(block3_conv6)
      with tf.variable_scope('ir_block3_layer7'):
        weighs = tf.get_variable('block3_w7_ir',[3,3,256,256], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('block3_b7_ir',[256],initializer=tf.constant_initializer(0.0))
        block3_conv7 = tf.contrib.layers.batch_norm(tf.nn.conv2d(block3_conv6,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block3_conv7 = lrelu(block3_conv7)
      with tf.variable_scope('ir_block3_layer8'):
        weighs = tf.get_variable('block3_w8_ir',[3,3,256,256], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('block3_b8_ir',[256],initializer=tf.constant_initializer(0.0))
        block3_conv8 = tf.contrib.layers.batch_norm(tf.nn.conv2d(block3_conv7,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block3_conv8 = lrelu(block3_conv8)
      with tf.variable_scope('ir_block3_layer9'):
        weighs = tf.get_variable('block3_w9_ir',[3,3,256,256], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('block3_b9_ir',[256],initializer=tf.constant_initializer(0.0))
        block3_conv9 = tf.contrib.layers.batch_norm(tf.nn.conv2d(block3_conv8,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block3_conv9 = lrelu(block3_conv9)

    with tf.variable_scope('ir_block4'):
      with tf.variable_scope('ir_block4_stride2'):
        weighs = tf.get_variable('block4_w1_ir',[3,3,256,512], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('block4_b1_ir',[512],initializer=tf.constant_initializer(0.0))
        block4_stride = tf.contrib.layers.batch_norm(tf.nn.conv2d(block3_conv9,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block4_stride = lrelu(block4_stride)
      with tf.variable_scope('ir_block4_layer2'):
        weighs = tf.get_variable('block4_w2_ir',[3,3,512,512], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('block4_b2_ir',[512],initializer=tf.constant_initializer(0.0))
        block4_conv2 = tf.contrib.layers.batch_norm(tf.nn.conv2d(block4_stride,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block4_conv2 = lrelu(block4_conv2)
      with tf.variable_scope('ir_block4_layer3'):
        weighs = tf.get_variable('block4_w3_ir',[3,3,512,512], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('block4_b3_ir',[512],initializer=tf.constant_initializer(0.0))
        block4_conv3 = tf.contrib.layers.batch_norm(tf.nn.conv2d(block4_conv2,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block4_conv3 = lrelu(block4_conv3)

    with tf.variable_scope('ir_intensity'):
      intensity_concat = tf.concat([block1_conv3, block2_conv5, block3_conv9, block4_conv3],axis=-1)
      with tf.variable_scope('ir_abstract_intensity'):
        weighs = tf.get_variable('ir_layer_intensity_w',[3,3,960,64], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('ir_layer_intensity_b',[64],initializer=tf.constant_initializer(0.0))
        intensity1 = tf.contrib.layers.batch_norm(tf.nn.conv2d(intensity_concat,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        intensity1 = lrelu(intensity1)
      
      with tf.variable_scope('ir_intensity_single'):
        weighs = tf.get_variable('ir_intensity_w',[3,3,64,1], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('ir_intensity_b',[1],initializer=tf.constant_initializer(0.0))
        intensity = tf.contrib.layers.batch_norm(tf.nn.conv2d(intensity1,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        intensity = lrelu(intensity)

    with tf.variable_scope('ir_gradient'):
      gradient_concat = tf.concat([block1_conv3, block2_conv5, block3_conv9, block4_conv3], axis=-1)
      with tf.variable_scope('ir_abstract_gradient'):
        weighs = tf.get_variable('ir_layer_gradient_w',[3,3,960,64], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('ir_layer_gradient_b',[64],initializer=tf.constant_initializer(0.0))
        gradient1 = tf.contrib.layers.batch_norm(tf.nn.conv2d(gradient_concat,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        gradient1 = lrelu(gradient1)
      with tf.variable_scope('ir_gradient_single'):
        weighs = tf.get_variable('ir_gradient_w',[3,3,64,1], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('ir_gradient_b',[1],initializer=tf.constant_initializer(0.0))
        gradient = tf.contrib.layers.batch_norm(tf.nn.conv2d(gradient1,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        gradient = lrelu(gradient)
    
    return intensity, gradient

  def vi_encoder(self, img_vi):
    # with tf.variable_scope('vi_encoder'):
    with tf.variable_scope('vi_block1'):
      with tf.variable_scope('vi_block1_stride2'):
        weighs = tf.get_variable('block1_w1_vi',[3,3,1,64], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('block1_b1_vi',[64],initializer=tf.constant_initializer(0.0))
        block1_stride = tf.contrib.layers.batch_norm(tf.nn.conv2d(img_vi,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block1_stride = lrelu(block1_stride)
      with tf.variable_scope('vi_block1_layer2'):
        weighs = tf.get_variable('block1_w2_vi',[3,3,64,64], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('block2_b2_vi',[64],initializer=tf.constant_initializer(0.0))
        block1_conv2 = tf.contrib.layers.batch_norm(tf.nn.conv2d(block1_stride,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block1_conv2 = lrelu(block1_conv2)
      with tf.variable_scope('vi_block1_layer3'):
        weighs = tf.get_variable('block1_w3_vi',[3,3,64,64], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('block1_b3_vi',[64],initializer=tf.constant_initializer(0.0))
        block1_conv3 = tf.contrib.layers.batch_norm(tf.nn.conv2d(block1_conv2,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block1_conv3 = lrelu(block1_conv3)

    with tf.variable_scope('vi_block2'):
      with tf.variable_scope('vi_block2_stride2'):
        weighs = tf.get_variable('block2_w1_vi',[3,3,64,128], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('block2_b1_vi',[128],initializer=tf.constant_initializer(0.0))
        block2_stride = tf.contrib.layers.batch_norm(tf.nn.conv2d(block1_conv3,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block2_stride = lrelu(block2_stride)
      with tf.variable_scope('vi_block2_layer2'):
        weighs = tf.get_variable('block2_w2_vi',[3,3,128,128], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('block2_b2_vi',[128],initializer=tf.constant_initializer(0.0))
        block2_conv2 = tf.contrib.layers.batch_norm(tf.nn.conv2d(block2_stride,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block2_conv2 = lrelu(block2_conv2)
      with tf.variable_scope('vi_block2_layer3'):
        weighs = tf.get_variable('block2_w3_vi',[3,3,128,128], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('block2_b3_vi',[128],initializer=tf.constant_initializer(0.0))
        block2_conv3 = tf.contrib.layers.batch_norm(tf.nn.conv2d(block2_conv2,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block2_conv3 = lrelu(block2_conv3)
      with tf.variable_scope('vi_block2_layer4'):
        weighs = tf.get_variable('block2_w4_vi',[3,3,128,128], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('block2_b4_vi',[128],initializer=tf.constant_initializer(0.0))
        block2_conv4 = tf.contrib.layers.batch_norm(tf.nn.conv2d(block2_conv3,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block2_conv4 = lrelu(block2_conv4)
      with tf.variable_scope('vi_block2_layer5'):
        weighs = tf.get_variable('block2_w5_vi',[3,3,128,128], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('block2_b5_vi',[128],initializer=tf.constant_initializer(0.0))
        block2_conv5 = tf.contrib.layers.batch_norm(tf.nn.conv2d(block2_conv4,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block2_conv5 = lrelu(block2_conv5)

    with tf.variable_scope('vi_block3'):
      with tf.variable_scope('vi_block3_stride2'):
        weighs = tf.get_variable('block3_w1_vi',[3,3,128,256], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('block3_b1_vi',[256],initializer=tf.constant_initializer(0.0))
        block3_stride = tf.contrib.layers.batch_norm(tf.nn.conv2d(block2_conv5,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block3_stride = lrelu(block3_stride)
      with tf.variable_scope('vi_block3_layer2'):
        weighs = tf.get_variable('block3_w2_vi',[3,3,256,256], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('block3_b2_vi',[256],initializer=tf.constant_initializer(0.0))
        block3_conv2 = tf.contrib.layers.batch_norm(tf.nn.conv2d(block3_stride,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block3_conv2 = lrelu(block3_conv2)
      with tf.variable_scope('vi_block3_layer3'):
        weighs = tf.get_variable('block3_w3_vi',[3,3,256,256], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('block3_b3_vi',[256],initializer=tf.constant_initializer(0.0))
        block3_conv3 = tf.contrib.layers.batch_norm(tf.nn.conv2d(block3_conv2,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block3_conv3 = lrelu(block3_conv3)
      with tf.variable_scope('vi_block3_layer4'):
        weighs = tf.get_variable('block3_w4_vi',[3,3,256,256], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('b4_vi',[256],initializer=tf.constant_initializer(0.0))
        block3_conv4 = tf.contrib.layers.batch_norm(tf.nn.conv2d(block3_conv3,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block3_conv4 = lrelu(block3_conv4)
      with tf.variable_scope('vi_block3_layer5'):
        weighs = tf.get_variable('block3_w5_vi',[3,3,256,256], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('block3_b5_vi',[256],initializer=tf.constant_initializer(0.0))
        block3_conv5 = tf.contrib.layers.batch_norm(tf.nn.conv2d(block3_conv4,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block3_conv5 = lrelu(block3_conv5)
      with tf.variable_scope('vi_block3_layer6'):
        weighs = tf.get_variable('block3_w6_vi',[3,3,256,256], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('block3_b6_vi',[256],initializer=tf.constant_initializer(0.0))
        block3_conv6 = tf.contrib.layers.batch_norm(tf.nn.conv2d(block3_conv5,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block3_conv6 = lrelu(block3_conv6)
      with tf.variable_scope('vi_block3_layer7'):
        weighs = tf.get_variable('block3_w7_vi',[3,3,256,256], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('block3_b7_vi',[256],initializer=tf.constant_initializer(0.0))
        block3_conv7 = tf.contrib.layers.batch_norm(tf.nn.conv2d(block3_conv6,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block3_conv7 = lrelu(block3_conv7)
      with tf.variable_scope('vi_block3_layer8'):
        weighs = tf.get_variable('block3_w8_vi',[3,3,256,256], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('block3_b8_vi',[256],initializer=tf.constant_initializer(0.0))
        block3_conv8 = tf.contrib.layers.batch_norm(tf.nn.conv2d(block3_conv7,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block3_conv8 = lrelu(block3_conv8)
      with tf.variable_scope('vi_block3_layer9'):
        weighs = tf.get_variable('block3_w9_vi',[3,3,256,256], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('block3_b9_vi',[256],initializer=tf.constant_initializer(0.0))
        block3_conv9 = tf.contrib.layers.batch_norm(tf.nn.conv2d(block3_conv8,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block3_conv9 = lrelu(block3_conv9)

    with tf.variable_scope('vi_block4'):
      with tf.variable_scope('vi_block4_stride2'):
        weighs = tf.get_variable('block4_w1_vi',[3,3,256,512], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('block4_b1_vi',[512],initializer=tf.constant_initializer(0.0))
        block4_stride = tf.contrib.layers.batch_norm(tf.nn.conv2d(block3_conv9,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block4_stride = lrelu(block4_stride)
      with tf.variable_scope('vi_block4_layer2'):
        weighs = tf.get_variable('block4_w2_vi',[3,3,512,512], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('block4_b2_vi',[512],initializer=tf.constant_initializer(0.0))
        block4_conv2 = tf.contrib.layers.batch_norm(tf.nn.conv2d(block4_stride,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block4_conv2 = lrelu(block4_conv2)
      with tf.variable_scope('vi_block4_layer3'):
        weighs = tf.get_variable('block4_w3_vi',[3,3,512,512], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('block4_b3_vi',[512],initializer=tf.constant_initializer(0.0))
        block4_conv3 = tf.contrib.layers.batch_norm(tf.nn.conv2d(block4_conv2,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        block4_conv3 = lrelu(block4_conv3)

    with tf.variable_scope('vi_intensity'):
      intensity_concat = tf.concat([block1_conv3, block2_conv5, block3_conv9, block4_conv3],axis=-1)
      with tf.variable_scope('vi_abstract_intensity'):
        weighs = tf.get_variable('vi_layer_intensity_w',[3,3,960,64], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('vi_layer_intensity_b',[64],initializer=tf.constant_initializer(0.0))
        intensity1 = tf.contrib.layers.batch_norm(tf.nn.conv2d(intensity_concat,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        intensity1 = lrelu(intensity1)
      with tf.variable_scope('vi_intensity_single'):
        weighs = tf.get_variable('vi_intensity_w',[3,3,64,1], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('vi_intensity_b',[1],initializer=tf.constant_initializer(0.0))
        intensity = tf.contrib.layers.batch_norm(tf.nn.conv2d(intensity1,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        intensity = lrelu(intensity)

    with tf.variable_scope('vi_gradient'):
      gradient_concat = tf.concat([block1_conv3, block2_conv5, block3_conv9, block4_conv3], axis=-1)
      with tf.variable_scope('vi_abstract_gradient'):
        weighs = tf.get_variable('vi_layer_gradient_w',[3,3,960,64], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('vi_layer_gradient_b',[64],initializer=tf.constant_initializer(0.0))
        gradient1 = tf.contrib.layers.batch_norm(tf.nn.conv2d(gradient_concat,weighs,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        gradient1 = lrelu(gradient1)
      with tf.variable_scope('vi_gradient_single'):
        weights = tf.get_variable('vi_gradient_w',[3,3,64,1], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('vi_gradient_b',[1],initializer=tf.constant_initializer(0.0))
        gradient = tf.contrib.layers.batch_norm(tf.nn.conv2d(gradient1,weights,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        gradient = lrelu(gradient)
    
    return intensity, gradient

  def ir_decoder(self, ir_concat):
    with tf.variable_scope('ir_decoder'):
      with tf.variable_scope('ir_layer1'):
        weights = tf.get_variable('ir_layer1_w', [2,2,2,64], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('ir_layer1_b',[64],initializer=tf.constant_initializer(0.0))
        conv1 = tf.contrib.layers.batch_norm(tf.nn.conv2d(ir_concat,weights,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        conv1 = lrelu(conv1)
      with tf.variable_scope('ir_layer2'):
        weights = tf.get_variable('ir_layer2_w', [2,2,64,16], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('ir_layer2_b',[16],initializer=tf.constant_initializer(0.0))
        conv2 = tf.contrib.layers.batch_norm(tf.nn.conv2d(conv1,weights,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        conv2 = lrelu(conv2)
      with tf.variable_scope('ir_fusion'):
        weights = tf.get_variable('ir_fusion_w', [2,2,16,1], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('ir_fusion_b',[1],initializer=tf.constant_initializer(0.0))
        ir_fusion = tf.contrib.layers.batch_norm(tf.nn.conv2d(conv2,weights,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        ir_fusion = lrelu(ir_fusion)
    
    return ir_fusion

  def vi_decoder(self, vi_concat):
    with tf.variable_scope('vi_decoder'):
      with tf.variable_scope('vi_layer1'):
        weights = tf.get_variable('vi_layer1_w', [2,2,2,64], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('vi_layer1_b',[64],initializer=tf.constant_initializer(0.0))
        conv1 = tf.contrib.layers.batch_norm(tf.nn.conv2d(vi_concat,weights,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        conv1 = lrelu(conv1)
      with tf.variable_scope('vi_layer2'):
        weights = tf.get_variable('vi_layer2_w', [2,2,64,16], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('vi_layer2_b',[16],initializer=tf.constant_initializer(0.0))
        conv2 = tf.contrib.layers.batch_norm(tf.nn.conv2d(conv1,weights,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        conv2 = lrelu(conv2)
      with tf.variable_scope('vi_fusion'):
        weights = tf.get_variable('vi_fusion_w', [2,2,16,1], initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable('vi_fusion_b',[1],initializer=tf.constant_initializer(0.0))
        vi_fusion = tf.contrib.layers.batch_norm(tf.nn.conv2d(conv2,weights,strides=[1,1,1,1],padding='SAME')+bias, decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
        vi_fusion = lrelu(vi_fusion)
    
    return vi_fusion

  def fusion_model(self,img_ir,img_vi):
####################  Layer1  ###########################
    with tf.variable_scope('fusion_model'):
        with tf.variable_scope('layer1'):
            weights=tf.get_variable("w1",[5,5,3,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b1",[16],initializer=tf.constant_initializer(0.0))
            conv1_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(img_ir, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv1_ir = lrelu(conv1_ir)   
        with tf.variable_scope('layer1_vi'):
            weights=tf.get_variable("w1_vi",[5,5,3,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b1_vi",[16],initializer=tf.constant_initializer(0.0))
            conv1_vi= tf.contrib.layers.batch_norm(tf.nn.conv2d(img_vi, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv1_vi = lrelu(conv1_vi)           
            

####################  Layer2  ###########################            
        with tf.variable_scope('layer2'):
            weights=tf.get_variable("w2",[3,3,16,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b2",[16],initializer=tf.constant_initializer(0.0))
            conv2_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv1_ir, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv2_ir = lrelu(conv2_ir)         
            
        with tf.variable_scope('layer2_vi'):
            weights=tf.get_variable("w2_vi",[3,3,16,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b2_vi",[16],initializer=tf.constant_initializer(0.0))
            conv2_vi= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv1_vi, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv2_vi = lrelu(conv2_vi)            
            

        conv_2_midle =tf.concat([conv2_ir,conv2_vi],axis=-1)    
       
  
        with tf.variable_scope('layer2_3'):
            weights=tf.get_variable("w2_3",[1,1,32,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b2_3",[16],initializer=tf.constant_initializer(0.0))
            conv2_3_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv_2_midle, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv2_3_ir = lrelu(conv2_3_ir)   
                    
                       
        with tf.variable_scope('layer2_3_vi'):
            weights=tf.get_variable("w2_3_vi",[1,1,32,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b2_3_vi",[16],initializer=tf.constant_initializer(0.0))
            conv2_3_vi= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv_2_midle, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv2_3_vi = lrelu(conv2_3_vi)       
            
            
####################  Layer3  ###########################               
        conv_12_ir=tf.concat([conv1_ir,conv2_ir,conv2_3_ir],axis=-1)
        conv_12_vi=tf.concat([conv1_vi,conv2_vi,conv2_3_vi],axis=-1)        
            
        with tf.variable_scope('layer3'):
            weights=tf.get_variable("w3",[3,3,48,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b3",[16],initializer=tf.constant_initializer(0.0))
            conv3_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv_12_ir, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv3_ir =lrelu(conv3_ir)
        with tf.variable_scope('layer3_vi'):
            weights=tf.get_variable("w3_vi",[3,3,48,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b3_vi",[16],initializer=tf.constant_initializer(0.0))
            conv3_vi= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv_12_vi, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv3_vi = lrelu(conv3_vi)
            

        conv_3_midle =tf.concat([conv3_ir,conv3_vi],axis=-1)    
       
  
        with tf.variable_scope('layer3_4'):
            weights=tf.get_variable("w3_4",[1,1,32,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b3_4",[16],initializer=tf.constant_initializer(0.0))
            conv3_4_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv_3_midle, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv3_4_ir = lrelu(conv3_4_ir)   
                    
                       
        with tf.variable_scope('layer3_4_vi'):
            weights=tf.get_variable("w3_4_vi",[1,1,32,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b3_4_vi",[16],initializer=tf.constant_initializer(0.0))
            conv3_4_vi= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv_3_midle, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv3_4_vi = lrelu(conv3_4_vi)  


            
####################  Layer4  ########################### 
        conv_123_ir=tf.concat([conv1_ir,conv2_ir,conv3_ir,conv3_4_ir],axis=-1)
        conv_123_vi=tf.concat([conv1_vi,conv2_vi,conv3_vi,conv3_4_vi],axis=-1)                   
            
        with tf.variable_scope('layer4'):
            weights=tf.get_variable("w4",[3,3,64,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b4",[16],initializer=tf.constant_initializer(0.0))
            conv4_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv_123_ir, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv4_ir = lrelu(conv4_ir)
        with tf.variable_scope('layer4_vi'):
            weights=tf.get_variable("w4_vi",[3,3,64,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b4_vi",[16],initializer=tf.constant_initializer(0.0))
            conv4_vi= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv_123_vi, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv4_vi = lrelu(conv4_vi)
            
 
        conv_ir_vi =tf.concat([conv1_ir,conv1_vi,conv2_ir,conv2_vi,conv3_ir,conv3_vi,conv4_ir,conv4_vi],axis=-1)
 
        
        with tf.variable_scope('layer5'):
            weights=tf.get_variable("w5",[1,1,128,1],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b5",[1],initializer=tf.constant_initializer(0.0))
            conv5_ir= tf.nn.conv2d(conv_ir_vi, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv5_ir=tf.nn.tanh(conv5_ir)
    return conv5_ir
    
  '''  
  def discriminator(self,img,reuse,update_collection=None):
    with tf.variable_scope('discriminator',reuse=reuse):
        print(img.shape)
        with tf.variable_scope('layer_1'):
            weights=tf.get_variable("w_1",[3,3,1,32],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights,update_collection=update_collection)
            bias=tf.get_variable("b_1",[32],initializer=tf.constant_initializer(0.0))
            conv1_vi=tf.nn.conv2d(img, weights, strides=[1,2,2,1], padding='VALID') + bias
            conv1_vi = lrelu(conv1_vi)
            #print(conv1_vi.shape)
        with tf.variable_scope('layer_2'):
            weights=tf.get_variable("w_2",[3,3,32,64],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights,update_collection=update_collection)
            bias=tf.get_variable("b_2",[64],initializer=tf.constant_initializer(0.0))
            conv2_vi= tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(conv1_vi, weights, strides=[1,2,2,1], padding='VALID') + bias,         )
            conv2_vi = lrelu(conv2_vi)
            #print(conv2_vi.shape)
        with tf.variable_scope('layer_3'):
            weights=tf.get_variable("w_3",[3,3,64,128],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights,update_collection=update_collection)
            bias=tf.get_variable("b_3",[128],initializer=tf.constant_initializer(0.0))
            conv3_vi= tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(conv2_vi, weights, strides=[1,2,2,1], padding='VALID') + bias,         )
            conv3_vi=lrelu(conv3_vi)
            #print(conv3_vi.shape)
        with tf.variable_scope('layer_4'):
            weights=tf.get_variable("w_4",[3,3,128,256],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights,update_collection=update_collection)
            bias=tf.get_variable("b_4",[256],initializer=tf.constant_initializer(0.0))
            conv4_vi= tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(tf.nn.conv2d(conv3_vi, weights, strides=[1,2,2,1], padding='VALID') + bias,         )
            conv4_vi=lrelu(conv4_vi)
            conv4_vi = tf.reshape(conv4_vi,[self.batch_size,6*6*256])
        with tf.variable_scope('line_5'):
            weights=tf.get_variable("w_5",[6*6*256,2],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights,update_collection=update_collection)
            bias=tf.get_variable("b_5",[2],initializer=tf.constant_initializer(0.0))
            line_5=tf.matmul(conv4_vi, weights) + bias
            #conv3_vi= tf.keras.layers. BatchNormalization( epsilon=1e-5, scale=True   )(conv3_vi,         )
    return line_5
    '''
  def save(self, checkpoint_dir, step):
    model_name = "CGAN.model"
    model_dir = "%s_%s" % ("0303", '1')
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
                    os.path.join(checkpoint_dir, model_name),
                    global_step=step)

  def load(self, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    model_dir = "%s_%s" % ("CGAN", self.label_size)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        print(ckpt_name)
        self.saver.restore(self.sess, os.path.join(checkpoint_dir,ckpt_name))
        return True
    else:
        return False




