# %%writefile main.py
# -*- coding: utf-8 -*-
from ast import parse
from ablation_model import CGAN

import tensorflow as tf
import pprint
import os
import argparse


os.environ['CUDA_VISIBLE_DECIVES']='0'

flags = tf.app.flags
flags.DEFINE_integer("epoch", 16, "Number of epoch [10]")
flags.DEFINE_integer("batch_size", 32, "The size of batch images [128]")
flags.DEFINE_integer("image_size", 132, "The size of image to use [33]")
flags.DEFINE_integer("label_size", 120, "The size of label to produce [21]")
flags.DEFINE_float("learning_rate", 1e-4, "The learning rate of gradient descent algorithm [1e-4]")
flags.DEFINE_integer("c_dim", 1, "Dimension of image color. [1]")
flags.DEFINE_integer("scale", 3, "The size of scale factor for preprocessing input image [3]")
flags.DEFINE_integer("stride", 14, "The size of stride to apply input image [14]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Name of checkpoint directory [checkpoint]")
flags.DEFINE_string("sample_dir", "sample", "Name of sample directory [sample]")
flags.DEFINE_string("summary_dir", "log", "Name of log directory [log]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [True]")
FLAGS = flags.FLAGS

def get_args():
  parser = argparse.ArgumentParser()
  # vi path loss weight
  parser.add_argument('--vi_gradient', type=float, default=1.8, help='vi_gradient weight')
  parser.add_argument('--vi_res', type=float, default=1.5, help='vi_res weight')
  parser.add_argument('--vi_fusion_gradient', type=float, default=1.8, help='vi fusion gradient weight')
  parser.add_argument('--vi_fusion_intensity', type=float, default=1.5, help='vi fusion intensity weight')
  # ir path loss weight
  parser.add_argument('--ir_intensity', type=float, default=1.5, help='ir_gradient weight')
  parser.add_argument('--ir_res', type=float, default=1.8, help='ir_res weight')
  parser.add_argument('--ir_fusion_intensity', type=float, default=1.5, help='ir fusion gradient weight')
  parser.add_argument('--ir_fusion_gradient', type=float, default=1.8, help='ir fusion intensity weight')
  # fusion image loss weight
  # parser.add_argument('--epochs', type=int, default=12)
  parser.add_argument('--vi_loss', type=float, default=1.0, help='vi path weight')
  parser.add_argument('--ir_loss', type=float, default=1.0, help='ir path weight')
  parser.add_argument('--fusion_vi_gradient', type=float, default=16.0, help='fusion vi gradient weight')
  parser.add_argument('--fusion_vi_intensity', type=float, default=1.5, help='fusion vi intensity weight')
  parser.add_argument('--fusion_ir_gradient', type=float, default=16.0, help='fusion  ir gradient weight')
  parser.add_argument('--fusion_ir_intensity', type=float, default=1.5, help='fusion ir intensity weight')
  parser.add_argument('--dir', type=str, default='auto',help='record')
  args, _ = parser.parse_known_args()
  return args

pp = pprint.PrettyPrinter()

def main(_):
  args = get_args()
  pp.pprint(flags.FLAGS.__flags)

  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)

  with tf.Session() as sess:
    srcnn = CGAN(sess, 
                  image_size=FLAGS.image_size, 
                  label_size=FLAGS.label_size, 
                  batch_size=FLAGS.batch_size,
                  c_dim=FLAGS.c_dim, 
                  checkpoint_dir=FLAGS.checkpoint_dir,
                  sample_dir=FLAGS.sample_dir,
                  vi_gradient=args.vi_gradient,
                  vi_res=args.vi_res,
                  vi_fusion_gradient=args.vi_fusion_gradient,
                  vi_fusion_intensity=args.vi_fusion_intensity,
                  ir_intensity=args.ir_intensity,
                  ir_res=args.ir_res,
                  ir_fusion_intensity=args.ir_fusion_intensity,
                  ir_fusion_gradient=args.ir_fusion_gradient,
                  vi_loss=args.vi_loss,
                  ir_loss=args.ir_loss,
                  fusion_vi_gradient=args.fusion_vi_gradient,
                  fusion_vi_intensity=args.fusion_vi_intensity,
                  fusion_ir_gradient=args.fusion_ir_gradient,
                  fusion_ir_intensity=args.fusion_ir_intensity,
                  dir_name=args.dir)

    srcnn.train(FLAGS)
    
if __name__ == '__main__':
  tf.app.run()
