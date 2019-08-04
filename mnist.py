#! -*- coding: utf-8 -*-

#---------------------------------
# モジュールのインポート
#---------------------------------
import os
import sys
import tqdm
import argparse
import pathlib
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#---------------------------------
# 関数
#---------------------------------
def model_000(input_dims=784, output_dims=10):
	x = tf.placeholder(tf.float32, [None, input_dims])
	
	W = tf.Variable(tf.zeros([input_dims, output_dims]))
	b = tf.Variable(tf.zeros([output_dims]))
	
	y = tf.add(tf.matmul(x, W), b, name='output')
	
	y_ = tf.placeholder(tf.float32, [None, output_dims])
	
	return x, y, y_
	
def model_001(input_dims=784, hidden1_dims=300, output_dims=10):
	x = tf.placeholder(tf.float32, [None, input_dims])
	
	W1 = tf.Variable(tf.truncated_normal([input_dims, hidden1_dims]))
	b1 = tf.Variable(tf.zeros([hidden1_dims]))
	h1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
	
	W2 = tf.Variable(tf.truncated_normal([hidden1_dims, output_dims]))
	b2 = tf.Variable(tf.zeros([output_dims]))
	y = tf.add(tf.matmul(h1, W2), b2, name='output')
	
	y_ = tf.placeholder(tf.float32, [None, output_dims])
	
	return x, y, y_

def model_002(input_dims=784, hidden1_dims=300, hidden2_dims=100, output_dims=10):
	x = tf.placeholder(tf.float32, [None, input_dims])
	
	W1 = tf.Variable(tf.truncated_normal([input_dims, hidden1_dims]))
	b1 = tf.Variable(tf.zeros([hidden1_dims]))
	h1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
	
	W2 = tf.Variable(tf.truncated_normal([hidden1_dims, hidden2_dims]))
	b2 = tf.Variable(tf.zeros([hidden2_dims]))
	h2 = tf.nn.sigmoid(tf.matmul(h1, W2) + b2)
	
	W3 = tf.Variable(tf.truncated_normal([hidden2_dims, output_dims]))
	b3 = tf.Variable(tf.zeros([output_dims]))
	y = tf.add(tf.matmul(h2, W3), b3, name='output')
	
	y_ = tf.placeholder(tf.float32, [None, output_dims])
	
	return x, y, y_

def model_003(input_dims=784, output_dims=10):
	x = tf.placeholder(tf.float32, [None, input_dims])
	
	W = tf.Variable(tf.zeros([input_dims, output_dims]))
	y = tf.matmul(x, W, name='output')
	
	y_ = tf.placeholder(tf.float32, [None, output_dims])
	
	return x, y, y_
	
def model_004(input_dims=784, hidden1_dims=300, output_dims=10):
	x = tf.placeholder(tf.float32, [None, input_dims])
	
	W1 = tf.Variable(tf.truncated_normal([input_dims, hidden1_dims]))
	h1 = tf.nn.sigmoid(tf.matmul(x, W1))
	
	W2 = tf.Variable(tf.truncated_normal([hidden1_dims, output_dims]))
	y = tf.matmul(h1, W2, name='output')
	
	y_ = tf.placeholder(tf.float32, [None, output_dims])
	
	return x, y, y_

def model_005(input_dims=784, hidden1_dims=300, hidden2_dims=100, output_dims=10):
	x = tf.placeholder(tf.float32, [None, input_dims])
	
	W1 = tf.Variable(tf.truncated_normal([input_dims, hidden1_dims]))
	h1 = tf.nn.sigmoid(tf.matmul(x, W1))
	
	W2 = tf.Variable(tf.truncated_normal([hidden1_dims, hidden2_dims]))
	h2 = tf.nn.sigmoid(tf.matmul(h1, W2))
	
	W3 = tf.Variable(tf.truncated_normal([hidden2_dims, output_dims]))
	y = tf.matmul(h2, W3, name='output')
	
	y_ = tf.placeholder(tf.float32, [None, output_dims])
	
	return x, y, y_

def model_006(input_dims=784, output_dims=10):
	x = tf.placeholder(tf.float32, [None, input_dims])
	
	b = tf.Variable(tf.zeros([output_dims]))
	W = tf.Variable(tf.zeros([input_dims, output_dims]))
	
	y = tf.add(b, tf.matmul(x, W), name='output')
	
	y_ = tf.placeholder(tf.float32, [None, output_dims])
	
	return x, y, y_
	
def model_007(input_dims=784, hidden1_dims=300, output_dims=10):
	x = tf.placeholder(tf.float32, [None, input_dims])
	
	b1 = tf.Variable(tf.zeros([hidden1_dims]))
	W1 = tf.Variable(tf.truncated_normal([input_dims, hidden1_dims]))
	h1 = tf.nn.sigmoid(b1 + tf.matmul(x, W1))
	
	b2 = tf.Variable(tf.zeros([output_dims]))
	W2 = tf.Variable(tf.truncated_normal([hidden1_dims, output_dims]))
	y = tf.add(b2, tf.matmul(h1, W2), name='output')
	
	y_ = tf.placeholder(tf.float32, [None, output_dims])
	
	return x, y, y_

def model_008(input_dims=784, hidden1_dims=300, hidden2_dims=100, output_dims=10):
	x = tf.placeholder(tf.float32, [None, input_dims])
	
	b1 = tf.Variable(tf.zeros([hidden1_dims]))
	W1 = tf.Variable(tf.truncated_normal([input_dims, hidden1_dims]))
	h1 = tf.nn.sigmoid(b1 + tf.matmul(x, W1))
	
	b2 = tf.Variable(tf.zeros([hidden2_dims]))
	W2 = tf.Variable(tf.truncated_normal([hidden1_dims, hidden2_dims]))
	h2 = tf.nn.sigmoid(b2 + tf.matmul(h1, W2))
	
	b3 = tf.Variable(tf.zeros([output_dims]))
	W3 = tf.Variable(tf.truncated_normal([hidden2_dims, output_dims]))
	y = tf.add(b3, tf.matmul(h2, W3), 'output')
	
	y_ = tf.placeholder(tf.float32, [None, output_dims])
	
	return x, y, y_

def train(mnist_data, x, y, y_, n_epoch=20, n_minibatch=100, model_dir='model'):
	loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y)
	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	
	init = tf.initialize_all_variables()
	config = tf.ConfigProto(
		gpu_options=tf.GPUOptions(
			allow_growth = True
		)
	)
	sess = tf.Session(config=config)
	sess.run(init)
	saver = tf.train.Saver()
	
	iter_minibatch = len(mnist_data.train.images)
	for epoch in tqdm.tqdm(range(n_epoch)):
		for _iter in range(iter_minibatch):
			batch_x, batch_y = mnist_data.train.next_batch(n_minibatch)
			sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})
	
	print(sess.run(accuracy, feed_dict={x: mnist_data.test.images, y_: mnist_data.test.labels}))
	
	os.makedirs(model_dir, exist_ok=True)
	saver.save(sess, os.path.join(model_dir, 'model.ckpt'))
	
	sess.close()
	tf.reset_default_graph()
	
	return

def get_ops(outfile):
	graph = tf.get_default_graph()
	all_ops = graph.get_operations()
	
	with open(outfile, 'w') as f:
		for _op in all_ops:
#			f.write('{}'.format(_op.op_def))
#			if ((_op.op_def.name == 'MatMul') or (_op.op_def.name == 'Add')):
#				f.write('<< {} >>\n'.format(_op.op_def.name))
#				for _input in _op.inputs:
#					f.write(' * {}\n'.format(_input))
			f.write('{}\n'.format(_op))
	
	return

def get_weights(sess, outfile):
	weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
	
	np_print_th = np.get_printoptions()['threshold']
	np.set_printoptions(threshold=np.inf)
	weights_shape = []
	with open(outfile, 'w') as f:
		for _weight in tqdm.tqdm(weights):
			weight_val = sess.run(_weight)
			f.write('{}\n{}\n\n'.format(_weight, weight_val))
			if (len(weights_shape) == 0):
#				weights_shape = np.array([weight_val.shape])
				weights_shape = [weight_val.shape]
			else:
#				weights_shape = np.vstack((weights_shape, weight_val.shape))
				weights_shape.append(weight_val.shape)
	np.set_printoptions(threshold=np_print_th)
	print(weights_shape)
	
#	if (len(weights_shape[-1]) == 1):
#		print('output nodes = {}'.format(weights_shape[-1]))
#	else:
#		print('output nodes = {}'.format(weights_shape[-1][1]))
	
	coef = 1
	flg_detect_weight = False
	flg_detect_bias = False
	flg_no_bias = False
	stack = None
	for i, weight_shape in enumerate(weights_shape):
		if (len(weight_shape) == 1):
			coef = 2
#			print('layer{}_bias : {}'.format(i // coef, weight_shape))
			if (flg_detect_weight):
				print('layer{}_weight : {}'.format(i // coef, stack))
				print('layer{}_bias : {}'.format(i // coef, weight_shape))
				flg_detect_bias = False
				flg_detect_weight = False
			else:
				stack = weight_shape
				flg_detect_bias = True
		else:
#			print('layer{}_weight : {}'.format(i // coef, weight_shape))
			if (flg_detect_bias):
				print('layer{}_weight : {}'.format(i // coef, weight_shape))
				print('layer{}_bias : {}'.format(i // coef, stack))
				flg_detect_bias = False
				flg_detect_weight = False
			else:
				if (flg_detect_weight):
					flg_no_bias = True
					print('layer{}_weight : {}'.format((i-1) // coef, stack))
				stack = weight_shape
				flg_detect_weight = True
	
	if ((flg_no_bias) or (i == 0)):
		print('layer{}_weight : {}'.format(i // coef, stack))
	
	return

def ArgParser():
	parser = argparse.ArgumentParser(description='TensorFlow Low Level APIのサンプル', formatter_class=argparse.RawTextHelpFormatter)
	
	parser.add_argument('--train', dest='flg_train', action='store_true', help='セット時，モデルを生成')
	parser.add_argument('--model', dest='model', type=str, default=None, required=False, help='TensorFlow学習済みモデルを指定')
	
	return parser.parse_args()
	
#---------------------------------
# メイン処理
#---------------------------------
def main():
	args = ArgParser()
	
	if (args.flg_train):
		print('load mnist data')
		mnist = input_data.read_data_sets(os.path.join('.', 'MNIST_data'), one_hot=True)
		
		models = [model_000, model_001, model_002, model_003, model_004, model_005, model_006, model_007, model_008]
		for i, model in enumerate(models):
			print('load model')
			x, y, y_ = model()
			print('train')
			train(mnist, x, y, y_, model_dir='model_{:03}'.format(i))
	else:
		config = tf.ConfigProto(
			gpu_options=tf.GPUOptions(
				allow_growth = True
			)
		)
		sess = tf.Session(config=config)
		
		saver = tf.train.import_meta_graph(args.model + '.meta', clear_devices=True)
		saver.restore(sess, args.model)
		
		model_dir = str(pathlib.Path(args.model).resolve().parent)
		
		get_ops(os.path.join(model_dir, 'operations.txt'))
		get_weights(sess, os.path.join(model_dir, 'weights.txt'))
		
	return

if __name__ == '__main__':
	main()

