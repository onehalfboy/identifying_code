#coding=utf-8
from gen_captcha import gen_captcha_text_and_image
from gen_captcha import number
from gen_captcha import alphabet
from gen_captcha import ALPHABET
#from train import convert2gray
#from train import crack_captcha_cnn
#from train import IMAGE_HEIGHT
#from train import IMAGE_WIDTH
#from train import CHAR_SET_LEN
#from train import X
#from train import keep_prob

import tensorflow as tf
import train_function

def crack_captcha(captcha_image):
	output = train_function.crack_captcha_cnn()
 
	saver = tf.train.Saver()
	with tf.Session() as sess:
		saver.restore(sess, tf.train.latest_checkpoint('./data'))

		predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, train_function.CHAR_SET_LEN]), 2)
		text_list = sess.run(predict, feed_dict={train_function.X: [captcha_image], train_function.keep_prob: 1})
		texts = text_list[0].tolist()
		chars = number+alphabet+ALPHABET
		text = []
		for index,num in enumerate(texts):
			text += chars[num]
			#print("index:{} num:{}".format(index, num))

		return text
 
text, image = gen_captcha_text_and_image()
MAX_CAPTCHA = len(text)
image = train_function.convert2gray(image)
image = image.flatten() / 255
predict_text = crack_captcha(image)
print("正确: {}  预测: {}".format(text, predict_text))
