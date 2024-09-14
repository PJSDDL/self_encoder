import tensorflow as tf
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import random

PIC_SIZE_X = 512
PIC_SIZE_Y = 512 - 1
PIC_CHANNEL = 3
PIC_SIZE = PIC_SIZE_X * PIC_SIZE_Y * PIC_CHANNEL
PIC_DIR = "RESIZETEST/"
TEST_DIR = "RESIZETEST/"

#图像拆分成左右两部分，并resize图片为一维数组
def resize(img):
    img_left = img[:, 0:PIC_SIZE_X-1, :]
    img_right = img[:, PIC_SIZE_X:2*PIC_SIZE_X-1, :]

    #归一化
    img_left = img_left / 255
    img_right = img_right / 255

    return img_left, img_right

#
def encode(img):
    
                
    return img
    

#
def decode(img):
    
            
    return img
    
#随机读取N张图片的第channel个颜色通道，加噪声，制成数据集
#N大于图片总数时，读取所有图片
def dataset(pic_dir, N):
    file_name = os.listdir(pic_dir)
    file_num = len(file_name)
    in_pic_arr = np.zeros((N, PIC_SIZE_X, PIC_SIZE_Y, PIC_CHANNEL))
    out_pic_arr = np.zeros((N, PIC_SIZE_X, PIC_SIZE_Y, PIC_CHANNEL))
    
    pic_range = 0
    if (N < file_num):
        rnd_num = random.sample(range(0, file_num), N)
        pic_range = N
    else :
        rnd_num = range(0, file_num)
        pic_range = file_num
        
    print("pic_num = ", rnd_num)

    #随机抽取N张图片
    for i in range(pic_range):
        index = rnd_num[i]
        img = cv.imread(pic_dir + file_name[index])

        #resize图片并保存
        img_l, img_r = resize(img)
        noise = 0.1 * np.random.random(size=[PIC_SIZE_X, PIC_SIZE_Y, PIC_CHANNEL])
        in_pic_arr[i] = encode(img_l) + noise
        out_pic_arr[i] = encode(img_r)

    return in_pic_arr, out_pic_arr


#定义自编码器
#定义自编码器
#三通道卷积
def rgb_conv(rgb, channel, acti=tf.nn.relu):
    conv_r = tf.layers.conv2d(rgb[0], channel, (3,3), padding='same', activation = acti)
    conv_g = tf.layers.conv2d(rgb[1], channel, (3,3), padding='same', activation = acti)
    conv_b = tf.layers.conv2d(rgb[2], channel, (3,3), padding='same', activation = acti)

    return [conv_r, conv_g, conv_b]

#三通道反卷积
def rgb_UpConv(rgb, channel):
    conv_r = tf.layers.conv2d_transpose(rgb[0], channel, (2,2), strides=(2,2), padding='same', activation=tf.nn.relu)
    conv_g = tf.layers.conv2d_transpose(rgb[1], channel, (2,2), strides=(2,2), padding='same', activation=tf.nn.relu)
    conv_b = tf.layers.conv2d_transpose(rgb[2], channel, (2,2), strides=(2,2), padding='same', activation=tf.nn.relu)

    return [conv_r, conv_g, conv_b]

    
#网络结构
inputs_ = tf.placeholder(tf.float32, (None, PIC_SIZE_X, PIC_SIZE_Y, PIC_CHANNEL), name='inputs')
targets_ = tf.placeholder(tf.float32, (None, PIC_SIZE_X, PIC_SIZE_Y, PIC_CHANNEL), name='targets')


### Encoder
#Conv
conv1 = tf.layers.conv2d(inputs_, 64, (3,3), padding='same', activation=tf.nn.relu)
#DownSample
maxpool1 = tf.layers.max_pooling2d(conv1, (2,2), (2,2), padding='same')
#Conv
conv2 = tf.layers.conv2d(maxpool1, 64, (3,3), padding='same', activation=tf.nn.relu)
#DownSample
maxpool2 = tf.layers.max_pooling2d(conv2, (2,2), (2,2), padding='same')
#Conv
conv3 = tf.layers.conv2d(maxpool2, 64, (3,3), padding='same', activation=tf.nn.relu)
#DownSample
maxpool3 = tf.layers.max_pooling2d(conv3, (2,2), (2,2), padding='same')
#Conv
conv4 = tf.layers.conv2d(maxpool3, 64, (3,3), padding='same', activation=tf.nn.relu)
#DownSample
maxpool4 = tf.layers.max_pooling2d(conv4, (2,2), (2,2), padding='same')
#Conv
conv5 = tf.layers.conv2d(maxpool4, 64, (3,3), padding='same', activation=tf.nn.relu)
#DownSample
maxpool5 = tf.layers.max_pooling2d(conv5, (2,2), (2,2), padding='same')
#Conv
conv6 = tf.layers.conv2d(maxpool5, 64, (3,3), padding='same', activation=tf.nn.relu)
#DownSample
maxpool6 = tf.layers.max_pooling2d(conv6, (2,2), (2,2), padding='same')
#Conv
conv7 = tf.layers.conv2d(maxpool6, 64, (3,3), padding='same', activation=tf.nn.relu)
#DownSample
maxpool7 = tf.layers.max_pooling2d(conv7, (2,2), (2,2), padding='same')

### Decoder
#UpConv
upconv_5 = rgb_UpConv([maxpool7, maxpool7, maxpool7], 64)
#Conv
conv_5 = rgb_conv(upconv_5, 64)
#UpConv
upconv_6 = rgb_UpConv(conv_5, 64)
#Conv
conv_6 = rgb_conv(upconv_6, 64)
#UpConv
upconv_7 = rgb_UpConv(conv_6, 64)
#Conv
conv_7 = rgb_conv(upconv_7, 64)
#UpConv
upconv_8 = rgb_UpConv(conv_7, 64)
#Conv
conv_8 = rgb_conv(upconv_8, 64)
#UpConv
upconv_9 = rgb_UpConv(conv_8, 64)
#Conv
conv_9 = rgb_conv(upconv_9, 64)
#UpConv
upconv_10 = rgb_UpConv(conv_9, 64)
#Conv
conv_10 = rgb_conv(upconv_10, 64)
#UpConv
upconv_11 = rgb_UpConv(conv_10, 64)
#Conv
conv_11 = rgb_conv(upconv_11, 64)
#concat
conv1_resize = tf.image.resize(conv1, [PIC_SIZE_X, PIC_SIZE_Y + 1])
conv_r_12 = tf.concat((conv_11[0], conv1_resize), axis=3)
conv_g_12 = tf.concat((conv_11[1], conv1_resize), axis=3)
conv_b_12 = tf.concat((conv_11[2], conv1_resize), axis=3)
#Conv
conv13 = rgb_conv([conv_r_12, conv_g_12, conv_b_12], 1, acti=tf.nn.tanh)
#concat  resize
img_concat = tf.concat((conv13[0], conv13[1], conv13[2]), axis=3)
decoded_def = tf.image.resize(img_concat, [PIC_SIZE_X, PIC_SIZE_Y])


###残差学习
decode = inputs_ + decoded_def

loss = tf.square(decode - targets_)
cost = tf.reduce_mean(loss)
opt = tf.train.AdamOptimizer(2e-4).minimize(cost)

sess = tf.Session()

variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
#打印网络参数
total_var_num = 0
for str_num in variables:
    var_shape = str_num.shape
    var_num = 1
    for i in var_shape:
        var_num *= i 
    total_var_num += var_num
print(str(total_var_num) + "\n\n")

epochs = 510000
batch_size = 200
CH_NUM = 2
err_trace = []
test_err = []
saver=tf.train.Saver()
sess.run(tf.global_variables_initializer())

#载入上一次保存的模型
USE_BEFORE = True
try:
    if USE_BEFORE:
        model_file=tf.train.latest_checkpoint('ckpt/')
        saver.restore(sess, model_file)
except:
    print("保存模型不可用\n\n")


batch_x, batch_y = dataset(PIC_DIR, 4)


for e in range(epochs):
    batch_cost, _ = sess.run([cost, opt], feed_dict={inputs_: batch_y, targets_: batch_x})
    
    print("Epoch: {}/{}...".format(e+1, epochs),
              "Training loss: {:.6f}".format(np.mean(batch_cost)))
    
    err_trace.append(np.mean(batch_cost))

    if (e % 20 == 1):
        #重新生成数据集
        batch_x, batch_y = dataset(PIC_DIR, 4)
        
    if (e % 200 == 1):
        #保存模型
        saver.save(sess,'ckpt/enco.ckpt',global_step=e)

    if (e % 100000 == 1):
        y_, x, y = sess.run([decode, inputs_, targets_], feed_dict={inputs_: batch_y, targets_: batch_x})
        
        plt.subplot(2, 3, 2)
        plt.imshow(y_[0])
        plt.subplot(2, 3, 1)
        plt.imshow(x[0])
        plt.subplot(2, 3, 3)
        plt.imshow(y[0])
        plt.subplot(2, 3, 5)
        plt.imshow(y_[1])
        plt.subplot(2, 3, 4)
        plt.imshow(x[1])
        plt.subplot(2, 3, 6)
        plt.imshow(y[1])
        plt.show()

        '''
        plt.subplot(2, 1, 1)
        plt.plot(range(len(err_trace)), err_trace)
        plt.subplot(2, 1, 2)
        plt.plot(range(len(test_err)), test_err)
        plt.show()
        '''


        
#测试网络
#batch_xs, batch_ys = dataset(PIC_DIR, 3)
#encode_out = sess.run(y, feed_dict={x: batch_xs, y_: batch_ys})

#plt.subplot(1, 2, 1)
#plt.imshow(de_resize(encode_out[1]))
#plt.subplot(1, 2, 2)
#plt.imshow(de_resize(encode_out[1]))
#plt.show()







