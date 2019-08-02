#-*- coding:utf-8 -*-
from captcha.image import ImageCaptcha
from PIL import Image
import numpy as np
import random
import string

class generateCaptcha():
    def __init__(self,
                 width = 160,#验证码图片的宽
                 height = 60,#验证码图片的高
                 char_num = 4,#验证码字符个数
                 characters = string.digits + string.ascii_uppercase + string.ascii_lowercase):#验证码组成，数字+大写字母+小写字母
        self.width = width
        self.height = height
        self.char_num = char_num
        self.characters = characters
        self.classes = len(characters)

    def gen_captcha(self,batch_size = 50):
        X = np.zeros([batch_size,self.height,self.width,1])
        img = np.zeros((self.height,self.width),dtype=np.uint8)
        Y = np.zeros([batch_size,self.char_num,self.classes])
        image = ImageCaptcha(width = self.width,height = self.height)

        while True:
            for i in range(batch_size):
                # 生成一个验证码字符串的随机变量，self.characters为62位的字符串（0~9A~Za~z），self.char_num=4（生成4个字符）。
                captcha_str = ''.join(random.sample(self.characters,self.char_num))
                # 使用的是ImageCaptcha类的内置方法，将字符串变为图片。convert(‘L’)：表示生成的是灰度图片，就是通道数为1的黑白图片。
                img = image.generate_image(captcha_str).convert('L')
                # 将此图像的内容作为包含像素值的序列对象返回。
                img = np.array(img.getdata())
                # 每个像素值都要除以255，这是为了归一化处理，因为灰度的范围是0~255，这里除以255就让每个像素的值在0~1之间，目的是为了加快收敛速度。
                X[i] = np.reshape(img,[self.height,self.width,1])/255.0
                # 用以生成对应的测试集Y，j和ch用以遍历刚刚生成的随机字符串，j记录index（0~3，表示第几个字符），ch记录字符串中的字符。找到Y的第i条数据中的第j个字符，然后把62长度的向量和ch相关的那个置为1。
                for j,ch in enumerate(captcha_str):
                    Y[i,j,self.characters.find(ch)] = 1
            Y = np.reshape(Y,(batch_size,self.char_num*self.classes))
            yield X,Y

    def decode_captcha(self,y):
        y = np.reshape(y,(len(y),self.char_num,self.classes))
        return ''.join(self.characters[x] for x in np.argmax(y,axis = 2)[0,:])

    def get_parameter(self):
        return self.width,self.height,self.char_num,self.characters,self.classes

    def gen_test_captcha(self):
        image = ImageCaptcha(width = self.width,height = self.height)
        captcha_str = ''.join(random.sample(self.characters,self.char_num))
        img = image.generate_image(captcha_str)
        img.save(captcha_str + '.jpg')

        X = np.zeros([1,self.height,self.width,1])
        Y = np.zeros([1,self.char_num,self.classes])
        img = img.convert('L')
        img = np.array(img.getdata())
        X[0] = np.reshape(img,[self.height,self.width,1])/255.0
        for j,ch in enumerate(captcha_str):
            Y[0,j,self.characters.find(ch)] = 1
        Y = np.reshape(Y,(1,self.char_num*self.classes))
        return X,Y