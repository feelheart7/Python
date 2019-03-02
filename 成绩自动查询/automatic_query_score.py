# coding=utf-8
# 操作图片模块
from PIL import Image
# 第三方OCR调用（识别率不高）
import pytesseract
# 范围随机模块
import random
# 操作系统模块
import os
# 矩阵计算与tensorflow（深度学习框架）
import numpy as np
import tensorflow as tf
# 爬虫模拟网页请求模块
import requests
# 调用系统浏览器
import webbrowser
# 爬取html指定内容
from bs4 import BeautifulSoup
# 正则模块
import re
# smtp邮箱
import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr
# 时间模块，用于延迟
import time
# 下载图片模块
from typing import Any
from urllib.request import urlretrieve
# TF_CPP_MIN_LOG_LEVEL默认值为 0 (显示所有logs)
# 设置为 1 隐藏 INFO logs, 2 额外隐藏WARNING logs
# 设置为3所有 ERROR logs也不显示
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 全局变量
IMAGE_HEIGHT = 22  # 验证码图片高度
IMAGE_WIDTH = 68   # 验证码图片宽度
MAX_CAPTCHA = 4    # 验证码的位数
CHAR_SET_LEN = 36  # 验证码的字符有多少种
# 验证码图片网址
IMAGE_URL = "http://jwfw1.sdjzu.edu.cn/ssfw/jwcaptcha.do"
# 下载验证码图片的数量
VERIFICATION_CODE_NUMBER = 10
# 验证码文件夹与文件绝对存储路径
VERIFICATION_CODE_PATH1 = os.path.dirname(__file__) + '/verification_code_images/'
VERIFICATION_CODE_PATH2 = os.path.dirname(__file__) + '/verification_code_images/{name}.png'
# 训练数据集绝对存储路径
VERIFICATION_CODE_TRAINING_PATH = os.path.dirname(__file__) + '/verification_code_training_images/'
# placeholder 是 Tensorflow 中的占位符，暂时储存变量 X Y ,keep_prob是dropout层保留概率
X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT*IMAGE_WIDTH])
Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA*CHAR_SET_LEN])
keep_prob = tf.placeholder(tf.float32)
# 邮箱信息
SENDER = '873773108@qq.com'
PASSWORD = 'hoxspyfkwmqcbddd'
# 用户信息
info = [{
        'account': '201611101122',
        'password': '174519',
        'email': '873773108@qq.com'
        },
        {
        'account': '201611101083',
        'password': '136574',
        'email': '1763992358@qq.com'
        }]
# String1为原始成绩,String2为最新成绩。创建方式为这样append防止默认copy（以免浅拷贝）
String1 = [[] for copy in range(len(info))]
String2 = [[] for copy in range(len(info))]
# 训练集所有图片list与
all_images = os.listdir(VERIFICATION_CODE_TRAINING_PATH)
all_images_size = len(all_images)


# 下载验证码图片安装序号存储图片
def download_verification_code():
    # 创建文件夹如果没有
    os.makedirs(VERIFICATION_CODE_PATH1, exist_ok=True)
    for i in range(0, VERIFICATION_CODE_NUMBER):
        urlretrieve(IMAGE_URL, VERIFICATION_CODE_PATH2.format(name=i))
    print("成功下载%s张图片！" % VERIFICATION_CODE_NUMBER)


# 调用pytesseract整体识别验证码(识别率低)
def pytesseract_verification_code():
    for i in range(0, VERIFICATION_CODE_NUMBER):
        img = Image.open(VERIFICATION_CODE_PATH2.format(name=i))
        char = pytesseract.image_to_string(img, config='--psm 8')
        # psm 各个值的说明
        # 0：定向脚本监测（OSD）
        # 1： 使用OSD自动分页
        # 2 ：自动分页，但是不使用OSD或OCR（Optical Character Recognition，光学字符识别）
        # 3 ：全自动分页，但是没有使用OSD（默认）
        # 4 ：假设可变大小的一个文本列。
        # 5 ：假设垂直对齐文本的单个统一块。
        # 6 ：假设一个统一的文本块。
        # 7 ：将图像视为单个文本行。
        # 8 ：将图像视为单个词。
        # 9 ：将图像视为圆中的单个词。
        # 10 ：将图像视为单个字符
        print(i, char)


# 二值化分割验证码再调用pytesseract识别验证码(识别率有所提高)
def pytesseract_devide_verification_code():
    # 随机读取图片并灰度化
    random_number = random.randint(0, VERIFICATION_CODE_NUMBER)
    img = Image.open(VERIFICATION_CODE_PATH2.format(name=random_number)).convert('L')
    # 二值化:173为我的验证图片有较好的效果的值，不同图片的值不一样，请根据自己验证码图片设置相应的值
    img = img.point(lambda x: 255 if x > 173 else 0)
    # 分离:crop函数带的参数为(起始点的横坐标，起始点的纵坐标，宽度，高度）
    img1 = img.crop((0 * IMAGE_WIDTH / MAX_CAPTCHA, 0, 1 * IMAGE_WIDTH / MAX_CAPTCHA, IMAGE_HEIGHT))
    img2 = img.crop((1 * IMAGE_WIDTH / MAX_CAPTCHA, 0, 2 * IMAGE_WIDTH / MAX_CAPTCHA, IMAGE_HEIGHT))
    img3 = img.crop((2 * IMAGE_WIDTH / MAX_CAPTCHA, 0, 3 * IMAGE_WIDTH / MAX_CAPTCHA, IMAGE_HEIGHT))
    img4 = img.crop((3 * IMAGE_WIDTH / MAX_CAPTCHA, 0, 4 * IMAGE_WIDTH / MAX_CAPTCHA, IMAGE_HEIGHT))
    # 调用pytesseract识别验证码
    char = pytesseract.image_to_string(img, config='--psm 8')
    char1 = pytesseract.image_to_string(img1, config='--psm 10')
    char2 = pytesseract.image_to_string(img2, config='--psm 10')
    char3 = pytesseract.image_to_string(img3, config='--psm 10')
    char4 = pytesseract.image_to_string(img4, config='--psm 10')
    print(char)
    print(char1)
    print(char2)
    print(char3)
    print(char4)


################################################################
### 通过tensorflow的CNN(卷积神经网络深度学习后识别验证码,识别率贼高####
################################################################


# 获取验证码名字和图片（训练数据集）
def get_name_and_image():
    # 获取数据集下的所有图片的数组all_images
    # all_images = os.listdir(VERIFICATION_CODE_TRAINING_PATH)
    random_image = random.randint(0, all_images_size - 1)
    # print (all_images_size)
    base = os.path.basename(VERIFICATION_CODE_TRAINING_PATH + all_images[random_image])  # 有扩展名
    name = os.path.splitext(base)[0]  # 无扩展名
    image = Image.open(VERIFICATION_CODE_TRAINING_PATH + all_images[random_image])
    image = np.array(image)
    return name, image


# 验证码名字转变成向量: 不同位数的需要重写这个函数,函数里的数字为ASCII码
def name2vec(name):
    vector = np.zeros(MAX_CAPTCHA*CHAR_SET_LEN)
    for i, c in enumerate(name):
        if ord(c) < 58:
            idx = i * 36 + ord(c)-48
            vector[idx] = 1
        else:
            idx = i * 36 + ord(c) - 87
            vector[idx] = 1
    return vector


# 向量转名字:注释部分是 最开始的向量 转 名字
# def vec2name(vec):
#     name = []
#     for i, c in enumerate(vec):
#         if c == 1.0:
#             name.append(i)
#     for i in range(0, 4):
#         if name[i] % 36 < 10:
#             name[i] = chr(name[i] % 36 + 48)
#         else:
#             name[i] = chr(name[i] % 36 + 87)
#     return "".join(name)


# 向量转名字: 训练是不用到这个函数，训练完成用这个函数得到最终结果
def vec2name(vec):
    name = []
    for i in vec:
        if i < 10:
            a = chr(i + 48)
            name.append(a)
        else:
            a = chr(i + 87)
            name.append(a)
    return "".join(name)


# 采样函数:默认一次采集64张验证码作为一次训练
# 需要注意通过get_name_and_image()函数获得的image是一个含布尔值的矩阵
# 在这里通过1*(image.flatten())函数转变成只含0和1的1行114*450列的矩阵
def get_next_batch(batch_size=64):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT*IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA*CHAR_SET_LEN])

    for i in range(batch_size):
        name, image = get_name_and_image()
        batch_x[i, :] = 1*(image.flatten())
        batch_y[i, :] = name2vec(name)
    return batch_x, batch_y


# 定义CNN(卷积神经网络):三个卷积层卷积神经网络结构
# 采用3个卷积层加1个全连接层的结构，在每个卷积层中都选用2*2的最大池化层和dropout层，卷积核尺寸选择5*5。，
# 我们的图片已经经过了3层池化层，也就是长宽都压缩了8倍(各自取整为3X9)
def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
    # 3个卷积层
    w_c1 = tf.Variable(w_alpha * tf.random_normal([5, 5, 1, 32]))
    b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob)

    w_c2 = tf.Variable(w_alpha * tf.random_normal([5, 5, 32, 64]))
    b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)

    w_c3 = tf.Variable(w_alpha * tf.random_normal([5, 5, 64, 64]))
    b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)

    # 1个全连接层
    w_d = tf.Variable(w_alpha * tf.random_normal([3*9*64, 1024]))
    b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
    dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)

    w_out = tf.Variable(w_alpha * tf.random_normal([1024, MAX_CAPTCHA * CHAR_SET_LEN]))
    b_out = tf.Variable(b_alpha * tf.random_normal([MAX_CAPTCHA * CHAR_SET_LEN]))
    out = tf.add(tf.matmul(dense, w_out), b_out)
    return out


# 训练函数：选择sigmoid_cross_entropy_with_logits()交叉熵来比较loss
# 用adam优化器来优化
# keep_prob = 0.3，控制着过拟合
def train_crack_captcha_cnn():
    output = crack_captcha_cnn()
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
    predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step = 0
        while True:
            batch_x, batch_y = get_next_batch(256)
            _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.2})
            print(step, loss_)
            # 每100 step计算一次准确率
            if step % 1000 == 0:
                batch_x_test, batch_y_test = get_next_batch(1000)
                acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
                print(step, acc)
                # 如果准确率大于99%,保存模型,完成训练
                if acc > 0.999:
                    saver.save(sess, "./crack_capcha.model", global_step=step)
                    break
            step += 1


# train_crack_captcha_cnn()


# 训练完成后,注释train_crack_captcha_cnn()，取消下面的注释，开始预测，注意更改预测集目录
# def crack_captcha():
#     output = crack_captcha_cnn()
#
#     saver = tf.train.Saver()
#     with tf.Session() as sess:
#         saver.restore(sess, tf.train.latest_checkpoint('.'))
#         n = 1
#         while n <= 10:
#             text, verification_code_training_images = get_name_and_image()
#             verification_code_training_images = 1 * (verification_code_training_images.flatten())
#             predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
#             text_list = sess.run(predict, feed_dict={X: [verification_code_training_images], keep_prob: 1})
#             vec = text_list[0].tolist()
#             predict_text = vec2name(vec)
#             print("正确: {}  预测: {}".format(text, predict_text))
#             n += 1
#
#
# crack_captcha()


def captcha():
    output = crack_captcha_cnn()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.'))
        first_time = True

        while True:
            users_number = 0
            for information in info:
                while True:
                    session = requests.Session()
                    headers = {"User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:39.0) Gecko/20100101 Firefox/39.0"}
                    html = session.get(IMAGE_URL, headers=headers).content
                    with open('./test_captcha/test.png', 'wb') as file:
                        file.write(html)
                    img = Image.open('./test_captcha/test.png').convert('L')
                    # 二值化
                    img = img.point(lambda x: 255 if x > 173 else 0)
                    img = np.array(img)
                    img = 1 * (img.flatten())
                    predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
                    text_list = sess.run(predict, feed_dict={X: [img], keep_prob: 1})
                    vec = text_list[0].tolist()
                    # print("预测:", vec2name(vec))

                    session.get('http://jwfw1.sdjzu.edu.cn/ssfw/login.jsp')
                    data = {'j_username': information['account'], 'j_password': information['password'],
                            'validateCode': vec2name(vec)}
                    r = session.post('http://jwfw1.sdjzu.edu.cn/ssfw/j_spring_ids_security_check', data=data, headers=headers)
                    if (re.search(r'校验码错误', r.text, re.I | re.M)) is None:
                        print("验证码正确通过！")
                        n = session.get(
                            'http://jwfw1.sdjzu.edu.cn/ssfw/jwnavmenu.do?menuItemWid=1E057E24ABAB4CAFE0540010E0235690',
                            headers=headers)
                        soup = BeautifulSoup(n.content, features='html.parser')
                        s = soup.select('div[title="有效成绩"] .t_con td[align="center"]')
                        subjects_number = int(len(s) / 11)
                        # print("科目数:", subjects_number)
                        for i in range(0, subjects_number):
                            # print('序号:', s[i * 11].get_text(strip=True))
                            # print('学年学期:', s[i * 11 + 1].get_text( strip=True))
                            # print('课程号:', s[i * 11 + 2].get_text(strip=True))
                            # print('课程名称:', s[i * 11 + 3].get_text( strip=True))
                            # print('课程类别:', s[i * 11 + 4].get_text(strip=True))
                            # print('任选课类别:', s[i * 11 + 5].get_text(strip=True))
                            # print('课程性质:', s[i * 11 + 6].get_text(strip=True))
                            # print('学分:', s[i * 11 + 7].get_text(strip=True))
                            # print('成绩:', s[i * 11 + 8].get_text(strip=True))
                            # print('****************')
                            ss = '{} {} 成绩: {}'.format(s[i * 11].get_text(strip=True), s[i * 11 + 3].get_text(strip=True),
                                                       s[i * 11 + 8].get_text(strip=True))
                            if first_time:
                                String1[users_number].append(ss)
                            else:
                                String2[users_number].append(ss)
                        # 发送邮件
                        my_user = '%s' % information['email']
                        sss = "".join(list(set(String2[users_number]).difference(set(String1[users_number]))))  # b中有而a中没有的
                        ret = True
                        if first_time:
                            text = '\n'.join(String1[users_number])

                            try:
                                msg = MIMEText(text, 'plain', 'utf-8')
                                msg['From'] = formataddr(["weijiajin", SENDER])  # 括号里的对应发件人邮箱昵称、发件人邮箱账号
                                msg['To'] = formataddr(["亲～:", my_user])  # 括号里的对应收件人邮箱昵称、收件人邮箱账号
                                msg['Subject'] = "全部成绩！好好学习！"  # 邮件的主题，也可以说是标题
                                server = smtplib.SMTP_SSL("smtp.qq.com", 465)  # 发件人邮箱中的SMTP服务器，端口是465
                                server.login(SENDER, PASSWORD)  # 括号中对应的是发件人邮箱账号、邮箱密码
                                server.sendmail(SENDER, [my_user, ], msg.as_string())  # 括号中对应的是发件人邮箱账号、收件人邮箱账号、发送邮件
                                server.quit()  # 关闭连接
                            except Exception:  # 如果 try 中的语句没有执行，则会执行下面的 ret=False
                                ret = False
                            if ret:
                                print("发送邮件成功！")
                            else:
                                print("发送邮件失败！")
                            String2[users_number].clear()
                            users_number = + 1
                            break
                        elif sss == "":
                            print("没有最新成绩！不发送邮件！")
                            String2[users_number].clear()
                            users_number = + 1
                            break
                        elif sss != "":
                            text = '\n'.join(String2[users_number])
                            title = "".join(list(set(String2[users_number]).difference(set(String1[users_number]))))
                            try:
                                msg = MIMEText(text, 'plain', 'utf-8')
                                msg['From'] = formataddr(["weijiajin", SENDER])  # 括号里的对应发件人邮箱昵称、发件人邮箱账号
                                msg['To'] = formataddr(["亲～", my_user])  # 括号里的对应收件人邮箱昵称、收件人邮箱账号
                                msg['Subject'] = "最新成绩出来啦～" + title  # 邮件的主题，也可以说是标题
                                server = smtplib.SMTP_SSL("smtp.qq.com", 465)  # 发件人邮箱中的SMTP服务器，端口是465
                                server.login(SENDER, my_pass)  # 括号中对应的是发件人邮箱账号、邮箱密码
                                server.sendmail(SENDER, [my_user, ], msg.as_string())  # 括号中对应的是发件人邮箱账号、收件人邮箱账号、发送邮件
                                server.quit()  # 关闭连接
                            except Exception:  # 如果 try 中的语句没有执行，则会执行下面的 ret=False
                                ret = False
                            if ret:
                                print("发送邮件成功！")

                            else:
                                print("发送邮件失败！")
                            String1[users_number].clear()
                            String1[users_number] = String2[users_number]
                            String2[users_number].clear()
                            users_number = + 1
                            break
                        time.sleep(300)
                    else:
                        print("校验码错误！重新尝试中......")

            first_time = False


def auto_download_train_images():
    output = crack_captcha_cnn()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.'))
        times1 = 0
        times2 = 0

        while True:
            session = requests.Session()
            headers = {"User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:39.0) Gecko/20100101 Firefox/39.0"}
            html = session.get(IMAGE_URL, headers=headers).content
            with open('./test_captcha/test.png', 'wb') as file:
                file.write(html)
            img = Image.open('./test_captcha/test.png').convert('L')
            # 二值化
            img = img.point(lambda x: 255 if x > 173 else 0)
            img1 = np.array(img)
            img1 = 1 * (img1.flatten())
            predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
            text_list = sess.run(predict, feed_dict={X: [img1], keep_prob: 1})
            vec = text_list[0].tolist()
            # print("预测:", vec2name(vec))
            session.get('http://jwfw1.sdjzu.edu.cn/ssfw/login.jsp')
            data = {'j_username': '201611101122', 'j_password': '174519',
                    'validateCode': vec2name(vec)}
            r = session.post('http://jwfw1.sdjzu.edu.cn/ssfw/j_spring_ids_security_check', data=data, headers=headers)
            if (re.search(r'校验码错误', r.text, re.I | re.M)) is None:
                # print("验证码正确通过！")
                # print(times1)
                image_name = '{image_name}.png'
                img.save(os.path.join('./verification_code_training_images/', image_name.format(image_name=vec2name(vec))))
                times1 = times1 + 1
            else:
                # print("校验码错误！")
                # print(times2)
                image_name = '{image_name}.png'
                img.save(os.path.join('./error_images/', image_name.format(image_name=vec2name(vec))))
                times2 = times2 + 1
            rate = times1/(times2+times1)
            print("总次数: % s" % (times2+times1)+"正确率：% s " % rate)


captcha()
# auto_download_train_images()
