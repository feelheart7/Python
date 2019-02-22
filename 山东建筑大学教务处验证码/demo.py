# coding=utf-8
from PIL import Image
import pytesseract
import random
import os
import numpy as np
import tensorflow as tf
import requests
import webbrowser
from bs4 import BeautifulSoup
import re
import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 全局变量

IMAGE_HEIGHT = 22
IMAGE_WIDTH = 68
MAX_CAPTCHA = 4
CHAR_SET_LEN = 36
all_times = 0
# 发件人邮箱账号
my_sender = '873773108@qq.com'
# 发件人邮箱密码(当时申请smtp给的口令)
my_pass = 'hoxspyfkwmqcbddd'
# 收件人邮箱账号，我这边发送给自己
# my_user = '136574390@qq.com'


s = './verification_code_training_images/img{name}.png'
IMAGE_URL = "http://jwfw1.sdjzu.edu.cn/ssfw/jwcaptcha.do"
# 获取当前运行脚本的绝对路径
path = os.path.dirname(__file__)
# 原始图片位置
origin_path = path + '/verification_code_training_images/'
path_image = path + '/verification_code_training_images/'
all_image = os.listdir(origin_path)

info = [
    {
        'account': '201611101122',
        'password': '174519',
        'email': '873773108@qq.com'
    },
    {
        'account': '201611101083',
        'password': '136574',
        'email': '1763992358@qq.com'
    }
]

String1 = [[]] * len(info)

# 下载验证码图片
'''
os.makedirs('./verification_code_training_images/', exist_ok=True)



def urllib_download():
    for i in range(759, 4000):
        from urllib.request import urlretrieve
        urlretrieve(IMAGE_URL, s.format(name=i))


urllib_download()
'''
# 调用pytesseract整体识别验证码
'''
for i in range(0, 200):
 im = Image.open(s.format(name=i))
 char = pytesseract.image_to_string(im, config='--psm 10')
 print(i,char)
'''


# 分割验证码然后识别

'''
#获取当前运行脚本的绝对路径
path = os.path.dirname(__file__)
#原始图片位置
origin_path = path + '/verification_code_training_images/img0.jpg'
#用来存放处理好的图片
new_path = path + '/clean_images/'
#用来存放测试的图片
test_path = path + '/test_images/'



# 读取图片并灰度化
img = Image.open(origin_path).convert('L')
#二值化
img = img.point(lambda  x:255 if x>173 else 0)

# 分离
img1 = img.crop((0, 0, 17, 22))
img2 = img.crop((17, 0, 33, 22))
img3 = img.crop((33, 0, 49, 22))
img4 = img.crop((49, 0, 65, 22))

char = pytesseract.image_to_string(img, config='--psm 6')
char1 = pytesseract.image_to_string(img1, config='--psm 8')
char2 = pytesseract.image_to_string(img2, config='--psm 8')
char3 = pytesseract.image_to_string(img3, config='--psm 8')
char4 = pytesseract.image_to_string(img4, config='--psm 8')
print(char)
print(char1)
print(char2)
print(char3)
print(char4)
img.save(os.path.join(test_path,'new.PNG'))
img1.save(os.path.join(test_path,'new1.PNG'))
img2.save(os.path.join(test_path,'new2.PNG'))
img3.save(os.path.join(test_path,'new3.PNG'))
img4.save(os.path.join(test_path,'new4.PNG'))
'''

'''
# 二值化
all_image1 = os.listdir(path_image)
for i in range(0, 1006):
    base = os.path.basename(path_image+all_image1[i])  # 有扩展名
    name = os.path.splitext(base)[0]  # 无扩展名
    # 读取图片并灰度化
    img = Image.open(path_image+base).convert('L')
    # 二值化
    img = img.point(lambda x:255 if x > 173 else 0)
    img.save(os.path.join(path_image,base))
'''


# 获取验证码名字和图片
def get_name_and_image():
    all_image1 = os.listdir(origin_path)
    random_file = random.randint(0, 1194)
    base = os.path.basename(origin_path + all_image1[random_file])
    name = os.path.splitext(base)[0]
    image = Image.open(origin_path + all_image1[random_file])

    image = np.array(image)
    return name, image


# 名字转变成向量
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


# 向量转名字
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


# 生成一个训练batch
def get_next_batch(batch_size=64):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT*IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA*CHAR_SET_LEN])

    for i in range(batch_size):
        name, image = get_name_and_image()
        batch_x[i, :] = 1*(image.flatten())
        batch_y[i, :] = name2vec(name)
    return batch_x, batch_y


######
X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT*IMAGE_WIDTH])
Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA*CHAR_SET_LEN])
keep_prob = tf.placeholder(tf.float32)


# 定义CNN
def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
    # 3 conv layer
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

    # Fully connected layer
    w_d = tf.Variable(w_alpha * tf.random_normal([3*9*64, 1024]))
    b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
    dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)

    w_out = tf.Variable(w_alpha * tf.random_normal([1024, MAX_CAPTCHA * CHAR_SET_LEN]))
    b_out = tf.Variable(b_alpha * tf.random_normal([MAX_CAPTCHA * CHAR_SET_LEN]))
    out = tf.add(tf.matmul(dense, w_out), b_out)
    return out


# 训练
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
            batch_x, batch_y = get_next_batch(64)
            _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.3})
            print(step, loss_)

            # 每100 step计算一次准确率
            if step % 100 == 0:
                batch_x_test, batch_y_test = get_next_batch(100)
                acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
                print(step, acc)
                # 如果准确率大于99%,保存模型,完成训练
                if acc > 0.999:
                    saver.save(sess, "./crack_capcha.model", global_step=step)
                    break

            step += 1


# train_crack_captcha_cnn()
# 训练完成后#掉train_crack_captcha_cnn()，取消下面的注释，开始预测，注意更改预测集目录
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


def test_captcha():
    global all_times
    output = crack_captcha_cnn()
    saver = tf.train.Saver()
    # os.makedirs('./test_captcha/', exist_ok=True
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.'))
        Error_Times = 0
        while True:
            try:
                for xx in info:
                    ii = 0
                    session = requests.Session()
                    headers = {"User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:39.0) Gecko/20100101 Firefox/39.0"}
                    html = session.get(IMAGE_URL, headers=headers).content
                    with open('./test_captcha/test.png', 'wb') as file:
                        file.write(html)
                    # from urllib.request import urlretrieve
                    # urlretrieve(IMAGE_URL, './test_captcha/test.png')
                    # 读取图片并灰度化
                    img = Image.open('./test_captcha/test.png').convert('L')
                    # 二值化
                    img = img.point(lambda x: 255 if x > 173 else 0)
                    # img.show()
                    # img.save('./test_captcha/test.png')
                    img = np.array(img)
                    img = 1 * (img.flatten())
                    predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
                    text_list = sess.run(predict, feed_dict={X: [img], keep_prob: 1})
                    vec = text_list[0].tolist()
                    # print("预测:", vec2name(vec))
                    session.get('http://jwfw1.sdjzu.edu.cn/ssfw/login.jsp')
                    data = {'j_username': xx['account'], 'j_password': xx['password'],
                            'validateCode': vec2name(vec)}
                    r = session.post('http://jwfw1.sdjzu.edu.cn/ssfw/j_spring_ids_security_check', data=data, headers=headers)
                    if (re.search(r'校验码错误', r.text, re.I | re.M)) is None:
                        print("验证码正确通过！")
                        print("尝试失败次数:", Error_Times)
                        n = session.get('http://jwfw1.sdjzu.edu.cn/ssfw/jwnavmenu.do?menuItemWid=1E057E24ABAB4CAFE0540010E0235690',  headers=headers)
                        # print(n.text)
                        soup = BeautifulSoup(n.content, features='html.parser')
                        # s1 = soup.select('div[title="有效成绩"] .tableTr')
                        # for s in s1:
                        #     print(s.get_text("|", strip=True))
                        #
                        # s2 = soup.select('div[title="有效成绩"] .t_con')
                        # for s in s2:
                        #     print(s.get_text("|", strip=True))

                        s3 = soup.select('div[title="有效成绩"] .t_con td[align="center"]')
                        subjects_number = int(len(s3)/11)
                        print('****************')
                        print("科目数:", subjects_number)


                        String2 = []
                        for i in range(0, subjects_number):
                            # print('序号:', s3[i * 11].get_text(strip=True))
                            # print('学年学期:', s3[i * 11 + 1].get_text( strip=True))
                            # print('课程号:', s3[i * 11 + 2].get_text(strip=True))
                            # print('课程名称:', s3[i * 11 + 3].get_text( strip=True))
                            # print('课程类别:', s3[i * 11 + 4].get_text(strip=True))
                            # print('任选课类别:', s3[i * 11 + 5].get_text(strip=True))
                            # print('课程性质:', s3[i * 11 + 6].get_text(strip=True))
                            # print('学分:', s3[i * 11 + 7].get_text(strip=True))
                            # print('成绩:', s3[i * 11 + 8].get_text(strip=True))
                            # print('****************')
                            ss = '{} {} 成绩: {}'.format(s3[i * 11].get_text(strip=True), s3[i * 11 + 3].get_text(strip=True), s3[i * 11 + 8].get_text(strip=True))
                            if all_times == 0:
                                String1[i].append(ss)
                            else:
                                String2.append(ss)
                            # print(ss)

                        # 发送邮件
                        my_user = xx['email']
                        sss = "".join(list(set(String2).difference(set(String1))))  # b中有而a中没有的
                        ret = True
                        if all_times == 0:
                            text = '\n'.join(String1)
                            try:

                                msg = MIMEText(text, 'plain', 'utf-8')
                                msg['From'] = formataddr(["发件人昵称:", my_sender])  # 括号里的对应发件人邮箱昵称、发件人邮箱账号
                                msg['To'] = formataddr(["收件人昵称:", my_user])  # 括号里的对应收件人邮箱昵称、收件人邮箱账号
                                msg['Subject'] = "全部成绩！好好学习！"  # 邮件的主题，也可以说是标题
                                server = smtplib.SMTP_SSL("smtp.qq.com", 465)  # 发件人邮箱中的SMTP服务器，端口是465
                                server.login(my_sender, my_pass)  # 括号中对应的是发件人邮箱账号、邮箱密码
                                server.sendmail(my_sender, [my_user, ], msg.as_string())  # 括号中对应的是发件人邮箱账号、收件人邮箱账号、发送邮件
                                server.quit()  # 关闭连接
                            except Exception:  # 如果 try 中的语句没有执行，则会执行下面的 ret=False
                                ret = False
                            if ret:
                                print("发送邮件成功！")
                            else:
                                print("发送邮件失败！")
                        elif sss != "":
                            text = '\n'.join(String2)
                            title = "".join(list(set(String2).difference(set(String1))))
                            try:
                                msg = MIMEText(text, 'plain', 'utf-8')
                                msg['From'] = formataddr(["发件人昵称:", my_sender])  # 括号里的对应发件人邮箱昵称、发件人邮箱账号
                                msg['To'] = formataddr(["收件人昵称:", my_user])  # 括号里的对应收件人邮箱昵称、收件人邮箱账号
                                msg['Subject'] = "最新成绩" + title  # 邮件的主题，也可以说是标题
                                server = smtplib.SMTP_SSL("smtp.qq.com", 465)  # 发件人邮箱中的SMTP服务器，端口是465
                                server.login(my_sender, my_pass)  # 括号中对应的是发件人邮箱账号、邮箱密码
                                server.sendmail(my_sender, [my_user, ], msg.as_string())  # 括号中对应的是发件人邮箱账号、收件人邮箱账号、发送邮件
                                server.quit()  # 关闭连接
                            except Exception:  # 如果 try 中的语句没有执行，则会执行下面的 ret=False
                                ret = False
                            if ret:
                                print("发送邮件成功！")

                            else:
                                print("发送邮件失败！")
                        # time.sleep(300)
                        all_times += 1
                        print("总尝试次数:", all_times)

                    else:
                        print("校验码错误！")
                        Error_Times = Error_Times+1
                        print("尝试失败次数:", Error_Times)
                    i =+ 1
            except Exception:
                time.sleep(30)
                continue


test_captcha()
