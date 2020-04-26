# 导入必要的包
import os
import paddle
import paddle.dataset.imdb as imdb
import paddle.fluid as fluid
import numpy as np
import matplotlib.pyplot as plt
from config import config
opt=config()
# 存画图的数据的数组
losses_lst = []
test_acc_lst = []

# 获取数据字典
print("加载数据字典中...")
word_dict = imdb.word_dict()
# 获取数据字典长度
dict_dim = len(word_dict)
print('完成')

imdb.train(word_dict)
# 获取训练和预测数据
print("加载训练数据中...")
train_reader = paddle.batch(paddle.reader.shuffle(imdb.train(word_dict),
                                                  512),
                            batch_size=opt.batch_size)
print("加载测试数据中...")
test_reader = paddle.batch(imdb.test(word_dict),
                           batch_size=opt.batch_size)
print('完成')


# 如果需要跑LSTM的网络，请直接取消这个block的注释，并注释下个代码块的代码
# 之前使用的LSTM网络，准确率大概在85%，后来我还尝试使用Bi-LSTM，见下方 当然这个代码也可以跑的 不过这个就是最直接的LSTM模型
def lstm_net(ipt, input_dim,opt_config):
    # emb layer
    emb = fluid.layers.embedding(input=ipt, size=[input_dim, opt.embedding_size], is_sparse=True)
    print(emb)
    # full connect layer
    fc1 = fluid.layers.fc(input=emb, size=opt.fc1,
                        param_attr = fluid.ParamAttr(
                            initializer=fluid.initializer.TruncatedNormal(scale=0.02,
                                                                          seed=opt.seed)),
                        bias_attr = fluid.ParamAttr(
                            initializer=fluid.initializer.Constant(0.))
                        )
    print(fc1)
    # full connect layer
    fc2 = fluid.layers.fc(input=fc1, size=opt.hidden_units,
                          param_attr=fluid.ParamAttr(
                              initializer=fluid.initializer.TruncatedNormal(scale=0.02,
                                                                            seed=opt.seed)),
                          bias_attr=fluid.ParamAttr(
                              initializer=fluid.initializer.Constant(0.))
                          )

    # lstm layer
    print(fc2)
    lstm1, _ = fluid.layers.dynamic_lstm(input=fc2,  # 返回：隐藏状态（hidden state），LSTM的神经元状态
                                         size=opt.hidden_units,
                                         param_attr = fluid.ParamAttr(
                                            initializer=fluid.initializer.TruncatedNormal(scale=0.02,
                                                                                          seed=opt.seed)),
                                         bias_attr = fluid.ParamAttr(
                                            initializer=fluid.initializer.Constant(0.))
                                         )  # size=4*hidden_size
    # max pooling layer
    print(lstm1)
    fc3 = fluid.layers.sequence_pool(input=fc2, pool_type='max',
                                     param_attr=fluid.ParamAttr(
                                         initializer=fluid.initializer.TruncatedNormal(scale=0.02,
                                                                                       seed=opt.seed)),
                                     bias_attr=fluid.ParamAttr(
                                         initializer=fluid.initializer.Constant(0.))
                                     )
    # max pooling layer
    lstm2 = fluid.layers.sequence_pool(input=lstm1, pool_type='max',
                                       param_attr=fluid.ParamAttr(
                                           initializer=fluid.initializer.TruncatedNormal(scale=0.02,
                                                                                         seed=opt.seed)),
                                       bias_attr=fluid.ParamAttr(
                                           initializer=fluid.initializer.Constant(0.))
                                       )
    # softmax layer 二分类也就是正负面
    out = fluid.layers.fc(input=[fc3, lstm2], size=opt.class_num, act='softmax',
                          param_attr=fluid.ParamAttr(
                              initializer=fluid.initializer.TruncatedNormal(scale=0.02,
                                                                            seed=opt.seed)),
                          bias_attr=fluid.ParamAttr(
                              initializer=fluid.initializer.Constant(0.))
                          )
    return out


# 如果需要跑LSTM的网络，请直接注释本代码块的代码，并取消上个block的注释，
# 改进后的栈式双向LSTM 我调来调去还是85%左右 网络改的更复杂的话反而效果会更差
def stack_lstm_net(data, input_dim,opt_config):
    class_dim = opt.class_num
    emb_dim = opt.embedding_size
    hid_dim = opt.hidden_units
    stacked_num = opt.stacked_num
    # 计算词向量
    emb = fluid.layers.embedding(
        input=data, size=[input_dim, emb_dim], is_sparse=True)

    # 第一层栈
    # 全连接层 tanh激活函数
    fc1 = fluid.layers.fc(input=emb, size=hid_dim,
                            param_attr = fluid.ParamAttr(
                                initializer=fluid.initializer.TruncatedNormal(scale=0.02,
                                                                              seed=opt.seed)),
                            bias_attr = fluid.ParamAttr(
                                initializer=fluid.initializer.Constant(0.))
                            )
    # lstm层
    lstm1, cell1 = fluid.layers.dynamic_lstm(input=fc1, size=hid_dim,
                                             param_attr=fluid.ParamAttr(
                                                 initializer=fluid.initializer.TruncatedNormal(scale=0.02,
                                                                                               seed=opt.seed)),
                                             bias_attr=fluid.ParamAttr(
                                                 initializer=fluid.initializer.Constant(0.))
                                             )

    inputs = [fc1, lstm1]

    # 其余的所有栈结构 tanh激活函数
    for i in range(2, stacked_num + 1):
        fc = fluid.layers.fc(input=inputs, size=hid_dim,
                             param_attr=fluid.ParamAttr(
                                 initializer=fluid.initializer.TruncatedNormal(scale=0.02,
                                                                               seed=opt.seed+i)),
                             bias_attr=fluid.ParamAttr(
                                 initializer=fluid.initializer.Constant(0.))
                             )
        lstm, cell = fluid.layers.dynamic_lstm(
            input=fc, size=hid_dim, is_reverse=(i % 2) == 0,
            param_attr=fluid.ParamAttr(

                initializer=fluid.initializer.TruncatedNormal(scale=0.02, seed=opt.seed+i)),
            bias_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(0.)),
        )
        inputs = [fc, lstm]

    # 池化层
    fc_last = fluid.layers.sequence_pool(input=inputs[0], pool_type='max')
    lstm_last = fluid.layers.sequence_pool(input=inputs[1], pool_type='max')

    # 全连接层，softmax预测
    prediction = fluid.layers.fc(
        input=[fc_last, lstm_last], size=class_dim, act='softmax',
                param_attr = fluid.ParamAttr(
                    initializer=fluid.initializer.TruncatedNormal(scale=0.02,
                                                                  seed=opt.seed)),
                             bias_attr = fluid.ParamAttr(
                    initializer=fluid.initializer.Constant(0.))
                )
    return prediction


# 如果需要跑LSTM的网络，请直接取消这个block的注释，并注释下个代码块的代码
# 之前使用的LSTM网络，准确率大概在85%，后来我还尝试使用Bi-LSTM，见下方 当然这个代码也可以跑的 不过这个就是最直接的LSTM模型
def single_lstm_net(ipt, input_dim):
    # emb layer
    emb = fluid.layers.embedding(input=ipt, size=[input_dim, 128], is_sparse=True)
    print(emb)
    # full connect layer
    fc1 = fluid.layers.fc(input=emb, size=128)
    print(fc1)
    # full connect layer
    fc2 = fluid.layers.fc(input=fc1, size=128)
    # lstm layer
    print(fc2)
    lstm1, _ = fluid.layers.dynamic_lstm(input=fc2,  # 返回：隐藏状态（hidden state），LSTM的神经元状态
                                         size=128)  # size=4*hidden_size
    # max pooling layer
    print(lstm1)
    # fc3 = fluid.layers.sequence_pool(input=fc2, pool_type='max')
    # max pooling layer
    # lstm2 = fluid.layers.sequence_pool(input=lstm1, pool_type='max')
    # softmax layer 二分类也就是正负面
    out = fluid.layers.fc(input=lstm1, size=2, act='softmax')
    return out


def train(lstm_net, i,opt_config):
    s=''
    with log_writter.mode("train%d" % opt_config.id) as train_log1:
        train_loss_log = train_log1.scalar(tag="loss")
        train_acc_log = train_log1.scalar(tag="acc")
        # num_samples = 10
        # self.train_histogram = train_log.histogram(tag="train_histogram", num_buckets=50)
    with log_writter.mode("test%d" % opt_config.id) as test_log1:
        test_loss_log = test_log1.scalar(tag="loss")
        test_acc_log = test_log1.scalar(tag="acc")
    # 定义输入数据， lod_level不为0指定输入数据为序列数据
    words = fluid.layers.data(name='words', shape=[1], dtype='int64', lod_level=1)
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    # 获取长短期记忆网络
    model = lstm_net(words, dict_dim,opt_config)
    # 获取损失函数和准确率
    cost = fluid.layers.cross_entropy(input=model, label=label)
    avg_cost = fluid.layers.mean(cost)
    acc = fluid.layers.accuracy(input=model, label=label)
    # 获取预测程序
    test_program = fluid.default_main_program().clone(for_test=True)
    # 定义优化方法 以及很重要的指标学习率learning rate
    optimizer = fluid.optimizer.Adagrad(learning_rate=opt_config.lr)
    opt = optimizer.minimize(avg_cost)
    # 定义使用CPU还是GPU，使用CPU时use_cuda = False,使用GPU时use_cuda = True
    use_cuda = False
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    # 进行参数初始化
    exe.run(fluid.default_startup_program())
    # 定义输入数据的维度
    # 定义数据数据的维度，数据的顺序是一条句子数据对应一个标签
    feeder = fluid.DataFeeder(place=place, feed_list=[words, label])

    # 这个是LSTM的训练结果，可以换成Stacked Bi-LSTM稍微好一些
    # 开始训练
    # %matplotlib inline
    num_epochs = 100
    early_stopping=opt_config.early_stopping
    best=0.
    for pass_id in range(num_epochs):
        # 进行训练
        train_mean_cost = 0.
        train_mean_acc = 0.
        for batch_id, data in enumerate(train_reader()):  # 遍历train_reader迭代器
            # print(data)
            train_cost, train_acc = exe.run(program=fluid.default_main_program(),  # 运行主程序
                                            feed=feeder.feed(data),  # 喂入一个batch的数据
                                            fetch_list=[avg_cost, acc])  # fetch均方误差
            train_mean_acc += train_acc[0]
            train_mean_cost += train_cost[0]
            if batch_id != 0 and batch_id % 40 == 0:  # 每40次batch打印一次训练、进行一次测试
                train_mean_cost = train_mean_cost / 40.0
                train_mean_acc = train_mean_acc / 40.0
                print('Pass:%d, Batch:%d, Cost:%0.5f,Acc:%0.5f' % (pass_id, batch_id, train_mean_cost, train_mean_acc))
                train_loss_log.add_record(pass_id * 160 + batch_id, train_mean_cost)
                train_acc_log.add_record(pass_id * 160 + batch_id, train_mean_acc)
                s=' | '.join([str(train_mean_cost), str(train_mean_acc)])+' | '
                with open('./work/log/train%d.txt' % opt_config.id, 'a', encoding='utf-8') as f:
                    f.write(','.join([str(pass_id * 160 + batch_id), str(train_mean_cost), str(train_mean_acc), '\n']))
                train_mean_cost = 0.
                train_mean_acc = 0.


        # 进行测试
        test_costs = []  # 测试的损失值
        test_accs = []  # 测试的准确率
        for batch_id, data in enumerate(test_reader()):
            test_cost, test_acc = exe.run(program=test_program,
                                          feed=feeder.feed(data),
                                          fetch_list=[avg_cost, acc])
            test_costs.append(test_cost[0])
            test_accs.append(test_acc[0])
        # 计算平均预测损失在和准确率
        test_cost = (sum(test_costs) / len(test_costs))
        test_acc = (sum(test_accs) / len(test_accs))
        print('Test:%d, Cost:%0.5f, ACC:%0.5f' % (pass_id, test_cost, test_acc))
        # losses_lst.append(test_cost)
        # test_acc_lst.append(test_acc)
        test_loss_log.add_record(pass_id * 160, test_cost)
        test_acc_log.add_record(pass_id * 160, test_acc)
        with open('./work/log/dev%d.txt' % opt_config.id, 'a', encoding='utf-8') as f:
            f.write(','.join([str(pass_id * 160 + batch_id), str(test_cost), str(test_acc), '\n']))
        if test_cost<best:
            best=test_cost
            early_stopping=opt_config.early_stopping
        else:
            early_stopping-=1
            if(early_stopping==0):
                break
    s=s+' | '.join([str(test_cost), str(test_acc)])
    opt_config.write(' | '+s)

from visualdl import LogWriter

log_writter = LogWriter("./log", sync_cycle=1)
# train(single_lstm_net,1)

#
# def gridsearch():
#     opt.id=4
#     hidden=[64,128,256,512]
#     for stack in range(1,4):
#         opt.stacked_num=stack
#         for h in hidden:
#             opt.hidden_units=h
#             train(stack_lstm_net,opt.modelid,opt)
#             opt.id+=1
# gridsearch()
opt.id=6
opt.hidden_units=256
opt.stacked_num=1
train(stack_lstm_net, opt.modelid,opt)