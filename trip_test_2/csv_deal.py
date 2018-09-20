
import tensorflow as tf
# from sklearn.preprocessing import OneHotEncoder
from Kmeans_sklearn import cluster_destination

features,label_1,center=cluster_destination()
print(center)
print(center.shape)

#one_hot变换
label=[]
for i in label_1:
    onehot_en = [ 0 if j !=i else 1 for j in range(18)]
    label.append(onehot_en)


in_units = 20
h1_units = 10
W1 = tf.Variable(tf.truncated_normal([in_units,h1_units],stddev=0.1))
b1 = tf.Variable(tf.zeros([h1_units]))
W2 = tf.Variable(tf.zeros([h1_units,18]))
b2 = tf.Variable(tf.zeros([18]))
W3 = tf.constant(center)

x = tf.placeholder(tf.float32,[None,in_units])
keep_prob = tf.placeholder(tf.float32)

hidden1 = tf.nn.relu(tf.matmul(x,W1)+ b1)#激活函数

hidden1_drop = tf.nn.dropout(hidden1,keep_prob)#正则化，神经元失活
hidden2 = tf.nn.softmax(tf.matmul(hidden1_drop,W2)+b2)
y_pre = tf.matmul(x,W3)


#训练部分
y_ture = tf.placeholder(tf.float32, [None, 18])
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ture * tf.log(y_pre), reduction_indices=[1]))#损失函数
#cross_entropy = tf.nn.softmax_cross_entropy_with_logits(y_ture,y_pre)
#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_ture,logits=y_pre))
cross_entropy = tf.sqrt(tf.add(tf.square(tf.matmul(y_pre[0]-y_ture[0],tf.cos(tf.matmul(0.5,y_pre[1]-y_ture[1])))),tf.square(y_pre[1]-y_ture[1])))
train_step = tf.train.GradientDescentOptimizer(0.3).minimize(cross_entropy)#更新网络

#定义一个InteractiveSession会话并初始化全部变量
correct_prediction = tf.equal(tf.arg_max(y_pre, 1), tf.arg_max(y_ture, 1))#Returns the truth value of (x == y) element-wise.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))#cast,转化类型;reduce_mean,计算平均值
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for i in range(10):
    avg_cost=0
    # batch_xs, batch_ys = mnist.train.next_batch(100)
    for j in range(1,7):
        for z in range(100):
            X_train=features[(j-1)*1000:j*1000]
            Y_train=label[(j-1)*1000:j*1000]
            # print(Y_train)
            train_step.run({x: X_train, y_ture: Y_train, keep_prob: 0.75})

            train_step_values, cross_entropy_values,accuracy_values,y_pre_values,W2_values,y_ture_values = sess.run([train_step, cross_entropy,accuracy,y_pre,W2,y_ture], feed_dict={x: X_train, y_ture: Y_train, keep_prob: 0.75})
            # avg_cost += cross_entropy_values
            # _,W2_values = sess.run([train_step,W2], feed_dict={x: X_train, y_ture: Y_train, keep_prob: 0.75})
            if z%50==0:
                # print('迭代：第',i,'次','batch:',j,'W2_values:',W2_values)
                # print('迭代：第',i,'次','batch:',j,'W2_values.size:',W2_values.shape)
                # print('迭代：第',i,'次','batch:',j,'avg_cost:',avg_cost)
                print('迭代：第',i,'次','batch:',j,'cross_entropy:',cross_entropy_values)
                # print('迭代：第',i,'次','batch:',j,'train_step:', train_step_values)
                print('迭代：第',i,'次','batch:',j,'accuracy',accuracy_values)
                print('迭代：第', i, '次', 'batch:', j, 'y_ture_values', 'training_arruracy:',accuracy.eval({x: features[7000:7500], y_ture: label[7000:7500],
                                                                                       keep_prob: 1.0}))
                # print('迭代：第',i,'次','batch:',j,'y_pre', y_pre_values)
                # print('迭代：第',i,'次','batch:',j,'y_pre', y_pre_values.shape)
                # print('迭代：第',i,'次','batch:',j,'y_ture', y_ture_values)
                # print('迭代：第',i,'次','batch:',j,'y_ture_values', y_ture_values.shape)
                # W1_values=sess.run(W1)
                # print(type(W1_values),W1_values)
        #     print('迭代：第',i,'次','batch:',j,'y_ture_values', 'training_arruracy:', accuracy.eval({x: features[7000:7005], y_ture: label[7000:7005],
        #                                                   keep_prob: 1.0}))
        #     print('迭代：第',i,'次','batch:',j,'y_ture_values', 'correct_prediction:', correct_prediction.eval({x: features[7000:7005], y_ture: label[7000:7005],
        #                                                   keep_prob: 1.0}))
    # print('迭代：第', i, '次', 'y_ture_values', 'training_arruracy:',
    #       accuracy.eval({x: features[7000:7005], y_ture: label[7000:7005],
    #                      keep_prob: 1.0}))
    # print('迭代：第', i, '次', 'y_ture_values', 'correct_prediction:',
    #       correct_prediction.eval({x: features[7000:7005], y_ture: label[7000:7005],
    #                                keep_prob: 1.0}))
        # y_pre_values = sess.run(y_pre)
        # print(type(y_pre_values), y_pre_values)
        # print('hidden1:',sess.run([hidden1],feed_dict={x: features[7000:7005] , keep_prob: 1}))
        # print('hidden1_drop:', sess.run([hidden1], feed_dict={x: features[7000:7005], keep_prob: 1}))
        # print('y_pre:', sess.run([hidden1], feed_dict={x: features[7000:7005], keep_prob: 1}))
    # if i % 10 ==0:
        # print(i, 'training_arruracy:', accuracy.eval({x: features[7000:7300], y_ture: label[7000:7300],
         #                                              keep_prob: 1.0}))
        # print(i,'y_pre:',y_pre.eval({x: features[7000:7010], y_ture: label[7000:7010],
        #                                               keep_prob: 1.0}))
        # pred_value = sess.run([y_pre], feed_dict={x: features[7000:7005] , keep_prob: 1.})
        # print("pred_value:", pred_value)
print('final_accuracy:', accuracy.eval({x: features[7000:7300],y_ture: label[7000:7300], keep_prob: 1.0}))
