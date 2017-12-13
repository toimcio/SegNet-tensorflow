import os
import numpy as np
import tensorflow as tf
import math
from inputs import get_filename_list, dataset_inputs, get_all_test_data
from evaluation import Normal_Loss, cal_loss, per_class_acc, get_hist, print_hist_summery, train_op
from inference import segnet_vgg, segnet_scratch, segnet_bayes_scratch, segnet_bayes_vgg

NUM_CLASS = 12


def Test():
    batch_size = 1
    train_dir = "segnet_vgg_bayes_30000/model.ckpt-30000"
    test_dir = "SegNet/CamVid/test.txt"
    image_w = 480
    image_h = 360
    image_c = 3
    FLAG_INFER = "segnet_bayes_vgg"

    image_filename, label_filename = get_filename_list(test_dir)
    test_data_tensor = tf.placeholder(tf.float32, shape=[batch_size, image_h, image_w, image_c])
    test_label_tensor = tf.placeholder(tf.int64, shape=[batch_size, image_h, image_w, 1])
    phase_train = tf.placeholder(tf.bool, name='phase_train')
    keep_prob = tf.placeholder(tf.float32, shape=None, name='keep_probability')
    phase_train_dropout = tf.placeholder(tf.bool, name='phase_train_dropout')
    if (FLAG_INFER == "segnet_scratch"):
        logits = segnet_scratch(test_data_tensor, test_label_tensor, batch_size, phase_train, keep_prob,
                                phase_train_dropout)
        FLAG_BAYES = False
    elif (FLAG_INFER == "segnet_vgg"):
        logits = segnet_vgg(test_data_tensor, test_label_tensor, batch_size, phase_train, keep_prob,
                            phase_train_dropout)
        FLAG_BAYES = False
    elif (FLAG_INFER == "segnet_bayes_scratch"):
        logits = segnet_bayes_scratch(test_data_tensor, test_label_tensor, batch_size, phase_train, keep_prob,
                                      phase_train_dropout)
        FLAG_BAYES = True
    elif (FLAG_INFER == "segnet_bayes_vgg"):
        logits = segnet_bayes_vgg(test_data_tensor, test_label_tensor, batch_size, phase_train, keep_prob,
                                  phase_train_dropout)
        FLAG_BAYES = True
    else:
        print("The model is not there YET")

    loss, accuracy, prediction = Normal_Loss(logits, test_label_tensor, NUM_CLASS)
    prob = tf.nn.softmax(logits, dim=-1)
    prob = tf.reshape(prob, [image_h, image_w, NUM_CLASS])
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, train_dir)
        hist = np.zeros((NUM_CLASS, NUM_CLASS))
        # images = np.load("/zhome/1c/2/114196/Documents/SegNet-tensorflow/test_image.npy")
        # labels = np.load("/zhome/1c/2/114196/Documents/SegNet-tensorflow/test_label.npy")
        images, labels = get_all_test_data(image_filename,label_filename)

        NUM_SAMPLE = []
        for i in range(30):
            NUM_SAMPLE.append(2 * i + 1)

        acc_final = []
        iu_final = []
        iu_mean_final = []
        # uncomment the line below to only run for two times.
        NUM_SAMPLE = [1, 30]
        for num_sample_generate in NUM_SAMPLE:

            loss_tot = []
            acc_tot = []
            pred_tot = []
            var_tot = []
            hist = np.zeros((NUM_CLASS, NUM_CLASS))
            step = 0
            for image_batch, label_batch in zip(images, labels):
                image_batch = np.reshape(image_batch, [1, image_h, image_w, image_c])
                label_batch = np.reshape(label_batch, [1, image_h, image_w, 1])
                # comment the code below to apply the dropout for all the samples
                if num_sample_generate == 1:
                    feed_dict = {test_data_tensor: image_batch, test_label_tensor: label_batch, phase_train: False,
                                 keep_prob: 0.5, phase_train_dropout: False}
                else:
                    feed_dict = {test_data_tensor: image_batch, test_label_tensor: label_batch, phase_train: False,
                                 keep_prob: 0.5, phase_train_dropout: True}
                # uncomment this code below to run the dropout for all the samples
                # feed_dict = {test_data_tensor: image_batch, test_label_tensor:label_batch, phase_train: False, keep_prob:0.5, phase_train_dropout:True}
                fetches = [loss, accuracy, logits, prediction]
                if FLAG_BAYES is False:
                    loss_per, acc_per, logit, pred = sess.run(fetches=fetches, feed_dict=feed_dict)
                    var_one = []
                else:
                    logit_iter_tot = []
                    loss_iter_tot = []
                    acc_iter_tot = []
                    prob_iter_tot = []
                    logit_iter_temp = []
                    for iter_step in range(num_sample_generate):
                        loss_iter_step, acc_iter_step, logit_iter_step, prob_iter_step = sess.run(
                            fetches=[loss, accuracy, logits, prob], feed_dict=feed_dict)
                        loss_iter_tot.append(loss_iter_step)
                        acc_iter_tot.append(acc_iter_step)
                        logit_iter_tot.append(logit_iter_step)
                        prob_iter_tot.append(prob_iter_step)
                        logit_iter_temp.append(np.reshape(logit_iter_step, [image_h, image_w, NUM_CLASS]))

                    loss_per = np.nanmean(loss_iter_tot)
                    acc_per = np.nanmean(acc_iter_tot)
                    logit = np.nanmean(logit_iter_tot, axis=0)
                    print(np.shape(prob_iter_tot))

                    prob_mean = np.nanmean(prob_iter_tot, axis=0)
                    prob_variance = np.var(prob_iter_tot, axis=0)
                    logit_variance = np.var(logit_iter_temp, axis=0)

                    # THIS TIME I DIDN'T INCLUDE TAU
                    pred = np.reshape(np.argmax(prob_mean, axis=-1), [-1])  # pred is the predicted label

                    var_sep = []  # var_sep is the corresponding variance if this pixel choose label k
                    length_cur = 0  # length_cur represent how many pixels has been read for one images
                    for row in np.reshape(prob_variance, [image_h * image_w, 12]):
                        temp = row[pred[length_cur]]
                        length_cur += 1
                        var_sep.append(temp)
                    var_one = np.reshape(var_sep, [image_h,
                                                   image_w])  # var_one is the corresponding variance in terms of the "optimal" label
                    pred = np.reshape(pred, [image_h, image_w])

                loss_tot.append(loss_per)
                acc_tot.append(acc_per)
                pred_tot.append(pred)
                var_tot.append(var_one)
                print("Image Index {}: TEST Loss{:6.3f}, TEST Accu {:6.3f}".format(step, loss_tot[-1], acc_tot[-1]))
                step = step + 1
                per_class_acc(logit, label_batch, NUM_CLASS)
                hist += get_hist(logit, label_batch)

            acc_tot = np.diag(hist).sum() / hist.sum()
            iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

            print("Total Accuracy for test image: ", acc_tot)
            print("Total MoI for test images: ", iu)
            print("mean MoI for test images: ", np.nanmean(iu))

            acc_final.append(acc_tot)
            iu_final.append(iu)
            iu_mean_final.append(np.nanmean(iu))
            # uncomment the code below to save the result!
        #            if num_sample_generate == max(NUM_SAMPLE):
        #               np.save("/zhome/1c/2/114196/Documents/SegNet-tensorflow/segnet_bayes_30000/Data/acc_final",acc_final)
        #               np.save("/zhome/1c/2/114196/Documents/SegNet-tensorflow/segnet_bayes_30000/Data/iu_final",iu_final)
        #               np.save("/zhome/1c/2/114196/Documents/SegNet-tensorflow/segnet_bayes_30000/Data/iu_mean_final",iu_mean_final)
        #               np.save("/zhome/1c/2/114196/Documents/SegNet-tensorflow/segnet_bayes_30000/Data/var_tot",var_tot)
        #               np.save("/zhome/1c/2/114196/Documents/SegNet-tensorflow/segnet_bayes_30000/Data/prob_variance",prob_variance)
        #               np.save("/zhome/1c/2/114196/Documents/SegNet-tensorflow/segnet_bayes_30000/Data/logit_variance",logit_variance)
        #               np.save("/zhome/1c/2/114196/Documents/SegNet-tensorflow/segnet_bayes_30000/Data/pred_tot",pred_tot)

        return acc_final, iu_final, iu_mean_final, prob_variance, logit_variance, pred_tot, var_tot


def Train():
    max_steps = 30001
    batch_size = 5
    train_dir = "/zhome/1c/2/114196/Documents/SegNet-tensorflow/segnet_vgg_bayes_30000/"
    image_dir = "/zhome/1c/2/114196/Documents/SegNet/CamVid/train.txt"
    val_dir = "/zhome/1c/2/114196/Documents/SegNet/CamVid/val.txt"
    image_w = 360
    image_h = 480
    image_c = 3
    FLAG_INFER = "segnet_bayes_vgg"

    # For train the bayes, the FLAG_OPT SHOULD BE SGD, BUT FOR TRAIN THE NORMAL SEGNET, THE FLAG_OPT SHOULD BE ADAM!!!

    image_filename, label_filename = get_filename_list(image_dir)
    val_image_filename, val_label_filename = get_filename_list(val_dir)

    with tf.Graph().as_default():
        images_train = tf.placeholder(tf.float32, [batch_size, image_w, image_h, image_c])
        labels_train = tf.placeholder(tf.int64, [batch_size, image_w, image_h, 1])
        phase_train = tf.placeholder(tf.bool, name="phase_train")
        phase_train_dropout = tf.placeholder(tf.bool, name="phase_train_dropout")
        keep_prob = tf.placeholder(tf.float32, shape=None, name="keep_rate")
        images_tr, labels_tr = dataset_inputs(image_filename, label_filename, batch_size)
        images_val, labels_val = dataset_inputs(val_image_filename, val_label_filename, batch_size)

        if (FLAG_INFER == "segnet_scratch"):
            logits = segnet_scratch(images_train, labels_train, batch_size, phase_train, keep_prob, phase_train_dropout)
            FLAG_OPT = "ADAM"
        elif (FLAG_INFER == "segnet_vgg"):
            logits = segnet_vgg(images_train, labels_train, batch_size, phase_train, keep_prob, phase_train_dropout)
            FLAG_OPT = "ADAM"
        elif (FLAG_INFER == "segnet_bayes_scratch"):
            logits = segnet_bayes_scratch(images_train, labels_train, batch_size, phase_train, keep_prob,
                                          phase_train_dropout)
            FLAG_OPT = "ADAM"
        elif (FLAG_INFER == "segnet_bayes_vgg"):
            logits = segnet_bayes_vgg(images_train, labels_train, batch_size, phase_train, keep_prob,
                                      phase_train_dropout)
            FLAG_OPT = "ADAM"
        else:
            raise ValueError("The Model Hasn't Finished Yet")

        loss, accuracy, prediction = Normal_Loss(logits=logits, labels=labels_train, number_class=NUM_CLASS)
        train, global_step = train_op(total_loss=loss, FLAG=FLAG_OPT)

        summary_op = tf.summary.merge_all()
        saver = tf.train.Saver(tf.global_variables())

        logs_path = "/zhome/1c/2/114196/Documents/SegNet-tensorflow/graph/own/bayes_vgg/"

        with tf.Session() as sess:
            sess.run(tf.variables_initializer(tf.global_variables()))
            sess.run(tf.local_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            # The queue runners basic reference: https://www.tensorflow.org/versions/r0.12/how_tos/threading_and_queues
            train_loss, train_accuracy = [], []
            val_loss, val_acc = [], []
            train_writer = tf.summary.FileWriter(logs_path, sess.graph)
            for step in range(max_steps):
                image_batch, label_batch = sess.run([images_tr, labels_tr])
                feed_dict = {images_train: image_batch,
                             labels_train: label_batch,
                             phase_train: True,
                             keep_prob: 0.5,
                             phase_train_dropout: True}

                _, _loss, _accuracy, summary = sess.run([train, loss, accuracy, summary_op], feed_dict=feed_dict)
                train_loss.append(_loss)
                train_accuracy.append(_accuracy)
                print("Iteration {}: Train Loss{:6.3f}, Train Accu {:6.3f}".format(step, train_loss[-1],
                                                                                   train_accuracy[-1]))

                if step % 100 == 0:
                    conv_classifier = sess.run(logits, feed_dict=feed_dict)
                    print('per_class accuracy by logits in training time',
                          per_class_acc(conv_classifier, label_batch, NUM_CLASS))
                    # per_class_acc is a function from utils
                    train_writer.add_summary(summary, step)

                if step % 1000 == 0:
                    print("start validating.......")
                    _val_loss = []
                    _val_acc = []
                    hist = np.zeros((NUM_CLASS, NUM_CLASS))
                    for test_step in range(int(20)):
                        fetches_valid = [loss, accuracy, logits]
                        image_batch_val, label_batch_val = sess.run([images_val, labels_val])
                        feed_dict_valid = {images_train: image_batch_val,
                                           labels_train: label_batch_val,
                                           phase_train: True,
                                           keep_prob: 1.0,
                                           phase_train_dropout: False}
                        # since we still using mini-batch, so in the batch norm we set phase_train to be
                        # true, and because we didin't run the trainop process, so it will not update
                        # the weight!
                        _loss, _acc, _val_pred = sess.run(fetches_valid, feed_dict_valid)
                        _val_loss.append(_loss)
                        _val_acc.append(_acc)
                        hist += get_hist(_val_pred, label_batch_val)

                    print_hist_summery(hist)

                    val_loss.append(np.mean(_val_loss))
                    val_acc.append(np.mean(_val_acc))

                    print(
                        "Iteration {}: Train Loss {:6.3f}, Train Accu{:6.3f}, Val Loss {:6.3f}, Val Acc {:6.3f}".format(
                            step, train_loss[-1], train_accuracy[-1], val_loss[-1], val_acc[-1]))

                if step == (max_steps - 1):
                    np.save("/zhome/1c/2/114196/Documents/SegNet-tensorflow/segnet_vgg_bayes_30000/Data/trainloss",
                            train_loss)
                    np.save("/zhome/1c/2/114196/Documents/SegNet-tensorflow/segnet_vgg_bayes_30000/Data/trainacc",
                            train_accuracy)
                    np.save("/zhome/1c/2/114196/Documents/SegNet-tensorflow/segnet_vgg_bayes_30000/Data/valloss",
                            val_loss)
                    np.save("/zhome/1c/2/114196/Documents/SegNet-tensorflow/segnet_vgg_bayes_30000/Data/valacc",
                            val_acc)
                    checkpoint_path = os.path.join(train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

            coord.request_stop()
            coord.join(threads)
