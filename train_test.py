import os
import numpy as np
import tensorflow as tf
import math
from inputs import get_filename_list, dataset_inputs
from evaluation import Normal_Loss, cal_loss, per_class_acc, get_hist, print_hist_summery, train_op, MAX_VOTE,var_calculate
from inference import segnet_vgg, segnet_scratch, segnet_bayes_scratch, segnet_bayes_vgg
#When We do the test part, there are several things we need to remember and check:
#1. The path for the saved session which is denoted by train_dir
#2. The inference model that we are using, there are four models now, we don't know if we will write some new model
#3. The MAX VOTE FLAG, if it's true, then we are using max vote to predict the label, if it's FALSE, then we are using mean value to predict the label
#4. The LOSS FLAG, if it's MFL, then we are using median frequency loss, if it's NL, then it's normal cross entropy loss
#5. The path that we save variables
#6. FLAG_BAYES, but this flag is determined automatically according to the inference model
NUM_CLASS = 12 
def Test():
    batch_size = 1
    train_dir = "/zhome/1c/2/114196/Documents/SegNet-tensorflow/segnet_vgg_bayes_MFL_WD_20000/model.ckpt-20000"
    test_dir = "/zhome/1c/2/114196/Documents/SegNet/CamVid/test.txt"
    out_path = "/zhome/1c/2/114196/Documents/SegNet-tensorflow/segnet_vgg_bayes_MFL_WD_20000/Data/Subset/"
    image_w = 480
    image_h = 360
    image_c = 3
    FLAG_INFER = "segnet_bayes_vgg"
    FLAG_MAX_VOTE = False
    #FLAG_LOSS "MFL" median frequency loss, FLAG_LOSS "NL": normal loss
    FLAG_LOSS = "MFL"
    FLAG_CLASS_SUB = True
    #FLAG_CLASS_SUB is utilized to noticify if the test accuracy is calculated with the class index 12
    image_filename, label_filename = get_filename_list(test_dir)
    test_data_tensor = tf.placeholder(tf.float32, shape = [batch_size, image_h, image_w, image_c])
    test_label_tensor = tf.placeholder(tf.int64, shape = [batch_size, image_h,image_w,1])
    phase_train = tf.placeholder(tf.bool, name = 'phase_train')
    keep_prob = tf.placeholder(tf.float32,shape = None, name = 'keep_probability')
    phase_train_dropout = tf.placeholder(tf.bool,name = 'phase_train_dropout')
    
    if (FLAG_INFER == "segnet_scratch"):
        logits = segnet_scratch(test_data_tensor,test_label_tensor,batch_size,phase_train,keep_prob,phase_train_dropout)
        FLAG_BAYES = False
    elif (FLAG_INFER == "segnet_vgg"):
        logits = segnet_vgg(test_data_tensor,test_label_tensor,batch_size,phase_train,keep_prob,phase_train_dropout)
        FLAG_BAYES = False
    elif (FLAG_INFER == "segnet_bayes_scratch"):
        logits = segnet_bayes_scratch(test_data_tensor,test_label_tensor,batch_size,phase_train,keep_prob,phase_train_dropout)
        FLAG_BAYES = True
    elif (FLAG_INFER == "segnet_bayes_vgg"):
        logits = segnet_bayes_vgg(test_data_tensor, test_label_tensor, batch_size,phase_train,keep_prob,phase_train_dropout)
        FLAG_BAYES = True
    else:
        print("The model is not there YET")
    if (FLAG_LOSS == "NL"):
        loss,accuracy,prediction = Normal_Loss(logits, test_label_tensor, NUM_CLASS)
    elif (FLAG_LOSS == "MFL"):
        loss,accuracy,prediction = cal_loss(logits,test_label_tensor)
    else:
        print("The Loss Function is not there YET")
    
    prob = tf.nn.softmax(logits,dim = -1)
    prob = tf.reshape(prob,[image_h,image_w,NUM_CLASS])
    saver = tf.train.Saver()
    print("\n =====================================================")
    print("Testing with model: ", FLAG_INFER)
    print("Loss calculated with: ", FLAG_LOSS)
    print("If inference model is bayes, generate samples by Max Vote: ", FLAG_MAX_VOTE)
    print("ckpt files are saved to: ", train_dir)
    print("Calculate class accuracy without using class 12:", FLAG_CLASS_SUB)
    print(" =====================================================")
    
    with tf.Session() as sess:
        saver.restore(sess, train_dir)
        if FLAG_CLASS_SUB is True:
            hist = np.zeros((NUM_CLASS-1,NUM_CLASS-1))
        else:
            hist = np.zeros((NUM_CLASS,NUM_CLASS))
        images = np.load("/zhome/1c/2/114196/Documents/SegNet-tensorflow/test_image.npy")
        labels = np.load("/zhome/1c/2/114196/Documents/SegNet-tensorflow/test_label.npy")
        #images, labels = get_all_test_data(image_filename,label_filename)
        #uncomment the two lines of code below to only test on specific image
        #images = np.reshape(images[-1],[1,image_h,image_w,image_c])
        #labels = np.reshape(labels[-1],[1,image_h,image_w,1])
        #uncomment the two lines of code below to only test on worst, average and best image
#        index_choose = [62,66,197,232]
#        images = images[index_choose]
#        labels = labels[index_choose]
           
        NUM_SAMPLE = []
        for i in range(30):
            NUM_SAMPLE.append(2*i+1)
        #uncomment the code below to save the result for generating different number of samples in the bayes inference model
#        acc_final = []
#        iu_final = []
#        iu_mean_final = []
#        class_average_accuracy_total = []
        #uncomment the line below to only run for two times. 
        NUM_SAMPLE = [1,30]
        for num_sample_generate in NUM_SAMPLE:
            num_sample_generate = 30
            acc_tot = []
            pred_tot = []
            var_tot = []
            
            if FLAG_CLASS_SUB is True:
               hist = np.zeros((NUM_CLASS-1,NUM_CLASS-1))
               class_average_accuracy = np.zeros((1,NUM_CLASS-1))
            else:
               hist = np.zeros((NUM_CLASS,NUM_CLASS))
               class_average_accuracy = np.zeros((1,NUM_CLASS))
            step = 0
            for image_batch, label_batch in zip(images,labels):
                image_batch = np.reshape(image_batch,[1,image_h,image_w,image_c])
                label_batch = np.reshape(label_batch,[1,image_h,image_w,1])
                #comment the code below to apply the dropout for all the samples
                if num_sample_generate == 1:
                    feed_dict = {test_data_tensor: image_batch, test_label_tensor:label_batch, phase_train: False, keep_prob:0.5, phase_train_dropout:False}
                else:
                    feed_dict = {test_data_tensor: image_batch, test_label_tensor:label_batch, phase_train: False, keep_prob:0.5, phase_train_dropout:True}
                
                fetches = [loss, accuracy, logits, prediction]
                if FLAG_BAYES is False:
                   loss_per, acc_per,logit,pred = sess.run(fetches = fetches, feed_dict = feed_dict)
                   var_one = []
                else:
                   logit_iter_tot = []
                   loss_iter_tot = []
                   acc_iter_tot = []
                   prob_iter_tot = []
                   pred_iter_tot = []
                   for iter_step in range(num_sample_generate):
                       loss_iter_step, acc_iter_step, logit_iter_step,prob_iter_step = sess.run(fetches = [loss,accuracy,logits,prob], feed_dict = feed_dict)
                       loss_iter_tot.append(loss_iter_step)                    
                       acc_iter_tot.append(acc_iter_step)                    
                       logit_iter_tot.append(logit_iter_step)
                       prob_iter_tot.append(prob_iter_step)
                       pred_iter_tot.append(np.reshape(np.argmax(prob_iter_step,axis = -1),[-1]))
                   
                   #uncomment the code below to predict label with max vote!------------------------------------#
                   if FLAG_MAX_VOTE is True:
                      logit,prob_variance,pred = MAX_VOTE(logit_iter_tot,pred_iter_tot,prob_iter_tot)
                      acc_per = np.mean(np.equal(pred,np.reshape(label_batch,[-1])))
                      var_one = var_calculate(pred,prob_variance)
                      pred = np.reshape(pred,[image_h,image_w])
                   else:
                      #loss_per = np.nanmean(loss_iter_tot)
                      acc_per = np.nanmean(acc_iter_tot)
                      logit = np.nanmean(logit_iter_tot,axis = 0)
                      prob_mean = np.nanmean(prob_iter_tot,axis = 0)
                      prob_variance = np.var(prob_iter_tot, axis = 0)
                      pred = np.reshape(np.argmax(prob_mean,axis = -1),[-1]) #pred is the predicted label with the mean of generated samples
                      #THIS TIME I DIDN'T INCLUDE TAU
                      var_one = var_calculate(pred,prob_variance)
                      pred = np.reshape(pred,[image_h,image_w])
                
                
                #loss_tot.append(loss_per)
                acc_tot.append(acc_per)
                pred_tot.append(pred)
                var_tot.append(var_one)

                print("Image Index {}: TEST Accu {:6.3f}".format(step, acc_tot[-1]))
                step = step + 1
                if FLAG_CLASS_SUB is True:
                    acc_class = per_class_acc(logit,label_batch,11)
                    hist+=get_hist(logit,label_batch,Class_Sub = True)
                    class_average_accuracy+=np.reshape(acc_class,[1,NUM_CLASS-1])
                else:
                    acc_class = per_class_acc(logit,label_batch,NUM_CLASS)
                    hist+=get_hist(logit,label_batch,Class_Sub = False)
                    class_average_accuracy+=np.reshape(acc_class,[1,NUM_CLASS])
                
                
                

  
            acc_tot = np.diag(hist).sum()/hist.sum() #pixel-wise accuracy corrected classified label / labeled pixels
            iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist)) #intersection over union= true positive / (true positive+false positive+false negative)
            class_average = class_average_accuracy/step       
            print("Global Accuracy: ", acc_tot) 
            print("Intersection of Union(IoU): ", iu)
            print("mIoU: ", np.nanmean(iu))
            print("Class Average Accuracy over all the images", class_average)
            #print("Class Average Accuracy",class_average_accuracy)
            #uncomment the code below to run this for different number of samples and see the relation between global accuracy and number of samples generated in bayes test part

#            acc_final.append(acc_tot)
#            iu_final.append(iu)
#            iu_mean_final.append(np.nanmean(iu))
#            class_average_accuracy_total.append(np.nanmean(class_average_accuracy))
            #uncomment the code below to save the result!
            if num_sample_generate == max(NUM_SAMPLE):
                if (FLAG_BAYES) is True:
#                    np.save(out_path + "acc_final",acc_final)
#                    np.save(out_path + "iu_final",iu_final)
#                    np.save(out_path + "iu_mean_final",iu_mean_final)
#                    np.save(out_path + "var_tot",var_tot)
#                    np.save(out_path + "prob_variance",prob_variance)
#                    np.save(out_path + "pred_tot",pred_tot)
                    #np.save(out_path + "prob_iter_tot_max_vote",prob_iter_tot)
                
                    print("Total Class Average Accuracy",np.nanmean(class_average))
                    break
                else:
#                    np.save(out_path + "acc_final",acc_final)
#                    np.save(out_path + "iu_final",iu_final)
#                    np.save(out_path + "iu_mean_final",iu_mean_final)
#                    np.save(out_path + "pred_tot",pred_tot)
                    
                    print("Total Class Average Accuracy",np.nanmean(class_average))
                    break #here I set break, because I only want to run for num_generate equal 30
        

        

        
def Train():
    max_steps = 2001
    batch_size = 5
    train_dir = "/zhome/1c/2/114196/Documents/SegNet-tensorflow/segnet_vgg_NL_LRD_WD_15000/"
    image_dir = "/zhome/1c/2/114196/Documents/SegNet/CamVid/train.txt"
    val_dir = "/zhome/1c/2/114196/Documents/SegNet/CamVid/val.txt"
    ckpt_dir = "/zhome/1c/2/114196/Documents/SegNet-tensorflow/segnet_vgg_NL_LRD_WD_15000/model.ckpt-15000"
    image_w = 360
    image_h = 480
    image_c = 3
    FLAG_INFER = "segnet_vgg"
    FLAG_DECAY = True
    #FLAG_LOSS "MFL" median frequency loss, FLAG_LOSS "NL": normal loss
    FLAG_LOSS = "NL"
    #For train the bayes, the FLAG_OPT SHOULD BE SGD, BUT FOR TRAIN THE NORMAL SEGNET, THE FLAG_OPT SHOULD BE ADAM!!!
    #Here, since sometimes we could see the model doesn't learn enough, so we need to increase the number of iterations, this pretrained model actually stands for like segnet_vgg_bayes_17000 model, instead of vgg16 pretrained weight.
    #The vgg16 pretrained is denoted in the inference model. 
    FLAG_PRETRAIN = True
    #Updated: since Gradient Descend is so slow, we use Adam Optimizer instead. 
    epsilon_opt = 1e-4
    image_filename,label_filename = get_filename_list(image_dir)
    val_image_filename, val_label_filename = get_filename_list(val_dir)
    
    with tf.Graph().as_default():
        images_train = tf.placeholder(tf.float32, [batch_size, image_w, image_h, image_c])
        labels_train = tf.placeholder(tf.int64, [batch_size, image_w, image_h, 1])
        phase_train = tf.placeholder(tf.bool, name = "phase_train")
        phase_train_dropout = tf.placeholder(tf.bool,name = "phase_train_dropout")
        keep_prob = tf.placeholder(tf.float32,shape = None, name = "keep_rate")
        images_tr, labels_tr = dataset_inputs(image_filename, label_filename, batch_size)
        images_val, labels_val = dataset_inputs(val_image_filename, val_label_filename, batch_size)
        
        if (FLAG_INFER == "segnet_scratch"):
            logits = segnet_scratch(images_train,labels_train,batch_size,phase_train,keep_prob,phase_train_dropout)
            FLAG_OPT = "ADAM"
         
        elif (FLAG_INFER == "segnet_vgg"):
            logits = segnet_vgg(images_train, labels_train, batch_size, phase_train, keep_prob,phase_train_dropout)
            FLAG_OPT = "ADAM"
            #FLAG_DECAY = True
        elif (FLAG_INFER == "segnet_bayes_scratch"):
            logits = segnet_bayes_scratch(images_train,labels_train,batch_size,phase_train,keep_prob,phase_train_dropout)
            FLAG_OPT = "ADAM"
            #FLAG_DECAY = False
        elif (FLAG_INFER == "segnet_bayes_vgg"):
            logits = segnet_bayes_vgg(images_train, labels_train, batch_size, phase_train, keep_prob,phase_train_dropout)
            FLAG_OPT = "ADAM"
            #FLAG_DECAY = True
        else:
            raise ValueError("The Model Hasn't Finished Yet") 

        if (FLAG_LOSS == "NL"):
            loss,accuracy,prediction = Normal_Loss(logits = logits, labels = labels_train, number_class = NUM_CLASS)
        elif (FLAG_LOSS == "MFL"):
            loss,accuracy,prediction = cal_loss(logits = logits, labels = labels_train)
        else:
            raise ValueError("The Loss Function is not there YET")
        #loss,accuracy,prediction = Normal_Loss(logits = logits, labels = labels_train, number_class = NUM_CLASS)
        #uncomment the code below to calculate loss with median-frequency-class
        #loss,accuracy,prediction = cal_loss(logits = logits, labels = labels_train)
        #remember to set FLAG_DECAY to be False, if don't add any weight decay parameter
        train,global_step = train_op(total_loss = loss, FLAG = FLAG_OPT,FLAG_DECAY = FLAG_DECAY,epsilon_opt = epsilon_opt)
      
        summary_op = tf.summary.merge_all()
        #comment this line below to use pretrained model, since sometimes the model doesn't learn enough, we need to increase the number of iterations
        if FLAG_PRETRAIN is False:
           saver = tf.train.Saver(tf.global_variables())
        #uncomment this line below to use pretrained model. 
        else:        
           saver = tf.train.Saver()
        
        logs_path = "/zhome/1c/2/114196/Documents/SegNet-tensorflow/graph/own/bayes_vgg_decay/"
        
        print("\n =====================================================")
        print("Testing with model: ", FLAG_INFER)
        print("Loss calculated with: ", FLAG_LOSS)
        print("ckpt files are saved to: ", train_dir)
        print("Epsilon used in Adam optimizer: ", epsilon_opt)
        print("Learning Rate Weight Decay",FLAG_DECAY)
        print("Max Steps: ", max_steps)
        print("Use pretrained SegNet model:",FLAG_PRETRAIN)
        print('Start from %d iterations' % int(ckpt_dir.split('-')[-1]))
        print(" =====================================================") 
        with tf.Session() as sess:
            #uncomment these two lines below to use pretrained model
            if FLAG_PRETRAIN is False:
               sess.run(tf.variables_initializer(tf.global_variables()))
               sess.run(tf.local_variables_initializer())
            else:
               saver.restore(sess,ckpt_dir)
        #comment the line below to train from scratch, since sometimes we can see the loss and accuracy doesn't converge enough, which means we will need to train more iterations.
            
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord = coord)            
        #The queue runners basic reference: https://www.tensorflow.org/versions/r0.12/how_tos/threading_and_queues
            train_loss, train_accuracy = [],[]
            val_loss, val_acc = [],[]
            train_writer = tf.summary.FileWriter(logs_path, sess.graph)
            for step in range(max_steps):
               
                image_batch, label_batch = sess.run([images_tr,labels_tr])
                feed_dict = {images_train: image_batch,
                             labels_train: label_batch,
                             phase_train: True,
                             keep_prob: 0.5,
                             phase_train_dropout: True}
 
                _, _loss, _accuracy,summary = sess.run([train, loss, accuracy,summary_op], feed_dict = feed_dict)
                train_loss.append(_loss)
                train_accuracy.append(_accuracy)
                print("Iteration {}: Train Loss{:6.3f}, Train Accu {:6.3f}".format(step,train_loss[-1],train_accuracy[-1]))
                
                if step % 100 == 0:
                    conv_classifier= sess.run(logits,feed_dict = feed_dict) 
                    print('per_class accuracy by logits in training time',per_class_acc(conv_classifier, label_batch,NUM_CLASS)) 
                    train_writer.add_summary(summary,step)
                   
                if step % 1000 == 0:
                    print("start validating.......")
                    _val_loss = []
                    _val_acc = []
                    hist = np.zeros((NUM_CLASS, NUM_CLASS))
                    for test_step in range(int(20)):
                        fetches_valid = [loss, accuracy,logits]
                        image_batch_val, label_batch_val = sess.run([images_val, labels_val])
                        feed_dict_valid = {images_train: image_batch_val,
                                          labels_train: label_batch_val,
                                          phase_train:True,
                                          keep_prob: 1.0,
                                          phase_train_dropout: False}
                                          #since we still using mini-batch, so in the batch norm we set phase_train to be
                                          #true, and because we didin't run the trainop process, so it will not update
                                          #the weight!
                        _loss, _acc, _val_pred = sess.run(fetches_valid, feed_dict_valid)
                        _val_loss.append(_loss)
                        _val_acc.append(_acc)
                        hist += get_hist(_val_pred, label_batch_val,False)

                        
                    print_hist_summery(hist)
        
                    val_loss.append(np.mean(_val_loss))
                    val_acc.append(np.mean(_val_acc))
                    
                    print("Iteration {}: Train Loss {:6.3f}, Train Accu{:6.3f}, Val Loss {:6.3f}, Val Acc {:6.3f}".format(
                    step, train_loss[-1], train_accuracy[-1], val_loss[-1], val_acc[-1]))
                    
                if step == (max_steps-1):
                    if FLAG_PRETRAIN is False:
                       np.save("/zhome/1c/2/114196/Documents/SegNet-tensorflow/segnet_vgg_bayes_NL_EP_17000/Data/trainloss",train_loss)
                       np.save("/zhome/1c/2/114196/Documents/SegNet-tensorflow/segnet_vgg_bayes_NL_EP_17000/Data/trainacc", train_accuracy)
                       np.save("/zhome/1c/2/114196/Documents/SegNet-tensorflow/segnet_vgg_bayes_NL_EP_17000/Data/valloss", val_loss)
                       np.save("/zhome/1c/2/114196/Documents/SegNet-tensorflow/segnet_vgg_bayes_NL_EP_17000/Data/valacc", val_acc)
                       checkpoint_path = os.path.join(train_dir,'model.ckpt')
                       saver.save(sess,checkpoint_path,global_step = step)
                    else:
#                       np.save("/zhome/1c/2/114196/Documents/SegNet-tensorflow/segnet_vgg_bayes_NL_EP_17000/Data/trainloss_pre",train_loss)
#                       np.save("/zhome/1c/2/114196/Documents/SegNet-tensorflow/segnet_vgg_bayes_NL_EP_17000/Data/trainacc_pre", train_accuracy)
#                       np.save("/zhome/1c/2/114196/Documents/SegNet-tensorflow/segnet_vgg_bayes_NL_EP_17000/Data/valloss_pre", val_loss)
#                       np.save("/zhome/1c/2/114196/Documents/SegNet-tensorflow/segnet_vgg_bayes_NL_EP_17000/Data/valacc_pre", val_acc)
                       checkpoint_path = os.path.join(train_dir,'model_pre.ckpt')
                       saver.save(sess,checkpoint_path,global_step = step+int(ckpt_dir.split('-')[-1])) 
                    
                    
                                        
            coord.request_stop()
            coord.join(threads)
            
            
            
        

        
    
    
                 
             
             
             
             
             
             
             
             
             
             
             
             
             
             
             
             
             
        
        

            
              
