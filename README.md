# 21DeepLearningProject
 The group project for deep learning course in FDU, 2021 spring.

 To run the code:
 cd 21DeepLearningProject
 mkdir check_points
 mkdir data
 mkdir trained_model

 Be sure that the cifar_10 data is at:
 data/cifar-10-batches-py/data_batch_k and also test_batch

 Testing:
 Save the trained pth file as
 trained_model/model.pth
 and then
 python cifar10_test.py -a k
 k=0,1,2 for sigmoid, tanh, relu activiation functions
 default k = 2

