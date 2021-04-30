# 21DeepLearningProject
 ## The group project for deep learning course in FDU, 2021 spring.

 ### To run the code:
```Bash
cd 21DeepLearningProject
mkdir check_points
mkdir data
mkdir trained_model
```
 - Be sure that the cifar_10 data is at:

 data/cifar-10-batches-py/data_batch_k and also test_batch
 ### Training:
 #### For training with different dataset transformers:
 ```Bash
 python cifar10_transform.py -t k
 ```
 
 - k=0,1,2,3 for different type of transforms
    - 0-notransformation 
    - 1-HorizontalFlip
    - 2-Erasing
    - 3-notransformation+halftrainset

 #### For training with different optimizer and activation:
 Just run corresponding python files directly.

 ### Testing:
 - Save the trained pth file as 
 trained_model/model.pth
 and then
 ```Bash
 python cifar10_test.py -a k
 ```
 - k=0,1,2 for sigmoid, tanh, relu activiation functions, default k = 2.

