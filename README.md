# dataset_classifier

Implement a classifiner to determine which dataset a image belongs to. The classifier is based on ResNet with pretrained weights. Train the last fully connected layer. The accuracy is pretty high(over 98%). The two datasets are https://www.kaggle.com/nih-chest-xrays/data and https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia. The classifier can serve as metric to evaluate the performance of the undoing-bias approach. Run main.py to generate the weights and see the result.
