# Semi-Supervised-Learning
The goal of the project is to train a CNN on CIFAR 10 using 250 annotated images. The other images without annotations can also be used. This is a semi-supervised learning approach. The objective is to ensure that the CNN performs well on the CIFAR 10 test set and does not overfit too much.

The lack of labeled data is a major obstacle in applying deep learning to practical domains such as medical imaging, which can lead to poor model performance. To overcome this challenge, it is possible to leverage unlabeled images using a semi-supervised learning approach called FixMatch. FixMatch is a simpler method than previous approaches such as UDA and ReMixMatch. The article also explains how FixMatch improved the state of the art in semi-supervised learning with a median accuracy of 78 percent and a maximum accuracy of 84 percent on CIFAR-10 using only 10 labeled images.

Consistency regularization is an important component of advanced semi-supervised algorithms. It uses unlabeled data by assuming that the model should produce similar predictions when fed with perturbed versions of the same image. The idea has been popularized by several research works and is implemented through a loss function that combines a supervised classification loss and a loss on unlabeled data. Pseudo-labeling is a method that uses the model itself to obtain artificial labels for unlabeled data. It uses hard labels and retains only those whose highest class probability is above a predefined threshold. These two methods are often used together to improve the performance of semi-supervised models.





