## CS295 Final Project - High-Dimensional Memory Augmented Neural Networks

The Omniglot dataset is the most popular benchmark for few-shot image classification. In this project, we are using the omniglot dataset of size 32x32 in greyscale as our evaluation model.

### 1. MANN Architecture with the binary-key memory using analog in-memory computations

For the classification task, we are using CNN as a controller for the HD-MANN architecture. The CNN constitutes of 2 convolutional layers each with 128 filters of shape 5x5 and dilation 2, a max-pooling with a 2x2 filter of stride 2, another 2 convlutional layers each with 128 filters of shape 3x3, another max-pooling layer with 2x2 filter of stride 2 and a fully connected layer with d units. Each convolutional layer uses ReLU activation. The embedding function is of non-linear mapping

$$
f: B^{32\times 32} \rightarrow R^d
$$

### 2. Attention Mechansim for the Key-Value Memory

Let $\alpha$ be the consine similarity and $\epsilon$ the sharpening function. The attention function $\sigma$ is defined as

$$
\sigma(q, K_i) = \frac{\epsilon (\alpha (q, K_i))}{\sum_{j=1^{mn}} \epsilon (\alpha (q, K_j))}.
$$

The sharpening function is a soft absolute(softabs) function proposed in the paper

$$
\epsilon(\alpha) = \frac{1}{1 + e^{-(\beta(\alpha-0.5))}} + \frac{1}{1 + e^{-(\beta(-\alpha-0.5))}}
$$

where $\beta = 10$ is a stiffness parameter.

The learned representations by softabs bering the support vectors of the same class close together in the Hd space, whie pushing the support vectors of different classes apart. We test a 5-way 1-shot problem comparing using softmax and softabs as sharpening functions.

