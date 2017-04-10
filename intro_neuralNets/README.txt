Ran using:

Software:
	Anaconda
	Tensorflow backend
	Keras DeepLearning library

Requires cats_and_dogs_medium from http://www.cs.virginia.edu/~connelly/share/cats_and_dogs_medium.zip
or other cats/dogs dataset from the web


For GPU:
	Install Nvidia CUDA
	Install Nvidia drivers
	Install cuDNN
	Create Python/Conda Environment
	pip install --upgrade tensorflow-gpu
	pip install keras

Details:
Project mainly testing various code on neural nets with Keras
	_1mlp.py confusion matrix
	_2cnn.py testing various convolutions
	_3catsdogs.py fine-tuned VGG-16 model to distinguish between cats and dogs
	_4catsdogs_test.py test catsdogs model on a single image