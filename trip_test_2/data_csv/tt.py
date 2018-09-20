from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from fuel.transformers import Flatten
from fuel.datasets import MNIST
mnist = MNIST(("train",))
data_stream = Flatten(DataStream.default_stream(
    mnist,
    iteration_scheme=SequentialScheme(mnist.num_examples, batch_size=256)))