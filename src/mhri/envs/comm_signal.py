# from abc import ABC, abstractmethod
from gymnasium import spaces
import numpy as np
import string

# implement an abstract class for communication signals that can be used in the environment action space
# including:
# - discrete signals: spaces.Discrete(5)
# - continuous signals: spaces.Box
# - single-modal signals:
# - text: spaces
# - image: spaces.Box


# implement a superclass for the communication signals
class CommunicationSignal(spaces.Space):
    """
    Abstract base class for different types of communication signals.
    """

    def __init__(self):
        self.space = None

    # @abstractmethod
    def emit(self):
        # Emit the signal
        raise NotImplementedError

    # @abstractmethod
    def sample(self):
        # Sample a random signal in the signal space
        raise NotImplementedError


class DiscreteSignal(CommunicationSignal):
    """
    A discrete signal class that uses predefined flags.
    """

    def __init__(self, size, start=0):
        self.size = size
        self.space = spaces.Discrete(size, start=start)

    def emit(self, signal):
        return signal

    def sample(self):
        return self.space.sample()


class ContinuousSignal(CommunicationSignal):
    """
    A continuous signal class that can range over a continuous spectrum.
    """

    def __init__(self, dim):
        self.dim = dim
        self.space = spaces.Box(low=-1, high=1, shape=(dim,))

    def emit(self, value):
        return value

    def sample(self):
        return self.space.sample()


class TextSignal(CommunicationSignal):
    """
    A text signal class for handling text communication.
    """

    def __init__(self, max_length):
        self.max_length = max_length
        self.space = spaces.Text(
            max_length=max_length,
            charset=string.ascii_lowercase
            + string.digits
            + string.punctuation
            + string.whitespace
            + string.ascii_uppercase
            + "\n",
        )

    def emit(self, text):
        return text

    def sample(self):
        return self.space.sample()


class ImageSignal(CommunicationSignal):
    """
    An image signal class for handling image communication.
    """

    def __init__(self, image_data):
        self.image_data = image_data
        self.space = spaces.Box(low=0, high=255, shape=image_data.shape)

    def emit(self):
        return self.image_data

    def sample(self):
        return self.space.sample()


""" TEST CASES """
# test discrete signal
discrete_signal: CommunicationSignal = DiscreteSignal(5)
assert discrete_signal.space == spaces.Discrete(5)
assert discrete_signal.emit(3) == 3
assert discrete_signal.sample() in range(5)


# test continuous signal
continuous_signal: CommunicationSignal = ContinuousSignal(5)
assert continuous_signal.space == spaces.Box(low=-1, high=1, shape=(5,))
random_value = np.random.rand(5)
assert continuous_signal.emit(random_value).all() == random_value.all()
assert continuous_signal.sample().shape == (5,)


# test text signal
text_signal: CommunicationSignal = TextSignal(10)
assert text_signal.space == spaces.Text(
    max_length=10,
    charset=string.ascii_lowercase
    + string.digits
    + string.punctuation
    + string.whitespace
    + string.ascii_uppercase
    + "\n",
)
assert text_signal.emit("Hello") == "Hello"
assert len(text_signal.sample()) <= 10


# test image signal
image_data = np.random.randint(0, 255, (32, 32, 3))
image_signal: CommunicationSignal = ImageSignal(image_data)
assert image_signal.space == spaces.Box(low=0, high=255, shape=image_data.shape)
assert image_signal.emit().all() == image_data.all()
assert image_signal.sample().shape == (32, 32, 3)
