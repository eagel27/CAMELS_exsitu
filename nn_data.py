import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import dtypes
from constants import (BATCHES, )

rng = tf.random.Generator.from_seed(123, alg='philox')


def channels_last(example):
    nn_output = example['exsitu_fraction']
    nn_input = example['image']

    # Reshape input from (3, 128, 128) -> (128, 128, 3)
    nn_input = tf.transpose(nn_input, (1, 2, 0))
    return nn_input, nn_output, example['object_id']


def augment(image, label, object_id):
    seed = rng.make_seeds(2)[0]
    image = tf.image.stateless_random_flip_left_right(image, seed)
    image = tf.image.stateless_random_flip_up_down(image, seed)
    return image, label, object_id


def _per_image_standardization(image):
    """ Linearly scales `image` to have zero mean and unit norm.
    This op computes `(x - mean) / adjusted_stddev`, where `mean` is the average
    of all values in image, and
    `adjusted_stddev = max(stddev, 1.0/sqrt(image.NumElements()))`.
    `stddev` is the standard deviation of all values in `image`. It is capped
    away from zero to protect against division by 0 when handling uniform images.
    Args:
    image: 1-D tensor of shape `[height, width]`.
    Returns:
    The standardized image with same shape as `image`.
    Raises:
    ValueError: if the shape of 'image' is incompatible with this function.
    """
    image = ops.convert_to_tensor(image, name='image')
    num_pixels = math_ops.reduce_prod(array_ops.shape(image))

    image = math_ops.cast(image, dtype=dtypes.float32)
    image_mean = math_ops.reduce_mean(image)

    variance = (math_ops.reduce_mean(math_ops.square(image)) -
                math_ops.square(image_mean))
    variance = gen_nn_ops.relu(variance)
    stddev = math_ops.sqrt(variance)

    # Apply a minimum normalization that protects us against uniform images.
    min_stddev = math_ops.rsqrt(math_ops.cast(num_pixels, dtypes.float32))
    pixel_value_scale = math_ops.maximum(stddev, min_stddev) + 1e-10
    pixel_value_offset = image_mean

    image = math_ops.subtract(image, pixel_value_offset)
    image = math_ops.div_no_nan(image, pixel_value_scale)
    return image


def per_image_standardization(image):
    norm_im = _per_image_standardization(image)
    return norm_im


def preprocessing(image, normalize=True):
    channels_processed = []
    for i in range(5):
        channel_input = image[:, :, i]
        channel_input_normalized = channel_input
        if normalize:
            channel_input_normalized = per_image_standardization(channel_input_normalized)
        channels_processed.append(channel_input_normalized)

    image_processed = tf.stack(channels_processed, axis=2)
    return image_processed


def input_fn_split(mode='train', dataset_str='tng_dataset',
                   batch_size=BATCHES):
    """
    Loads datasets from already split version.
    mode: 'train' or 'test' or 'validation'
    """

    shuffle = mode in ('train', 'validation')
    dataset = tfds.load(
        dataset_str,
        split=mode,
        shuffle_files=shuffle
    )

    if shuffle:
        dataset = dataset.repeat()
        dataset = dataset.shuffle(10000)

    dataset = dataset.map(channels_last, num_parallel_calls=tf.data.AUTOTUNE)
    if mode == 'train':
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

    # Apply data preprocessing
    dataset = dataset.map(lambda x, y, z: (preprocessing(x), y),
                          num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size, drop_remainder=True)
    # fetch next batches while training current one (-1 for autotune)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def get_num_examples(mode='train', dataset_str='tng_dataset'):
    builder = tfds.builder(dataset_str)
    splits = builder.info.splits
    num_examples = splits[mode].num_examples
    return num_examples


def get_data(dataset, batches=10):
    data = dataset.take(batches)
    images, y_true = [], []
    for d in list(data):
        images.extend(d[0].numpy())
        y_true.extend(d[1].numpy())

    images = np.stack(images)
    y_true = np.array(y_true)

    return images, y_true