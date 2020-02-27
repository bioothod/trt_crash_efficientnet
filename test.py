import argparse
import logging
import os

import numpy as np
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

logger = logging.getLogger('test')

import efficientnet as efn

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='efficientnet-b0', help='Model name')
parser.add_argument('--num_classes', type=int, default=8, help='Num classes')
parser.add_argument('--data_format', type=str, default='channels_last', choices=['channels_first', 'channels_last'], help='Data format: [channels_first, channels_last]')
parser.add_argument('--output_dir', type=str, required=True, help='Output dir where saved models will be stored')
FLAGS = parser.parse_args()

def main():
    params = {
        'num_classes': FLAGS.num_classes,
        'data_format': FLAGS.data_format,
    }
    model, image_size = efn.build_model(FLAGS.model_name, params)

    class MyModel(tf.keras.Model):
        def __init__(self, model, **kwargs):
            super().__init__(**kwargs)

            self.model = model

        @tf.function(input_signature=[tf.TensorSpec([None, image_size * image_size * 3], tf.uint8, name='model_input_images')])
        def __call__(self, inputs):
            images = tf.reshape(inputs, [-1, image_size, image_size, 3])
            images = tf.cast(images, tf.float32)
            images -= 128
            images /= 128

            logits = self.model(images, False)
            return tf.nn.softmax(logits, -1)

    model = MyModel(model)

    logger.info('model {} has been created, image size: {}'.format(FLAGS.model_name, image_size))

    output_dir = os.path.join(FLAGS.output_dir, 'test_saved_model')

    tf.saved_model.save(model, output_dir)

    logger.info('Saved model into {}'.format(output_dir))

    converter = trt.TrtGraphConverterV2(input_saved_model_dir=output_dir)
    converter.convert()
    converter.save('{}_trt'.format(output_dir))

if __name__ == '__main__':
    main()
