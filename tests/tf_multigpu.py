#!/usr/bin/env python

#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script is adapted from
# https://github.com/tensorflow/docs/blob/master/site/en/tutorials/distribute/keras.ipynb

import tensorflow_datasets as tfds
import tensorflow as tf
import argparse
import os
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--dummy_data", action='store_true')
parser.add_argument("--run_dir", type=str, default=None)
args = parser.parse_args()

rundir = args.run_dir if args.run_dir is not None else os.path.expandvars('$SCRATCH/nersc-tf-tests')
os.makedirs(os.path.dirname(rundir), exist_ok=True)

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

BUFFER_SIZE = 10000
BATCH_SIZE_PER_REPLICA = 64
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

if args.dummy_data:

    # Use random data
    print('Using random data')
    def generate_dummy_dataset(num_samples=1000, image_shape=(28, 28), num_classes=10):
        images = np.random.rand(num_samples, *image_shape, 1)
        labels = np.random.randint(0, num_classes, size=num_samples)

        return images, labels

    train_images, train_labels = generate_dummy_dataset(num_samples=60000)
    test_images, test_labels = generate_dummy_dataset(num_samples=10000)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(BATCH_SIZE)
    eval_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(BATCH_SIZE)

else:

    # Use mnist data (default)
    print('Using MNIST data')
    data_dir = os.path.join(rundir, 'data')

    datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True,
                               data_dir=data_dir)

    mnist_train, mnist_test = datasets['train'], datasets['test']

    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255

        return image, label

    train_dataset = mnist_train.map(scale).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    eval_dataset = mnist_test.map(scale).batch(BATCH_SIZE)


with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=['accuracy'])

# Define the checkpoint directory to store the checkpoints.
checkpoint_dir = os.path.join(rundir, 'training_checkpoints')
# Define the name of the checkpoint files.
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

# Define a function for decaying the learning rate.
# You can define any decay function you need.
def decay(epoch):
    if epoch < 3:
        return 1e-3
    elif epoch >= 3 and epoch < 7:
        return 1e-4
    else:
        return 1e-5

# Define a callback for printing the learning rate at the end of each epoch.
class PrintLR(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print('\nLearning rate for epoch {} is {}'.format(epoch + 1, model.optimizer.lr.numpy()))
    
# Put all the callbacks together.
tb_log_dir = os.path.join(rundir, 'logs')
callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir=tb_log_dir),
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                       save_weights_only=True),
    tf.keras.callbacks.LearningRateScheduler(decay),
    PrintLR()
]

EPOCHS = 12

model.fit(train_dataset, epochs=EPOCHS, callbacks=callbacks)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

eval_loss, eval_acc = model.evaluate(eval_dataset)

print('Eval loss: {}, Eval accuracy: {}'.format(eval_loss, eval_acc))

savepath = os.path.join(rundir, 'saved_models/my_model.keras')
os.makedirs(os.path.dirname(savepath), exist_ok=True)

model.save(savepath)

unreplicated_model = tf.keras.models.load_model(savepath)

unreplicated_model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy'])

eval_loss, eval_acc = unreplicated_model.evaluate(eval_dataset)

print('Eval loss: {}, Eval Accuracy: {}'.format(eval_loss, eval_acc))

with strategy.scope():
    replicated_model = tf.keras.models.load_model(savepath)
    replicated_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                             optimizer=tf.keras.optimizers.Adam(),
                             metrics=['accuracy'])

    eval_loss, eval_acc = replicated_model.evaluate(eval_dataset)
    print('Eval loss: {}, Eval Accuracy: {}'.format(eval_loss, eval_acc))

print('All done!')

