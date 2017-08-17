import os.path
import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import numpy as np


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = sess.graph
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name) 
    
    return image_input, keep_prob, layer3_out, layer4_out, layer7_out
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    weight_initializer = variance_scaling_initializer()
    ## fcn layers
    layer3_fcn = tf.layers.conv2d(vgg_layer3_out, num_classes,
                                kernel_size=(1, 1), strides=(1, 1),
                                kernel_initializer=weight_initializer,
                                name="layer3_fcn", padding="SAME",
                                activation=tf.nn.relu)

    layer4_fcn = tf.layers.conv2d(vgg_layer4_out, num_classes,
                                kernel_size=(1, 1), strides=(1, 1),
                                kernel_initializer=weight_initializer,
                                name="layer4_fcn", padding="SAME",
                                activation=tf.nn.relu)

    layer7_fcn = tf.layers.conv2d(vgg_layer7_out, num_classes,
                                kernel_size=(1, 1), strides=(1, 1),
                                kernel_initializer=weight_initializer,
                                name="layer7_fcn", padding="SAME",
                                activation=tf.nn.relu)

    ## upsampling and skipping
    layer7_up = tf.layers.conv2d_transpose(layer7_fcn, num_classes,
                                        kernel_size=(4, 4), strides=(2, 2),
                                        kernel_initializer=weight_initializer,
                                        name="layer7_up", padding="SAME",
                                        activation=tf.nn.relu)

    layer4_skip = tf.add(layer4_fcn, layer7_up, name="layer4_skip")

    layer4_up = tf.layers.conv2d_transpose(layer4_skip, num_classes,
                                        kernel_size=(4, 4), strides=(2, 2),
                                        kernel_initializer=weight_initializer,
                                        name="layer4_up", padding="SAME",
                                        activation=tf.nn.relu)

    layer3_skip = tf.add(layer3_fcn, layer4_up, name="layer3_skip")

    class_heatmap = tf.layers.conv2d_transpose(layer3_skip, num_classes,
                                        kernel_size=(16, 16), strides=(8, 8),
                                        kernel_initializer=weight_initializer,
                                        name="class_heatmap", padding="SAME",
                                        activation=tf.nn.relu)

    return class_heatmap
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, [-1, num_classes])
    labels = tf.reshape(correct_label, [-1, num_classes])
    
    xentropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.reduce_mean(xentropy)
    
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return logits, train_op, loss
tests.test_optimize(optimize)


def augment_data(images, gt_images):
    """Augument data by some image transformation,
    return pair of image patches..
    Only support horizontal flipping for now
    """
    # flipping horizontally
    hf_images = images[:, :, ::-1, :]
    hf_gt_images = gt_images[:, :, ::-1, :]
    aug_images = np.concatenate([images, hf_images], axis=0)
    aug_gt_images = np.concatenate([gt_images, hf_gt_images], axis=0)
    return aug_images, aug_gt_images


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    ## training
    for epoch in range(epochs):
        batches = get_batches_fn(batch_size)
        for b, (seed_images, seed_gt_images) in enumerate(batches):
            ## for the intrusive test `tests.test_train_nn`
            if seed_images.ndim == 4:
                images, gt_images = augment_data(seed_images, seed_gt_images)
            else:
                images, gt_images = seed_images, seed_gt_images
            _, loss_val = sess.run([train_op, cross_entropy_loss],
                                    feed_dict={
                                        input_image: images,
                                        correct_label: gt_images,
                                        keep_prob: 0.5,
                                        learning_rate: 1e-4 })
            if b % 10 == 0:
                print("epoch %i batch %i loss=%.3f" % (epoch, b, loss_val))

    ## save model
    saver = tf.train.Saver()
    saver.save(sess,'./models/model.ckpt')
    saver.export_meta_graph('./models/model.meta')
    tf.train.write_graph(sess.graph_def, "./models", "model.pb", False)
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576) #(32, 128)
    n_epochs = 100
    batch_size = 8
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network
        # see the implementation of `train_nn`

        # TODO: Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)

        # TODO: Train NN using the train_nn function
        model_output = layers(layer3_out, layer4_out, layer7_out, num_classes)

        target = tf.placeholder(dtype=tf.float32, shape=[None, None, None, num_classes])

        learning_rate = tf.placeholder(dtype=tf.float32)
        
        logits, train_op, loss = optimize(model_output, target, learning_rate, num_classes)

        sess.run(tf.global_variables_initializer())
        
        train_nn(sess, n_epochs, batch_size, get_batches_fn, train_op, loss, input_image,
                target, keep_prob, learning_rate)

        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
