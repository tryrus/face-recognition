from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import align.detect_face
from scipy import misc
import numpy as np
import glob
import csv
import os
import facenet
import cv2


gpu_memory_fraction = 0.2
image_size = 160
margin = 44
minsize = 20  # minimum size of face
threshold = [0.6, 0.7, 0.7]  # three steps's threshold
factor = 0.709  # scale factor



def mtcnn_net():
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

    return pnet, rnet, onet

# def load_and_align_data(image_paths, image_size, margin):
#     nrof_samples = len(image_paths)
#     img_list = [None] * nrof_samples
#     pnet, rnet, onet = mtcnn_net()
#     for i in range(nrof_samples):
#         #print(i)
#         img = misc.imread(os.path.expanduser(image_paths[i]))
#         img_size = np.asarray(img.shape)[0:2]
#         bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
#         det = np.squeeze(bounding_boxes[0, 0:4])
#         bb = np.zeros(4, dtype=np.int32)
#         bb[0] = np.maximum(det[0] - margin / 2, 0)
#         bb[1] = np.maximum(det[1] - margin / 2, 0)
#         bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
#         bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
#         cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
#         aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
#         prewhitened = facenet.prewhiten(aligned)
#         img_list[i] = prewhitened
#     images = np.stack(img_list)
#     return images


# def compute_feature(model, folders):
#     features_mean_all = []
#     with tf.Graph().as_default():
#         with tf.Session() as sess:
#             #Load the model
#             facenet.load_model(model)
#             for name in folders:
#                 #print(name)
#                 img_files = glob.glob(os.path.join(name, '*.jpg'))
#                 images = load_and_align_data(img_files, image_size, margin)
#
#                 # Get input and output tensors
#                 images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
#                 embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
#                 phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
#
#                 # Run forward pass to calculate embeddings
#                 feed_dict = {images_placeholder: images, phase_train_placeholder: False}
#                 emb = sess.run(embeddings, feed_dict=feed_dict)
#
#                 features_mean = np.array(emb).mean(axis=0)
#                 features_mean_all.append(list(features_mean))
#
#     return features_mean_all

def load_and_align_data(image_paths, image_size):
    nrof_samples = len(image_paths)
    img_list = [None] * nrof_samples
    pnet, rnet, onet = mtcnn_net()
    for i in range(nrof_samples):
        #print(i)
        img = misc.imread(os.path.expanduser(image_paths[i]))
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

        bounding_boxes = np.squeeze(bounding_boxes[0, 0:4])
        bounding_boxes = bounding_boxes.astype(int)

        crop_face = np.copy(img[max(0, bounding_boxes[1]):(bounding_boxes[3]),
                            max(0, bounding_boxes[0]):(bounding_boxes[2]),])
        crop_face = cv2.resize(crop_face, (image_size, image_size), interpolation=cv2.INTER_CUBIC)

        crop_face = facenet.prewhiten(crop_face)
        img_list[i] = crop_face
    images = np.stack(img_list)
    return images

def compute_feature(model, images):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Load the model
            facenet.load_model(model)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Run forward pass to calculate features
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            emb = sess.run(embeddings, feed_dict=feed_dict)
            features = np.array(emb).mean(axis=0)

    return features


