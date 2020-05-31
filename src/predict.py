#!/usr/bin/env python -W ignore::DeprecationWarning

import argparse
import json
import logging
import logging.config
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pdb
from PIL import Image
import scipy
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from tensorflow.keras import layers

logging.config.fileConfig("../../../logging.ini")
logger=logging.getLogger()

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
np.set_printoptions(formatter={'float_kind':'{:f}'.format})

dsize=(224, 224)
num_classes=102

def parse_args():
    # python -W ignore predict.py -h
    parser = argparse.ArgumentParser(description="Udacity Image Classifier",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_path', nargs='?', default='../training_1/best_weights.h5',
                        help='Model Path')
    parser.add_argument('--img_path', nargs='?', default='../test_images/training_data_check.jpg',
                        help='Image Location')
    parser.add_argument('--top_k', nargs='?', default=5,
                    help='Return the top K most likely classes')
    parser.add_argument('--category_names', nargs='?', default='../label_map.json',
                    help='Path to a JSON file mapping labels to flower names:')
    return parser.parse_args()

def process_image(image):
    global dsize
    image=tf.convert_to_tensor(image,tf.float32)
    image=tf.image.resize(image, dsize)
    image/=255
    return image

def predict(image_path=None, model=None, top_k=None):
    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_test_image = process_image(test_image)    
    processed_test_image=np.expand_dims(processed_test_image,0)
    probs=model.predict(processed_test_image)
    return tf.nn.top_k(probs, k=top_k)

def load_json(path):
    with open(path, 'r') as f:
        class_names = json.load(f)
    return class_names

def filtered(classes=None,class_names=None):
    return [class_names.get(str(key)) if key else "Placeholder" for key in classes.numpy().squeeze().tolist()]

def run():
    args = parse_args()
    logger.info(args)
    class_names=load_json(args.category_names)
    logger.info(f"Category Names Loaded from {args.category_names}")

    classifier_url = "https://tfhub.dev/google/imagenet/mobilenet_v1_050_160/classification/4"
    classifier = tf.keras.Sequential([
        hub.KerasLayer(classifier_url, input_shape=(224,224,3)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes,activation='softmax'),
    ])    
    logger.info(classifier.summary())

    earlystop_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy', min_delta=0.0001,
    patience=1)

    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
    classifier.compile(optimizer='adam',\
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(),\
                   metrics=['accuracy',
                           ],
                   callbacks=[cp_callback,earlystop_callback]
                  )
    logger.info("Model Architecture Built")
    classifier.load_weights(args.model_path)
    logger.info("Model Weights Loaded")
    probs, classes = predict(image_path=args.img_path, model=classifier, top_k=args.top_k)
    pred_dict={filtered(classes,class_names)[i]: probs[0][i].numpy() for i in range(len(filtered(classes,class_names)))} 
    logger.info("**"*50)
    logger.info(f"File: {args.img_path} \n\n\n Probability: {probs[0]}\n Classes: {classes} \n Labels: {filtered(classes,class_names)}\n Dictionary: {pred_dict}\n")
    return probs, classes,filtered(classes,class_names),pred_dict    
    
if __name__ == '__main__':
    logger.info("**"*50)
    logger.info("Starting Prediction Process")
    run()
    logger.info("Code Run Completed")
    logger.info("**"*50)
