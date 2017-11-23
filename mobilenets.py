# -*- coding: utf-8 -*-
"""Convert MobileNet v1 models for Keras into Core ML moodels
    alpha: 1.0, 0.75, 0.50, 0.25
    image_size: 224. 192, 168, 128
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import coremltools
import json
import operator
import os
import urllib

import numpy as np

from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.utils.data_utils import get_file

CLASS_INDEX = None

def get_imagenet_class_labels():
    global CLASS_INDEX
    CLASS_INDEX_PATH = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'

    if CLASS_INDEX is None:
        fpath = get_file('imagenet_class_index.json',
                         CLASS_INDEX_PATH,
                         cache_subdir='models')
    CLASS_INDEX = json.load(open(fpath))

    # Create a list of the class labels
    class_labels = []
    for i in range(len(CLASS_INDEX.keys())):
        class_labels.append(CLASS_INDEX[str(i)][1].encode('ascii', 'ignore'))

    return class_labels

def topn(class_label_probs, top_n=5):
    results = sorted(class_label_probs.items(), key=operator.itemgetter(1), reverse=True)
    return results[:top_n]

if __name__ == "__main__":
    alpha = 1.0
    image_size = 224
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, help="width multiplier")
    parser.add_argument("--image_size", type=int, help="image resolution")
    args = parser.parse_args()
    
    if args.alpha:
        alpha = args.alpha
    if args.image_size:
        image_size = args.image_size
    
    model = MobileNet(weights='imagenet',
                      alpha=alpha,
                      input_shape=(image_size, image_size, 3))

    img_path = 'grace_hopper.jpg'
    img = image.load_img(img_path, target_size=(image_size, image_size))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    print('Predicted:', decode_predictions(preds, top=5)[0])
    # for 1.0 224
    # Predicted:, [(u'n03763968', u'military_uniform', 0.70124328), (u'n02883205', u'bow_tie', 0.18690184), (u'n04350905', u'suit', 0.026838111), (u'n02817516', u'bearskin', 0.021798119), (u'n03929855', u'pickelhaube', 0.0089936024)])

    class_labels = get_imagenet_class_labels()
    coreml_model = coremltools.converters.keras.convert(model,
                                                        input_names='image',
                                                        image_input_names='image',
                                                        image_scale=2./255,
                                                        red_bias=-1.0,
                                                        green_bias=-1.0,
                                                        blue_bias=-1.0,
                                                        class_labels=class_labels,
                                                        output_names='classLabelProbs')

    coreml_model.author = u'Original Paper: Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam. Keras Implementation: François Chollet'
    coreml_model.short_description = 'Detects the dominant objects present in an image from a set of 1000 categories such as trees, animals, food, vehicles, person etc.'
    coreml_model.license = 'MIT License. More information available at https://github.com/fchollet/keras/blob/master/LICENSE'
    coreml_model.input_description['image'] = 'Input image to be classified'
    coreml_model.output_description['classLabel'] = 'Most likely image category'
    coreml_model.output_description['classLabelProbs'] = 'Probability of each category'

    # Core ML names models after file names, dot/decimal point in class name will cause problem. Remove it.
    no_decimal_point = {1.0: '10', 0.75: '075', 0.50: '050', 0.25: '025'}
    coreml_model.save('MobileNet_{}_{}.mlmodel'.format(no_decimal_point[alpha], image_size))
    result = coreml_model.predict({'image': img})

    print(result['classLabel'])
    print(topn(result['classLabelProbs']))
