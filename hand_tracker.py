import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

import cv2
cap = cv2.VideoCapture(0)

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")


# ## Object detection imports
# Here are the imports from the object detection module.

# In[3]:

from utils import label_map_util

from utils import visualization_utils as vis_util

pts1 = np.float32([[69,108],[595,15],[42,390],[596,469]])
pts2 = np.float32([[0,0],[640,0],[0,480],[640,480]])

M = cv2.getPerspectiveTransform(pts1,pts2)

# while(True):
#   ret, image_np = cap.read()
#   image_np = cv2.resize(image_np, (640, 480))
#   imgae_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
#   cv2.imshow('original', image_np)

#   image_np = cv2.warpPerspective(image_np,M,(640,480))
#   cv2.imshow('warp', image_np)
#   cv2.waitKey(30)
# # Model preparation 

# ## Variables
# 
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.  
# 
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# In[4]:

# What model to download.
MODEL_NAME = 'deep-finger'
MODEL_NAME_F = 'finger-deep'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_CKPT_F = MODEL_NAME_F + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'deep-finger.pbtxt'
PATH_TO_LABELS_F = 'finger-deep.pbtxt'

NUM_CLASSES = 1


# ## Load a (frozen) Tensorflow model into memory.

# In[6]:

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


## FINGER ###############################
detection_graph_f = tf.Graph()
with detection_graph_f.as_default():
  od_graph_def_f = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT_F, 'rb') as fid:
    serialized_graph_f = fid.read()
    od_graph_def_f.ParseFromString(serialized_graph_f)
    tf.import_graph_def(od_graph_def_f, name='')

# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

label_map_f = label_map_util.load_labelmap(PATH_TO_LABELS_F)
categories_f = label_map_util.convert_label_map_to_categories(label_map_f, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index_f = label_map_util.create_category_index(categories_f)


# ## Helper code

# In[8]:

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# # Detection

def crop(img, xmin, ymin, xmax, ymax):
    return img[ymin:ymax, xmin:xmax]

# In[10]:


with tf.Session(graph=detection_graph) as sess:
    with tf.Session(graph=detection_graph_f) as sess_f:
        while True:
          ret, image_np = cap.read()
          image_np = cv2.resize(image_np, (640, 480))
          image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
          cv2.imshow('original', image_np)
          img = image_np.copy()

          # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
          image_np_expanded = np.expand_dims(image_np, axis=0)
          image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
          # Each box represents a part of the image where a particular object was detected.
          boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
          # Each score represent how level of confidence for each of the objects.
          # Score is shown on the result image, together with the class label.
          scores = detection_graph.get_tensor_by_name('detection_scores:0')
          classes = detection_graph.get_tensor_by_name('detection_classes:0')
          num_detections = detection_graph.get_tensor_by_name('num_detections:0')
          # Actual detection.
          (boxes, scores, classes, num_detections) = sess.run(
              [boxes, scores, classes, num_detections],
              feed_dict={image_tensor: image_np_expanded})
          # Visualization of the results of a detection.

          vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              category_index,
              use_normalized_coordinates=True,
              line_thickness=8,
              min_score_thresh=0.05,
              max_boxes_to_draw=1)

          cv2.imshow('object detection', cv2.resize(image_np, (800,600)))

          chosen = boxes[0][0]
          tlx = max(int(chosen[1]*640) - 5, 0)
          tly = max(int(chosen[0]*480) - 5, 0)
          brx = int(chosen[3]*640)
          bry = int(chosen[2]*480)

          image_np = crop(img, tlx, tly, brx, bry)
          cv2.imshow('hand', image_np)

          ###########################################################

          image_np_expanded = np.expand_dims(image_np, axis=0)
          image_tensor = detection_graph_f.get_tensor_by_name('image_tensor:0')
          # Each box represents a part of the image where a particular object was detected.
          boxes = detection_graph_f.get_tensor_by_name('detection_boxes:0')
          # Each score represent how level of confidence for each of the objects.
          # Score is shown on the result image, together with the class label.
          scores = detection_graph_f.get_tensor_by_name('detection_scores:0')
          classes = detection_graph_f.get_tensor_by_name('detection_classes:0')
          num_detections = detection_graph_f.get_tensor_by_name('num_detections:0')
          # Actual detection.
          (boxes, scores, classes, num_detections) = sess_f.run(
              [boxes, scores, classes, num_detections],
              feed_dict={image_tensor: image_np_expanded})
          # Visualization of the results of a detection.

          vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              category_index,
              use_normalized_coordinates=True,
              line_thickness=8,
              min_score_thresh=0.05,
              max_boxes_to_draw=1)

          cv2.imshow('finger', image_np)

          ###########################################################

          if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break