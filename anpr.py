import os
# load train model
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
# detection from image
import cv2 
import numpy as np
from matplotlib import pyplot as plt
# ocr
import easyocr
import time

import base64


###########################################Setup Paths############################################
CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'
paths = {
    'SCRIPTS_PATH': os.path.join('Tensorflow','scripts'),
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
    'IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images'),
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME)
 }
files = {
    'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}
###########################################Create Label Map############################################
labels = [{'name':'licence', 'id':1}]
with open(files['LABELMAP'], 'w') as f:
    for label in labels:
        f.write('item { \n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')

###########################################Load Train Model From Checkpoint############################################

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)
# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-11')).expect_partial()
@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


class plateRecognizor(object):

    def __init__(self, source, readFile=False):
        self.img_path = ''
        self.img_np = ''
        self.readFile = readFile
        if self.readFile:
            self.img_path = source
        else:
            self.img_path = os.path.join(paths['IMAGE_PATH'], 'test', 'Cars428.png')
            self.img_np = source

    def __del__(self):
        print("Recognizer deleted")


    def detectPlate(self, img_np=''):
        ###########################################Detect from an Image############################################
        
        if img_np!='':
            self.img_np = img_np
            image_np = self.img_np
        else:
            img = cv2.imread(self.img_path)
            image_np = np.array(img)

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np_with_detections,
                    detections['detection_boxes'],
                    detections['detection_classes']+label_id_offset,
                    detections['detection_scores'],
                    category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=5,
                    min_score_thresh=.8,
                    agnostic_mode=False)
        detection_scores = detections['detection_scores'][0]
        # print('detection scores')
        # print(detections['detection_scores'][0])
        if detection_scores>0.7:
            return [True, detections, image_np_with_detections]
        else:
            self.roi_plate = None
            return [False, detections, image_np_with_detections] 

    def detectOCR(self, detections, image_np_with_detections):
        ###########################################Apply OCR to Detection############################################
        detection_threshold = 0.6
        image = image_np_with_detections
        scores = list(filter(lambda x:x> detection_threshold, detections['detection_scores']))
        boxes = detections['detection_boxes'][:len(scores)]
        classes = detections['detection_classes'][:len(scores)]
        width = image.shape[1]
        height = image.shape[0]
        text_array = []
        roi_plate = None
        img_plate_b64 = ''
        for idx, box in enumerate(boxes):
            roi = box*[height, width, height, width]
            region = image[int(roi[0]):int(roi[2]),int(roi[1]):int(roi[3])]
            reader = easyocr.Reader(['en'])
            reader_result = reader.readtext(region)
            for text in reader_result:
                text_array.append(text[1])
            if len(reader_result)>0:
                roi_plate = region
        retval, buffer = cv2.imencode('.png', roi_plate)
        img_plate_b64 = base64.b64encode(buffer)
        img_plate_b64 = img_plate_b64.decode()
        print(text_array)
        return [text_array, img_plate_b64]

category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])


# IMAGE_PATH = os.path.join(paths['IMAGE_PATH'], 'test', 'Cars428.png')

# while(True):
# plateRecognizor1 = plateRecognizor(IMAGE_PATH, True)
# plateRecognizor1.detectPlate()
#     plateRecognizor1.detectOCR()
#     plateRecognizor1.OCRReader()
#     del plateRecognizor1