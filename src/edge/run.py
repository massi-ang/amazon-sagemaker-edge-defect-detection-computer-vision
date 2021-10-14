# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import os
import numpy as np
import json
import logging
import PIL.Image
import glob
import random
import re
from timeit import default_timer as timer
from threading import Thread, Event
from flask import Flask
from flask import render_template
from waitress import serve
import time

flask_app = Flask(__name__)

import app

# Configuration constants
#SM_EDGE_AGENT_HOME = os.environ['SM_EDGE_AGENT_HOME']
AGENT_SOCKET = os.environ.get('SM_AGENT_SOCKET', '/tmp/aws.greengrass.SageMakerEdgeManager.sock')
SM_EDGE_MODEL_PATH = os.environ.get('MODEL_PATH','model/dev')
IMAGE_FOLDER = os.environ.get('IMAGE_FOLDER', './static')
CONFIG_FILE_PATH = './models_config.json'
SM_APP_ENV = os.environ.get('SM_APP_ENV', 'dev')
IMG_WIDTH = 224
IMG_HEIGHT = 224

# Set up logging
logging.basicConfig(level=logging.INFO)
logging.debug('Initializing...')

# Loading config file
with open(CONFIG_FILE_PATH, 'r') as f:
    config = json.load(f)

# Send logs to cloud via MQTT topics
logger = app.Logger(os.environ['AWS_IOT_THING_NAME'], os.environ.get('AWS_IOT_ENDPOINT', 'https://acwprzq1nws9h-ats.iot.eu-west-1.amazonaws.com' ), int(os.environ.get('LOG_BATCH_SIZE', '1')))

# Initialize the Edge Manager agent
edge_agent = app.EdgeAgentClient(AGENT_SOCKET)


def create_model_identifier(name, version):
    """Get a compatible string as a combination of name and version"""
    new_name = "%s-%s" % (name, str(version).replace('.', '-'))
    return new_name

def get_model_from_name(name, config_dict):
    """Returns the model dict from the config dict"""
    model_obj = next((x for x in config_dict['models'] if x['name'] == name), None)
    if model_obj is not None:
        return model_obj
    else:
        logging.warning('Model object not found in config')
        return None

def get_square_image(img):
    """Returns a squared image by adding black padding"""
    padding_color = (0, 0, 0)
    width, height = img.size
    if width == height:
        return img
    elif width > height:
        result = PIL.Image.new(img.mode, (width, width), padding_color)
        result.paste(img, (0, (width - height) // 2))
        return result
    else:
        result = PIL.Image.new(img.mode, (height, height), padding_color)
        result.paste(img, ((height - width) // 2, 0))
        return result


def preprocess_image(img, img_width, img_height):
    """Preprocesses the image before feeding it into the ML model"""
    x = get_square_image(img)
    x = np.asarray(img.resize((img_width, img_height))).astype(np.float32)
    x_transposed = x.transpose((2,0,1))
    x_batchified = np.expand_dims(x_transposed, axis=0)
    return x_batchified
       
def run_segmentation_inference(agent, filename):
    """Runs inference on the given image file. Returns prediction and model latency."""

    # Check if model for segmentation is downloaded
    model_name_img_seg = config['mappings']['image-segmentation-app']
    model_is_loaded = any([m['name']==model_name_img_seg for m in model_monitor.models_loaded])
    if not model_is_loaded:
        logging.info('Model for image segmentation not loaded, waiting for deployment...')
        return None, None

    # Get the identifier of the currently loaded model
    model_dict_img_seg = next((x for x in model_monitor.models_loaded if x['name'] == model_name_img_seg), None)
    if not model_dict_img_seg:
        logging.info('Model for image segmentation not loaded, waiting for deployment...')
        return None, None
    model_id_img_seg = model_dict_img_seg['identifier']

    logging.info('\nSegmentation inference with file %s and model %s' % (filename, model_id_img_seg))
    image = PIL.Image.open(filename)
    image = image.convert(mode='RGB')

    # Preprocessing
    x_batchified = preprocess_image(image, IMG_WIDTH, IMG_HEIGHT)

    # Fit into 0-1 range, as the unet model expects this
    x_batchified = x_batchified/255.0

    # Run inference
    t_start = timer()
    y = agent.predict(model_id_img_seg, x_batchified)
    t_stop = timer()
    t_ms = np.round((t_stop - t_start) * 1000, decimals=0)

    y_mask = y[0] > 0.5
    agent.capture_data(model_id_img_seg, x_batchified, y.astype(np.float32))

    return y_mask, t_ms


def run_classification_inference(agent, filename):
    """Runs inference on the given image file. Returns prediction and model latency."""
    # Check if the model for image classification is available
    # The application always uses the latest version of the model in the list of loaded models
    model_name_img_clf = config['mappings']['image-classification-app']
    model_is_loaded = any([m['name']==model_name_img_clf for m in model_monitor.models_loaded])
    if not model_is_loaded:
        logging.info('Model for image classification not loaded, waiting for deployment...')
        return None, None

    # Get the identifier of the currently loaded model
    model_dict_img_clf = next((x for x in model_monitor.models_loaded if x['name'] == model_name_img_clf), None)
    if not model_dict_img_clf:
        logging.info('Model for image classification not loaded, waiting for deployment...')
        return None, None
    model_id_img_clf = model_dict_img_clf['identifier']

    logging.info(f'Classification inference for {filename} with model {model_id_img_clf}')
    image = PIL.Image.open(filename)
    image = image.convert(mode='RGB')

    # Preprocessing
    x_batchified = preprocess_image(image, IMG_WIDTH, IMG_HEIGHT)

    # Run inference with agent and time taken
    
    t_start = timer()
    y = agent.predict(model_id_img_clf, x_batchified)
    t_stop = timer()
    t_ms = np.round((t_stop - t_start) * 1000, decimals=0)

    agent.capture_data(model_id_img_clf, x_batchified, y)
    y = y.ravel()
    logging.info(y)

    img_clf_class_labels = ['normal', 'anomalous']

    for indx, l in enumerate(img_clf_class_labels):
        logging.info('Class probability label "%s": %f' % (img_clf_class_labels[indx], y[indx]))
    return y, t_ms


# Get list of supported model names
models_supported = config['mappings'].values()

class ModelMonitor(object):
    def __init__(self, path):
        self.path = path
        self.m = Thread(target=self.monitor)
        self.m.start()
        self.current_version = 0
        self.models_loaded=[]
        self.loading_model = False

    def get_model_meta(self):
        with open(os.path.join(self.path, 'sagemaker_edge_manifest')) as f:
            model_meta = json.loads(f.read())['metadata']
        return model_meta

    def monitor(self):
        while True:
            meta = self.get_model_meta()
            if int(meta['model_version']) > self.current_version:
                self.load_model()
            time.sleep(1)
    
    def load_model(self):
        
        logging.info('New model version detected. Loading...')

        model_meta = self.get_model_meta()
        
        logging.info(f'Loading model {model_meta}')
        """Loads the model into the edge agent and unloads previous versions if any."""
        # Create a model name string as a concatenation of name and version
        identifier = "%s-%s" % (model_meta['model_name'], model_meta['model_version'])
        current_models = edge_agent.update_models_list()
        if identifier in current_models:
            logging.warn(f'Model already loaded - skipping')
            self.current_version = int(model_meta['model_version'])
            if len([x for x in self.models_loaded if x['identifier'] == identifier]) == 0:
                self.models_loaded.append({'name':model_meta['model_name'], 'version':model_meta['model_version'], 'identifier':identifier})
            return

        self.loading_model = True
        for m in current_models.keys():
            logging.info(f'Unloading model {m}')
            
            edge_agent.unload_model(m)
            self.models_loaded = [x for x in self.models_loaded if x['identifier'] != m ]

        resp = edge_agent.load_model(identifier, self.path)
        
        if resp is None:
            logging.error('It was not possible to load the model. Is the agent running?')
        else:
            logging.info(f'Successfully loaded new model version into agent {model_meta}')
            self.models_loaded.append({'name':model_meta['model_name'], 'version':model_meta['model_version'], 'identifier':identifier})
            self.current_version = int(model_meta['model_version'])

        self.loading_model = False

    def unload_all(self):
        for m in self.models_loaded:
            logging.info("Unloading model %s" % m)
            edge_agent.unload_model(m['identifier'])

    def dispose(self):
        self.t.stop()
        del self.t
        del self.e

model_monitor = ModelMonitor(os.environ.get('MODEL_PATH','/greengrass/v2/work/aws.example.model.defect_detection'))

@flask_app.route('/')
def homepage():
    global model_monitor
    # Get a random image from the directory
    list_img_inf = glob.glob(os.path.join(IMAGE_FOLDER,'**/*.png'))

    if len(list_img_inf) == 0:
        return render_template('main_noimg.html',
            loaded_models=model_monitor.models_loaded
        )

    inference_img_path = random.choice(list_img_inf)
    inference_img_filename = re.search(r'(?<=\/static\/).+$', inference_img_path)[0]

    if not model_monitor.loading_model :
        try:
            # Run inferece on this image
            y_clf, t_ms_clf = run_classification_inference(edge_agent, inference_img_path)
            y_segm, t_ms_segm = run_segmentation_inference(edge_agent, inference_img_path)


            # Synthesize mask into binary image
            if y_segm is not None:
                segm_img_encoded = app.create_b64_img_from_mask(y_segm)
                segm_img_decoded_utf8 = segm_img_encoded.decode('utf-8')
                logging.info('Model latency: t_clf=%fms, t_segm=%fms' % (t_ms_clf, t_ms_segm))
            else:
                segm_img_encoded = None
                segm_img_decoded_utf8 = None

            # Extract predictions from the y array
            # Assuming that the entry at index=0 is the probability for "normal" and the other for "anomalous"
            clf_class_labels = ['normal', 'anomalous']
            if y_clf is not None:
                y_clf_normal = np.round(y_clf[0], decimals=6)
                y_clf_anomalous = np.round(y_clf[1], decimals=6)
                y_clf_class = clf_class_labels[np.argmax(y_clf)]
                logging.info('Model latency: t_classification=%fms' % t_ms_clf)
            else:
                y_clf_normal = None
                y_clf_anomalous = None
                y_clf_class = None

            logger.publish_logs(json.dumps({'classification': { "normal": str(y_clf_normal), "anomalous": str(y_clf_anomalous), "class": str(y_clf_class), "ms":str(t_ms_clf) }}))
            return render_template('main.html',
                loaded_models=model_monitor.models_loaded,
                image_file=inference_img_filename,
                y_clf_normal=y_clf_normal,
                y_clf_anomalous=y_clf_anomalous,
                y_clf_class=y_clf_class,
                y_segm_img=segm_img_decoded_utf8,
                latency_clf=t_ms_clf,
                latency_segm=t_ms_segm
        )
        except Exception as e:
            logging.warn(e)
    return render_template('main.html', loaded_models=model_monitor.models_loaded,
        image_file=inference_img_filename)
    # Return rendered HTML page with predictions
    


if __name__ == '__main__':
    try:
        if SM_APP_ENV == 'prod':
            serve(flask_app, host='0.0.0.0', port=8081)
        elif SM_APP_ENV == 'dev':
            flask_app.run(debug=False, use_reloader=False, host='0.0.0.0', port=8081)
        else:
            raise Exception('SM_APP_ENV needs to be either "prod" or "dev"')

    except KeyboardInterrupt as e:
        pass
    except Exception as e:
        logging.error(e)

    logging.info('Shutting down')

    model_monitor.unload_all()


    # Updating config file
    config['models'] = model_monitor.models_loaded

    with open(CONFIG_FILE_PATH, 'w') as f:
        json.dump(config, f)

    del edge_agent
    del model_monitor
    del logger