# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import threading
import json
import logging
import app.util as util

IOT_BASE_TOPIC = 'edge-manager-app'

class Logger(object):
    def __init__(self, device_name, iot_params, buffer_len=10):
        '''
            This class is responsible for sending application logs
            to the cloud via MQTT and IoT Topics
        '''
        self.buffer_len = buffer_len
        self.device_name = device_name
        logging.info("Device Name: %s" % self.device_name)
        self.iot_params = iot_params
        logging.info("Getting the IoT client")
        self.iot_data_client = util.get_client('iot-data', self.iot_params)
        self.logs_buffer = []
        self.__log_lock = threading.Lock()

    def __run_logs_upload_job__(self):        
        '''
            Launch a thread that will read the logs buffer
            prepare a json document and send the logs
        '''
        self.cloud_log_sync_job = threading.Thread(target=self.__upload_logs__)
        self.cloud_log_sync_job.start()
        
    def __upload_logs__(self):
        '''
            Invoked by the thread to publish the latest logs
        '''
        self.__log_lock.acquire(True)
        f = json.dumps({'logs': self.logs_buffer})
        self.logs_buffer = [] # clean the buffer
        try:
            self.iot_data_client.publish( topic='%s/logs/%s' % (IOT_BASE_TOPIC, self.device_name), payload=f.encode('utf-8') )
            logging.info("New log file uploaded. len: %d" % len(f))
        except Exception as e:
            logging.error(e)


        self.__log_lock.release()
 
    def publish_logs(self, data):
        '''
            Invoked by the application, it buffers the logs            
        '''
        if self.__log_lock.acquire(False):
            self.logs_buffer.append(data)
            buffer_len = len(self.logs_buffer)
            self.__log_lock.release()
            # else: job is running, discard the new data
            if buffer_len >= self.buffer_len:
                # run the sync job
                self.__run_logs_upload_job__()
        else:
            logging.error('Failed to acquire lock')