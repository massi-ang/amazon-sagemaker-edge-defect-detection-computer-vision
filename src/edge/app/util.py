# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import numpy as np
import boto3
import requests
import PIL
import io
import base64

def create_dataset(X, time_steps=1, step=1):
    '''
        Format a timeseries buffer into a multidimensional tensor
        required by the model
    '''
    Xs = []
    for i in range(0, len(X) - time_steps, step):
        v = X[i:(i + time_steps)]
        Xs.append(v)
    return np.array(Xs)

def get_client(service_name, iot_params):
    
    return boto3.client(
        service_name, endpoint_url=iot_params
    )

def create_b64_img_from_mask(mask):
    """Creates binary stream from (1, SIZE, SIZE)-shaped binary mask"""
    img_size = mask.shape[1]
    mask_reshaped = np.reshape(mask, (img_size, img_size))
    img = PIL.Image.fromarray(np.uint8(mask_reshaped)*255)
    img_binary = io.BytesIO()
    img.save(img_binary, 'PNG')
    img_b64 = base64.b64encode(img_binary.getvalue())
    return img_b64