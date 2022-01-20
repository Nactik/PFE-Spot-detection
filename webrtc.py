# Copyright (c) 2021 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

import asyncio
import base64
import json
import logging
import sys
import threading
import time

#Added import
from datetime import datetime
import os
###

from aiortc import (
    RTCConfiguration,
    RTCPeerConnection,
    RTCSessionDescription,
    MediaStreamTrack,
)
from aiortc.contrib.media import MediaRecorder
import requests

import cv2
from goTo import goTo
import numpy as np
import torch
import pandas as pd


from bosdyn.client.command_line import (Command, Subcommands)
from bosdyn.client.spot_cam.audio import AudioClient
from bosdyn.api.spot_cam import audio_pb2
from webrtc_client import WebRTCClient

logging.basicConfig(level=logging.DEBUG, filename='webrtc.log', filemode='a+')
STDERR = logging.getLogger('stderr')

whT = 320

confThreshold = 0.5
nmsThreshold = 0.3

moving = False

classesFile = 'model/coco.names'
classNames = []
with open(classesFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
# print(classNames)
# print(len(classNames))

#modelConfiguration = 'yolov3-tiny.cfg'
#modelWeights = 'yolov3-tiny.weights'

model = torch.hub.load('../yolov5', 'custom', path='../yolov5/yolov5s.pt', source='local')  # local repo


class InterceptStdErr:
    """Intercept all exceptions and print them to StdErr without interrupting."""
    _stderr = sys.stderr

    def __init__(self):
        pass

    def write(self, data):
        STDERR.error(data)

class WebRTCCommands(Subcommands):
    """Commands related to the Spot CAM's WebRTC service"""

    NAME = 'webrtc'

    def __init__(self, subparsers, command_dict):
        super(WebRTCCommands, self).__init__(subparsers, command_dict,
                                             [WebRTCSaveCommand, WebRTCRecordCommand])


class WebRTCSaveCommand(Command):
    """Save webrtc stream as a sequence of images"""

    NAME = 'save'

    def __init__(self, subparsers, command_dict):
        super(WebRTCSaveCommand, self).__init__(subparsers, command_dict)
        self._parser.add_argument('track', default='video', const='video', nargs='?',
                                  choices=['video'])
        self._parser.add_argument('--sdp-filename', default='h264.sdp',
                                  help='File being streamed from WebRTC server')
        self._parser.add_argument('--sdp-port', default=31102, help='SDP port of WebRTC server')
        self._parser.add_argument('--cam-ssl-cert', default=None,
                                  help="Spot CAM's client cert path to check with Spot CAM server")
        self._parser.add_argument('--dst-prefix', default='h264.sdp',
                                  help='Filename prefix to prepend to all output data')
        self._parser.add_argument('--count', type=int, default=1,
                                  help='Number of images to save. 0 to stream without saving.')

    def _run(self, robot, options):
        # Suppress all exceptions and log them instead.
        sys.stderr = InterceptStdErr()

        if not options.cam_ssl_cert:
            options.cam_ssl_cert = False

        shutdown_flag = threading.Event()
        webrtc_thread = threading.Thread(target=start_webrtc,
                                         args=[shutdown_flag, options, robot, process_frame], daemon=True)
        webrtc_thread.start()

        try:
            webrtc_thread.join()
            print('Successfully saved webrtc images to local directory.')
        except KeyboardInterrupt:
            shutdown_flag.set()
            webrtc_thread.join(timeout=3.0)


class WebRTCRecordCommand(Command):
    """Save webrtc stream as video or audio"""

    NAME = 'record'

    def __init__(self, subparsers, command_dict):
        super(WebRTCRecordCommand, self).__init__(subparsers, command_dict)
        self._parser.add_argument('track', default='video', const='video', nargs='?',
                                  choices=['video', 'audio'])
        self._parser.add_argument('--sdp-filename', default='h264.sdp',
                                  help='File being streamed from WebRTC server')
        self._parser.add_argument('--sdp-port', default=31102, help='SDP port of WebRTC server')
        self._parser.add_argument('--cam-ssl-cert', default=None,
                                  help="Spot CAM's client cert path to check with Spot CAM server")
        self._parser.add_argument('--dst-prefix', default='h264.sdp',
                                  help='Filename prefix to prepend to all output data')
        self._parser.add_argument('--time', type=int, default=10,
                                  help='Number of seconds to record.')

    def _run(self, robot, options):
        # Suppress all exceptions and log them instead.
        sys.stderr = InterceptStdErr()

        if not options.cam_ssl_cert:
            options.cam_ssl_cert = False

        if options.track == 'video':
            recorder = MediaRecorder(f'{options.dst_prefix}.mp4')
        else:
            recorder = MediaRecorder(f'{options.dst_prefix}.wav')

        # run event loop
        loop = asyncio.get_event_loop()
        loop.run_until_complete(record_webrtc(options, recorder))


# WebRTC must be in its own thread with its own event loop.
async def record_webrtc(options, recorder):
    config = RTCConfiguration(iceServers=[])
    client = WebRTCClient(options.hostname, options.username, options.password, options.sdp_port,
                          options.sdp_filename, options.cam_ssl_cert, config,
                          media_recorder=recorder, recorder_type=options.track)
    await client.start()

    # wait for connection to be established before recording
    while client.pc.iceConnectionState != 'completed':
        await asyncio.sleep(0.1)

    # start recording
    await recorder.start()
    try:
        await asyncio.sleep(options.time)
    except KeyboardInterrupt:
        pass
    finally:
        # close everything
        await client.pc.close()
        await recorder.stop()


# WebRTC must be in its own thread with its own event loop.
def start_webrtc(shutdown_flag, options, robot, process_func, recorder=None):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    config = RTCConfiguration(iceServers=[])
    client = WebRTCClient(options.hostname, options.username, options.password, options.sdp_port,
                          options.sdp_filename, options.cam_ssl_cert, config,
                          media_recorder=recorder)

    asyncio.gather(client.start(), process_func(client, options,robot, shutdown_flag),
                   monitor_shutdown(shutdown_flag, client))
    loop.run_forever()

def findObjects(outputs,img,robot):
    hT, wT, cT = img.shape
    humans = outputs.loc[(outputs["class"] == 0) & (outputs["confidence"] >= 0.5)  ]
   
    if not humans.empty : 
        for i in range(len(humans)):
            label = f'{humans.loc[i,"name"]} {humans.loc[i,"confidence"]:.2f}'

            x = int(humans.loc[i,"xmin"])
            y = int(humans.loc[i,"ymin"])

            w = int(humans.loc[i,"xmax"]) - x
            h = int(humans.loc[i,"ymax"]) - y

            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
            cv2.putText(img,label,(x,y-10), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)

        
        localize_human(img,x,y,w,h,robot)
        alert_human_detected(img)

#Function to localize where the detected human is
#The full left back being 0 degree and full right back being 360 degrees
def localize_human(img,x,y,w,h,robot): 
    global moving
    Xcenter = int(x+(w/2))
    Ycenter = int(y+(h/2))
    img_width = int(img.shape[1])
    img_height = int(img.shape[0])
    if y < img_height/2 :
        print("avant (caméra avant)")
    #if y > img_height/2 :
    ratio = Xcenter / img_width
    
    #La position devant est 0°
    position = ratio*360 - 180

    print(f'Humain détecté au degré : {position}')
    if position < -180 :
        print("error")
    if position >= -180 and position < -90 :
        print("arriere gauche")
    if position >= -90 and position < -10 :
        print("avant gauche")
    if position >= -10 and position < 10 :
        print("avant")
    if position >= 10 and position < 90 :
        print("avant droit")
    if position >= 90 and position <= 180 :
        print("arriere droit")    

    robot_audio_client = robot.ensure_client(AudioClient.default_service_name)
    robot_audio_client.set_volume(90.0)
    sound = audio_pb2.Sound(name='bark')
    gain = 20
    if gain:
        gain = max(gain, 0.0)

    robot_audio_client.play_sound(sound, gain)

    goTo(robot, dyaw=position)


def alert_human_detected(img):
    print("Human detected, saving image")
    dir_path = os.getcwd()
    print(dir_path)
    #dir_path = dir_path.replace(os.sep, '/')
    dt_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    filename = dir_path+"/images/human_"+dt_string+".png"
    #print(filename)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    print(cv2.imwrite(filename, img))

# Frame processing occurs; otherwise it waits.
async def process_frame(client, options,robot, shutdown_flag):

    #Define IA
    
    
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,500)
    fontScale              = 1
    fontColor              = (255,255,255)
    thickness              = 1
    lineType               = 2
    fpsCount = 0
    count = 0
    im_count = 0

    start_time = time.time()
    while asyncio.get_event_loop().is_running():
        try:
            frame = await client.video_frame_queue.get()
            
            if options.count == 0:
                fpsCount += 1
                if 0 == 0:
                    #print(fpsCount)
                    pil_image = frame.to_image()
                    cv_image = np.array(pil_image)
                    scale_percent = 60 #percent
                    width = int(cv_image.shape[1] * scale_percent / 100)
                    height = int(cv_image.shape[0] * scale_percent / 100)
                    dim = (width, height)
                    #cv_image = cv2.resize(cv_image, dim, interpolation=cv2.INTER_AREA)
                    Imwidth = cv_image.shape[1]
                    Imheight = cv_image.shape[0]

                    x = 400
                    cv_image = cv_image[x : Imheight, 0 : Imwidth]

                    outputs = model(cv_image)
                    outputDf = outputs.pandas().xyxy[0]

                    findObjects(outputDf,cv_image, robot)
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
                    cv2.imshow('display', cv_image)
                    
                    cv2.waitKey(1)
                continue

            frame.to_image().save(f'{options.dst_prefix}-{count}.jpg')  
            count += 1
            if count >= options.count:
                break
        except :
            pass

    shutdown_flag.set()


# Flag must be monitored in a different coroutine and sleep to allow frame
# processing to occur.
async def monitor_shutdown(shutdown_flag, client):
    while not shutdown_flag.is_set():
        await asyncio.sleep(1.0)

    await client.pc.close()
    asyncio.get_event_loop().stop()
