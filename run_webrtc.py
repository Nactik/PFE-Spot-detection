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
import math

#Added import
from datetime import datetime
import os
###

from aiortc import (
    RTCConfiguration
)

import cv2
import numpy as np
import torch
import pandas as pd
from bosdyn.client.command_line import (Command, Subcommands)
from webrtc_client import WebRTCClient
from goTo import signal
from goTo import goTo
from dotenv import load_dotenv


logging.basicConfig(level=logging.DEBUG, filename='webrtc.log', filemode='a+')
STDERR = logging.getLogger('stderr')

light = False
moving = False
whT = 320
lastPostion = 0;

confThreshold = 0.5
nmsThreshold = 0.3

 
model = torch.hub.load('../yolov5', 'custom', path='../yolov5/yolov5s.pt', source='local')  # local repo

class InterceptStdErr:
    """Intercept all exceptions and print them to StdErr without interrupting."""
    _stderr = sys.stderr

    def __init__(self):
        pass

    def write(self, data):
        STDERR.error(data)


class WebRTCSaveCommand():
    """Save webrtc stream as a sequence of images"""

    NAME = 'save'

    def __init__(self):
        load_dotenv()

        self.options = {
            'username' : os.getenv('ROBOT_USERNAME'),
            'hostname': os.getenv("ROBOT_IP"),
            'password' : os.getenv('ROBOT_PASSWORD'),
            'track': 'video',
            'sdp_filename' : 'h264.sdp',
            'sdp_port' : 31102,
            'cam_ssl_cert' : False,
        }

    def run(self,robot):
        # Suppress all exceptions and log them instead.
        sys.stderr = InterceptStdErr()

        shutdown_flag = threading.Event()
        webrtc_thread = threading.Thread(target=start_webrtc,
                                         args=[shutdown_flag, self.options, robot,process_frame], daemon=True)
        webrtc_thread.start()

        try:
            webrtc_thread.join()
            print('Successfully saved webrtc images to local directory.')
        except KeyboardInterrupt:
            shutdown_flag.set()
            webrtc_thread.join(timeout=3.0)




# WebRTC must be in its own thread with its own event loop.
def start_webrtc(shutdown_flag, options, robot, process_func, recorder=None):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    config = RTCConfiguration(iceServers=[])
    client = WebRTCClient(options["hostname"], options["username"], options["password"], options["sdp_port"],
                          options["sdp_filename"], options["cam_ssl_cert"], config,
                          media_recorder=recorder)

    asyncio.gather(client.start(), process_func(client, options, robot, shutdown_flag),
                   monitor_shutdown(shutdown_flag, client))
    loop.run_forever()

async def findObjects(outputs,img,robot):
    global light
    hT, wT, cT = img.shape
    humans = outputs.loc[(outputs["class"] == 0) & (outputs["confidence"] >= 0.5)  ]

    if not humans.empty and not light :
        await signal(robot,1)
        light = True
    
    if humans.empty and light:
        await signal(robot,2)
        light = False

   
    if not humans.empty : 

        middle = int(img.shape[1])/2
        
        humans["xcenter"] = humans["xmin"] + ((humans["xmax"] - humans["xmin"])/2)
        humans["distance"] = abs((humans["xcenter"] - middle))

        for i in range(len(humans)):
            label = f'{humans.loc[i,"name"]} {humans.loc[i,"confidence"]:.2f}'

            ## On récupère les coordonnées de chaque humains
            x = int(humans.loc[i,"xmin"])
            y = int(humans.loc[i,"ymin"])

            w = int(humans.loc[i,"xmax"]) - x
            h = int(humans.loc[i,"ymax"]) - y

            ## On dessine les bounding box autour d'eux
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
            cv2.putText(img,label,(x,y-10), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)
        

        # on récupère l'human avec le plus de confidence
        mostConfident = humans.iloc[humans['confidence'].idxmax()]

        xMostConf = int(mostConfident["xmin"])
        yMostConf = int(mostConfident["ymin"])

        wMostConf = int(mostConfident["xmax"]) - xMostConf
        hMostConf = int(mostConfident["ymax"]) - yMostConf

        print(f'X most Confident : {xMostConf}')

        await localize_human(img,xMostConf,yMostConf,wMostConf,hMostConf,robot)
        await alert_human_detected(img)

    # On affiche l'image sur l'ecran
    cv_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow('display', cv_image)
                
    cv2.waitKey(1)


#Function to localize where the detected human is
#The full left back being 0 degree and full right back being 360 degrees
async def localize_human(img,x,y,w,h,robot): 
    global moving
    
    # On calcul la position en degré de la personne
    Xcenter = int(x+(w/2))
    img_width = int(img.shape[1])


    ratio = Xcenter / img_width
    position = ratio*360 - 180

    """
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
    """
    print(f'Moving : {moving}')   

    # On check si le robot n'est pas déjà en mouvement
    if not moving  :
        ##On bouge
        await goTo(robot,dyaw=position)

    # SOLUTION 1 pour tester si le robot à fini son mouvement
    # On check si le robot est en train de bouger et si la position de la personne est revenue à 0 (avec un offset)
    if moving and int(position)>-5 and int(position) <5 :
        moving = False

## Fonction pour enregistrer l'image lors d'une détection
async def alert_human_detected(img):
    print("Human detected, saving image")
    dir_path = os.getcwd()
    #print(dir_path)
    #dir_path = dir_path.replace(os.sep, '/')
    dt_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    filename = dir_path+"/images/human_"+dt_string+".png"
    #print(filename)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    print(cv2.imwrite(filename, img))

# Fonction qui récupère chaque frame de la SpotCam
# Frame processing occurs; otherwise it waits.
async def process_frame(client, options,robot, shutdown_flag):

    #Define IA
    global model
    
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
            # On récupère la frame
            frame = await client.video_frame_queue.get()
            
            fpsCount += 1
            ## On prend 1 frame sur 10 pour limiter le flux
            if fpsCount%10 == 0:
                ## On resize l'image
                pil_image = frame.to_image()
                cv_image = np.array(pil_image)
                scale_percent = 60 #percent
                width = int(cv_image.shape[1] * scale_percent / 100)
                height = int(cv_image.shape[0] * scale_percent / 100)
                dim = (width, height)
                #cv_image = cv2.resiz e(cv_image, dim, interpolation=cv2.INTER_AREA)
                Imwidth = cv_image.shape[1]
                Imheight = cv_image.shape[0]
                x = 400
                cv_image = cv_image[x : Imheight, 0 : Imwidth]
                
                ## On lance l'analyse de l'image par l'IA (fonction model)
                outputs = model(cv_image)
                outputDf = outputs.pandas().xyxy[0] 

                await findObjects(outputDf,cv_image,robot)
            continue
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
