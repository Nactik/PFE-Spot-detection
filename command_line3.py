# Copyright (c) 2021 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

import argparse
from tkinter import E
from acquireLease import acquireLease

import bosdyn.client


from webrtc import WebRTCSaveCommand
from bosdyn.client import spot_cam
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.robot_command import RobotCommandClient, RobotCommandBuilder, blocking_stand
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.spot_cam.audio import AudioClient
from bosdyn.api.spot_cam import audio_pb2
from connect import connect


def main(args=None):
    # Create robot object and authenticate.
    sdk = bosdyn.client.create_standard_sdk('Spot CAM Client')
    spot_cam.register_all_service_clients(sdk)

    robot = connect(sdk)

    assert not robot.is_estopped(), "Robot is estopped. Please use an external E-Stop client, " \
                                "such as the estop SDK example, to configure E-Stop."

    ## TODO : Mettre dans une fonction à part (setup() ??)
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    robot_command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    robot_audio_client = robot.ensure_client(AudioClient.default_service_name)


    sound = audio_pb2.Sound(name='bark')
    with open("assets/dog-bark4.wav", 'rb') as fh:
        data = fh.read()
    
    robot_audio_client.load_sound(sound, data)

    lease_client, lease = acquireLease(robot)

    with LeaseKeepAlive(lease_client, return_at_exit=True):
        # Power on the robot and stand it up.
        robot.time_sync.wait_for_sync()
        robot.power_on()
        blocking_stand(robot_command_client)

        try:
            print("Run detection")
            result = WebRTCSaveCommand.run(robot)
            if args is None and result:
                # Invoked as a CLI, so print result
                print(result)

            return result
        finally:
            # Send a Stop at the end, regardless of what happened.
            robot_command_client.robot_command(RobotCommandBuilder.stop_command())


if __name__ == '__main__':
    main()
