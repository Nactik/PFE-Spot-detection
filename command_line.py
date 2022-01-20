# Copyright (c) 2021 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

import argparse
from acquireLease import acquireLease

import bosdyn.client
from bosdyn.client.util import (add_common_arguments, setup_logging)


from webrtc import WebRTCCommands
from bosdyn.client import spot_cam
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.robot_command import RobotCommandClient, RobotCommandBuilder, blocking_stand
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.spot_cam.audio import AudioClient
from bosdyn.api.spot_cam import audio_pb2
from connect import connect



def register_all_commands(subparsers, command_dict):
    COMMANDS = [
        WebRTCCommands
    ]

    for register_command in COMMANDS:
        register_command(subparsers, command_dict)


def main(args=None):
    """Command-line interface for interacting with Spot CAM"""
    parser = argparse.ArgumentParser(prog='bosdyn.api.spot_cam', description=main.__doc__)
    add_common_arguments(parser)

    command_dict = {}  # command name to fn which takes parsed options
    subparsers = parser.add_subparsers(title='commands', dest='command')
    subparsers.required = True

    register_all_commands(subparsers, command_dict)

    options = parser.parse_args(args=args)

    setup_logging(verbose=options.verbose)

    # Create robot object and authenticate.
    sdk = bosdyn.client.create_standard_sdk('Spot CAM Client')
    spot_cam.register_all_service_clients(sdk)

    robot = sdk.create_robot(options.hostname)

    robot2 = connect()

    assert not robot2.is_estopped(), "Robot is estopped. Please use an external E-Stop client, " \
                                "such as the estop SDK example, to configure E-Stop."

    robot_state_client = robot2.ensure_client(RobotStateClient.default_service_name)
    robot_command_client = robot2.ensure_client(RobotCommandClient.default_service_name)

    sound = audio_pb2.Sound(name='bark')
    with open("asset/dog-bark4.wav", 'rb') as fh:
        data = fh.read()
    
    robot_audio_client = robot2.ensure_client(AudioClient.default_service_name)
    robot_audio_client.load_sound(sound, data)

    lease_client, lease = acquireLease(robot2);

    try :
        with LeaseKeepAlive(lease_client, return_at_exit=True):
            # Power on the robot and stand it up.
            robot2.time_sync.wait_for_sync()
            robot2.power_on()
            blocking_stand(robot_command_client)

            try:
                result = command_dict[options.command].run(robot2, options)
                if args is None and result:
                    # Invoked as a CLI, so print result
                    print(result)

                return result
            finally:
                # Send a Stop at the end, regardless of what happened.
                robot_command_client.robot_command(RobotCommandBuilder.stop_command())
                
    finally :
        print("coucou")
        #lease_client.return_lease(lease)

if __name__ == '__main__':
    main()
