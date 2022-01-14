# Copyright (c) 2021 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

import argparse

import bosdyn.client

from bosdyn.client.util import (add_common_arguments, setup_logging)

from webrtc2 import WebRTCCommands

from bosdyn.client import spot_cam


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

    result = command_dict[options.command].run(robot, options)
    if args is None and result:
        # Invoked as a CLI, so print result
        print(result)

    return result


if __name__ == '__main__':
    main()
