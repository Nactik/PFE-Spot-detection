# Copyright (c) 2021 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

"""Command the robot to go to an offset position using a trajectory command."""

from curses.ascii import NUL
from glob import glob
from string import ascii_uppercase
import time
from bosdyn.api.basic_command_pb2 import RobotCommandFeedbackStatus
import bosdyn.client.util
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.robot_command import RobotCommandClient, RobotCommandBuilder
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import ODOM_FRAME_NAME, VISION_FRAME_NAME, BODY_FRAME_NAME, get_se2_a_tform_b
import asyncio
import run_webrtc

cmd_id = None
hasFinishedMoving = False
isDone = False

def checkDoneMoving(result):
    global cmd_id
    run_webrtc.moving = True
    cmd_id = result.result()

    print(f"Command done, cmd_id : {cmd_id}")

def checkFeedBack(result):
    global hasFinishedMoving

    feedback = result.result()
    mobility_feedback = feedback.feedback.synchronized_feedback.mobility_command_feedback
    if mobility_feedback.status != RobotCommandFeedbackStatus.STATUS_PROCESSING:
        print("Failed to reach the goal")
    traj_feedback = mobility_feedback.se2_trajectory_feedback
    if (traj_feedback.status == traj_feedback.STATUS_AT_GOAL and
            traj_feedback.body_movement_status == traj_feedback.BODY_STATUS_SETTLED):
        print("Arrived at the goal.")

        run_webrtc.moving = False
        hasFinishedMoving = True



async def goTo(robot,dx: float = 0,dy: float = 0, dyaw: float = 0,frame=ODOM_FRAME_NAME,stairs:bool=False):
    global cmd_id
    global hasFinishedMoving
    # Setup clients for the robot state and robot command services.

    print(f"JE VAIS AU DEGREE : {dyaw}")

    cmd_id = None
    
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    robot_command_client = robot.ensure_client(RobotCommandClient.default_service_name)

    transforms = robot_state_client.get_robot_state().kinematic_state.transforms_snapshot

    # Build the transform for where we want the robot to be relative to where the body currently is.
    body_tform_goal = math_helpers.SE2Pose(x=dx, y=dy, angle=dyaw)
    # We do not want to command this goal in body frame because the body will move, thus shifting
    # our goal. Instead, we transform this offset to get the goal position in the output frame
    # (which will be either odom or vision).
    out_tform_body = get_se2_a_tform_b(transforms, frame, BODY_FRAME_NAME)
    out_tform_goal = out_tform_body * body_tform_goal

    # Command the robot to go to the goal point in the specified frame. The command will stop at the
    # new position.
    robot_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
        goal_x=out_tform_goal.x, goal_y=out_tform_goal.y, goal_heading=out_tform_goal.angle,
        frame_name=frame, params=RobotCommandBuilder.mobility_params(stair_hint=stairs))
    end_time = 10.0
    cmdFuture = robot_command_client.robot_command_async(lease=None, command=robot_cmd,
                                                end_time_secs=time.time() + end_time)

    cmdFuture.add_done_callback(checkDoneMoving)

    while not cmd_id:
        await asyncio.sleep(0.1)

    """
    hasFinishedMoving = False

    while not hasFinishedMoving:
        if cmd_id :
            feedbackFuture = robot_command_client.robot_command_feedback_async(cmd_id)
            feedbackFuture.add_done_callback(checkFeedBack)
    """