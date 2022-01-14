import os 
import bosdyn.client
from dotenv import load_dotenv


def connect():
    load_dotenv()

    sdk = bosdyn.client.create_standard_sdk('RobotCommandMaster')
    robot = sdk.create_robot(os.getenv("ROBOT_IP"))
    robot.authenticate(os.getenv('ROBOT_USERNAME'), os.getenv('ROBOT_PASSWORD'))

    return robot