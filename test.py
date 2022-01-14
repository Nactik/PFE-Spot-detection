from stand import stand
from goTo import goTo

import bosdyn.client
import bosdyn.client.util


def main():
    #try:
    #    stand();
    #    return True
    #except Exception as exc:  # pylint: disable=broad-except
    #    logger = bosdyn.client.util.get_logger()
    #    logger.error("Threw an exception: %s\n%s", exc, traceback.format_exc())
    #    return False

    goTo(dyaw=180);

main()