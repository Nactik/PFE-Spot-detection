import bosdyn.client
from bosdyn.client.lease import LeaseClient


## Fonction pour creer et récuperer la lease du robot
def acquireLease(robot):
    # Create the lease client.
    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    return (lease_client, lease_client.acquire())
