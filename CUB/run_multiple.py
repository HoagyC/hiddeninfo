import subprocess
import sys
import os
import time 
from itertools import product

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vast_servers import get_servers_info

VAST_AI_SERVERS = [(17204, 4), (19866, 6)]

ATTN_COEFS = [0.1, 1, 10]
ATTR_SPARSITIES = [1, 3, 10]
run_params = list(product(ATTN_COEFS, ATTR_SPARSITIES))

def setup_multiple(server_list = None):
    if server_list is None:
        server_list = server_list = [(s['ssh_port'], int(s['ssh_idx'])) for s in get_servers_info()]

    processes = []
    for ndx, server in enumerate(server_list):
        port, num = server
        addr = "root@ssh" + str(num) + ".vast.ai"

        # Install python3-venv and build-essential
        pre_install_cmd = f"ssh -oStrictHostKeyChecking=no -p {port} {addr} \"apt update && apt install -y python3-venv build-essential\""
        # Create an async process to run the command which we will then wait for later
        print(f"Installing python3-venv and build-essential on server {server}")
        subprocess.run(pre_install_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Setup the server for running experiments
        print(f"Setting up server {server}")
        setup_cmd = f"make ssh-setup SSH_DESTINATION=root@ssh{num}.vast.ai SSH_PORT=" + str(port)
        # Running with output piped to /dev/null to avoid the output being printed to the terminal
        processes.append(subprocess.Popen(setup_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)) 
    
    # Wait for all the servers to be setup
    running = [p.poll() is None for p in processes]
    n_minutes = 0
    while any(running):
        print(f"Setting up for {n_minutes}. Current status: {['Running' if r else 'Done' for r in running]}")
        time.sleep(60)
        running = [p.poll() is None for p in processes]
        n_minutes += 1
    
    [p.wait() for p in processes]
    print("All servers setup")

def run_multiple(server_list = None) -> None:
    start_ndx = 13
    if server_list is None:
        server_list = VAST_AI_SERVERS
    
    for ndx, server in enumerate(server_list):
        port, num = server
        addr = "root@ssh" + str(num) + ".vast.ai"
        
        # Sync the codebase
        print(f"Syncing server {server}")
        sync_cmd = f"make ssh-sync SSH_DESTINATION=root@ssh{num}.vast.ai SSH_PORT=" + str(port)
        sync_proc = subprocess.run(sync_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Run the experiment with the given parameters
        # attr_coef, sparsity = run_params[ndx]
        print(f"Running experiment {start_ndx + ndx} on server {server}")
        run_cmd = f"ssh -fn -p {port} {addr} \" cd ~/hoagy-hiddeninfo-sync && source .env/bin/activate && python CUB/train_CUB.py --cfg-index={start_ndx + ndx}\"&"
        run_proc = subprocess.Popen(run_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def sync_multiple(server_list = None):
    # Run ssh-sync with each of the servers
    if server_list is None:
        server_list = VAST_AI_SERVERS
    for server in server_list:
        port, num = server
        sync_cmd = f"make ssh-sync SSH_DESTINATION=root@ssh{num}.vast.ai SSH_PORT=" + str(port)
        print(f"Syncing server {server}")
        subprocess.run(sync_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)




if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sync":
        sync_multiple()
    elif len(sys.argv) > 1 and sys.argv[1] == "setup":
        setup_multiple()
    else:
        run_multiple()
