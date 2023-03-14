import subprocess
import sys
import os
import time 
from itertools import product

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vast_servers import get_servers_info, reboot_server

VAST_AI_SERVERS = [(16612, 5), (14696, 4)]


def setup_multiple(n_start: int = 0):
    server_list = server_list = [(s['ssh_port'], int(s['ssh_idx'])) for s in get_servers_info()]

    processes = []
    files = []
    for ndx, server in enumerate(server_list[n_start:]):
        port, num = server
        addr = "root@ssh" + str(num) + ".vast.ai"

        # Install python3-venv and build-essential
        pre_install_cmd = f"ssh -oStrictHostKeyChecking=no -p {port} {addr} \"apt update && apt install -y python3-venv build-essential\""
        # Create an async process to run the command which we will then wait for later
        print(f"Installing python3-venv and build-essential on server {server}")
        subprocess.run(pre_install_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # File descriptor for the output of the setup command
        if not os.path.exists("multilog"):
            os.mkdir("multilog")
        
        setup_out_f = open(f"multilog/setup_{server[0]},{server[1]}.out", "w")
        setup_err_f = open(f"multilog/setup_{server[0]},{server[1]}.err", "w")
        files.extend([setup_out_f, setup_err_f])

        # Setup the server for running experiments
        print(f"Setting up server {server}")
        setup_cmd = f"make ssh-setup SSH_DESTINATION=root@ssh{num}.vast.ai SSH_PORT=" + str(port)
        # Running with output piped to /dev/null to avoid the output being printed to the terminal
        processes.append(subprocess.Popen(setup_cmd, shell=True, stdout=setup_out_f, stderr=setup_err_f)) 

        # Close the file descriptors
        setup_out_f.close()
        setup_err_f.close()
    
    # Wait for all the servers to be setup
    running = [p.poll() is None for p in processes]
    n_minutes = 0
    while any(running):
        print(f"Setting up for {n_minutes} minute/s. Current status: {['Running' if r else 'Done' for r in running]}")
        time.sleep(60)
        running = [p.poll() is None for p in processes]
        n_minutes += 1

    
    [p.wait() for p in processes]
    print("All servers setup")
    for f in files:
        f.close()
        
def kill_existing(n_start: int = 0) -> None:
    start_ndx = 0 # The index of the first experiment to run in the configs list at the end of train_CUB.py
    server_info = get_servers_info()
    server_ids = [(s['id']) for s in server_info]

    for id in server_ids[n_start:]:
        reboot_server(id)

def run_multiple(n_start: int = 0) -> None:
    start_ndx = n_start # The index of the first experiment to run in the configs list at the end of train_CUB.py
    server_info = get_servers_info()
    server_list = server_list = [(s['ssh_port'], int(s['ssh_idx'])) for s in server_info]
    
    processes = []
    files = []
    for ndx, server in enumerate(server_list[n_start:]):
        port, num = server
        addr = "root@ssh" + str(num) + ".vast.ai"
        
        # Sync the codebase
        print(f"Syncing server {server}")
        sync_cmd = f"make ssh-sync SSH_DESTINATION=root@ssh{num}.vast.ai SSH_PORT=" + str(port)
        sync_proc = subprocess.run(sync_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


        # File descriptor for the output of the experiment command
        if not os.path.exists("multilog"):
            os.mkdir("multilog")
        exp_out_f = open(f"multilog/exp_{port},{num}.out", "w")
        exp_err_f = open(f"multilog/exp_{port},{num}.err", "w")
        files += [exp_out_f, exp_err_f]

        # Run the experiment with the given parameters
        print(f"Running experiment {start_ndx + ndx} on server {server}")
        run_cmd = f"ssh -fn -p {port} {addr} \" cd ~/hoagy-hiddeninfo-sync && source .env/bin/activate && python CUB/train_CUB.py --cfg-index={start_ndx + ndx}\"&"
        run_proc = subprocess.Popen(run_cmd, shell=True, stdout=exp_out_f, stderr=exp_err_f)
        processes.append(((server_info[ndx]["id"], port, num), run_proc))



    # Wait for all the servers to be setup
    running = [p.poll() is None for _, p in processes]
    n_hours = 0.
    while any(running):
        print(running)
        print(f"Running up for {n_hours} hours. Current status: {['Running' if r else 'Done' for r in running]}")
        time.sleep(3)
        running = [p.poll() is None for _, p in processes]
        n_hours += 0.5
        for id, p in processes:
            if p.poll() is not None:
                print(f"Server {'_'.join([str(x) for x in id])} has finished running")
                # destroy_server(id[0])
                # print(f"Server {'_'.join([str(x) for x in id])} has been destroyed")

    for f in files:
        f.close()


def sync_multiple(n_start=0):
    # Run ssh-sync with each of the servers
    server_info = get_servers_info()
    server_list = [(s['ssh_port'], int(s['ssh_idx'])) for s in server_info]
    for server in server_list[n_start:]:
        port, num = server
        sync_cmd = f"make ssh-sync SSH_DESTINATION=root@ssh{num}.vast.ai SSH_PORT=" + str(port)
        print(f"Syncing server {server}")
        subprocess.run(sync_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        n_start = int(sys.argv[2])
    else:
        n_start = 0
    if len(sys.argv) > 1 and sys.argv[1] == "sync":
        sync_multiple(n_start)
    elif len(sys.argv) > 1 and sys.argv[1] == "setup":
        setup_multiple(n_start)
    elif len(sys.argv) > 1 and sys.argv[1] == "kill":
        kill_existing()
    elif len(sys.argv) > 1 and sys.argv[1] == "run":
        run_multiple(n_start)
