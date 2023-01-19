import subprocess
import sys
from itertools import product

VAST_AI_SERVERS = [(19952, 6)]

ATTN_COEFS = [0.1, 1, 10]
ATTR_SPARSITIES = [1, 3, 10]
run_params = list(product(ATTN_COEFS, ATTR_SPARSITIES))

def run_multiple(setup: bool) -> None:
    for ndx, server in enumerate(VAST_AI_SERVERS):
        if ndx >= len(run_params):
            break
        port, num = server
        addr = "root@ssh" + str(num) + ".vast.ai"

        if setup:
            # Install python3-venv and build-essential
            pre_install_cmd = f"ssh -p {port} {addr} \"apt update && apt install -y python3-venv build-essential\""
            subprocess.run(pre_install_cmd, shell=True)

            # Setup the server for running experiments
            setup_cmd = f"make ssh-setup SSH_DESTINATION=root@ssh{num}.vast.ai SSH_PORT=" + str(port)
            subprocess.run(setup_cmd, shell=True)
        
        # Sync the codebase
        sync_cmd = f"make ssh-sync SSH_DESTINATION=root@ssh{num}.vast.ai SSH_PORT=" + str(port)
        subprocess.run(sync_cmd, shell=True)

        # Run the experiment with the given parameters
        # attr_coef, sparsity = run_params[ndx]
        run_cmd = f"ssh -fn -p {port} {addr} \" cd ~/hoagy-hiddeninfo-sync && source .env/bin/activate && python CUB/train_CUB.py --cfg-index={ndx}\"&"
        subprocess.Popen(run_cmd, shell=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        setup = True
    else:
        setup = False
    run_multiple(setup=setup)