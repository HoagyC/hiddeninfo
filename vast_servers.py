"""
Utility functions for creating, destroying and inspecting VAST.ai servers
"""

import subprocess
import json 
import time 

def get_offers():
    vast_cmd = "./vast search offers --raw 'reliability > 0.99  num_gpus=1 rentable=True inet_up>200 dph<0.4' -o 'dlperf-'"
    raw_out = subprocess.Popen(vast_cmd, stdout=subprocess.PIPE, shell=True, stderr=subprocess.PIPE)
    out, err = raw_out.communicate()
    return json.loads(out)

def create_server(id):
    create_cmd = f"./vast create instance {id} --raw --image pytorch/pytorch --disk 100"
    raw_out = subprocess.Popen(create_cmd, stdout=subprocess.PIPE, shell=True, stderr=subprocess.PIPE)
    out, err = raw_out.communicate()
    return json.loads(out)

def get_servers_info():
    info_cmd = f"./vast show instances --raw"
    raw_out = subprocess.Popen(info_cmd, stdout=subprocess.PIPE, shell=True, stderr=subprocess.PIPE)
    out, err = raw_out.communicate()
    
    if 'error 429' in str(out):
        print("429 error, sleeping for 10 seconds")
        time.sleep(10)
        return get_servers_info()
    return json.loads(out)

def destroy_server(id):
    destroy_cmd = f"./vast destroy instance {id} --raw"
    raw_out = subprocess.Popen(destroy_cmd, stdout=subprocess.PIPE, shell=True, stderr=subprocess.PIPE)
    out, err = raw_out.communicate()
    return json.loads(out)

    
def get_vast_ai_servers(n_servers=1, clear_existing=False):
    time.sleep(1) # Give time, else get 429 error
    if clear_existing:
        servers = get_servers_info()
        for server in servers:
            delete_cmd = f"./vast destroy instance {server['id']}"
            subprocess.run(delete_cmd, shell=True)
    
    offers = get_offers()

    if len(offers) < n_servers:
        raise ValueError(f"Only {len(offers)} offers available, but {n_servers} servers requested")

    current_servers = []
    offer_n = 0
    while len(current_servers) < n_servers:
        server_info = create_server(offers[offer_n]["id"])
        current_servers.append(server_info["new_contract"])
        print(f"Created server {server_info['new_contract']}")
        offer_n += 1
    
    print([server_info['cur_state'] for server_info in get_servers_info()])
    working_servers = [server_info['cur_state'] in ["running", "unloaded"] for server_info in get_servers_info()]
    assert all(working_servers), f"Some servers failed to start up, {working_servers}"
    while not all([server_info['cur_state'] in ["running"] for server_info in get_servers_info()]):
        print([server_info['cur_state'] for server_info in get_servers_info()])
        time.sleep(5) # Wait for the servers to start up
        
    print("All servers 'running', waiting 45 seconds for them to actually be ready")
    time.sleep(60) # Give extra time for loading, doesn't seem to be ready even when 'running'
    return [(s['ssh_port'], int(s['ssh_idx'])) for s in get_servers_info()]


if __name__ == "__main__":
    print(get_vast_ai_servers(1, clear_existing=False))