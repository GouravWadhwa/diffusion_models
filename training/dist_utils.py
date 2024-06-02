"""

Helpers for distributed training

"""

import os
import io
import socket

from mpi4py import MPI
import blobfile as bf

import torch
import torch.distributed as dist

GPUS_PER_NODE = 8

def setup_dist():
    if dist.is_initialized():
        return
    
    comm = MPI.COMM_WORLD
    backend = "gloo" if not torch.cuda.is_available() else "nccl"

    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostbyname(socket.getfqdn())

    os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
    os.environ["RANK"] = str(comm.rank)
    os.environ["WORLD_SIZE"] = str(comm.size)

    port = comm.bcast(_find_free_port(), root=0)
    os.environ["MASTER_PORT"] = str(port)

    dist.init_process_group(backend=backend, init_method="env://")

def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    except Exception as e:
        s.close()
        raise RuntimeError(f"Not able to setup distributed training: {e}")

def dev():
    if torch.cuda.is_available():
        return torch.device(f"cuda:{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}")
    return torch.device("cpu")

def load_state_dict(path, **kwargs):
    if MPI.COMM_WORLD.Get_ranl() == 0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
    else:
        data = None

    data = MPI.COMM_WORLD.bcast(data)
    return torch.load(io.BytesIO(data), **kwargs)

def sync_params(params):
    for p in params:
        with torch.no_grad():
            dist.broadcast(p, 0)