# wrapper of distributed date parallel
import os
import torch

class DDPHelper(object):
    @staticmethod
    def setup(rank=0, world_size=1, backend='gloo', init_method='tcp://localhost:3456', mode="torchrun"):
        # dist initialed:
        if mode == "torchrun":
            torch.distributed.init_process_group(backend=backend)
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        else:
            # os.environ["MASTER_ADDR"] -> "localhost"
            # os.environ["MASTER_PORT"] -> "3456"
            torch.distributed.init_process_group(backend=backend, init_method=init_method, rank=rank, world_size=world_size)
            torch.cuda.set_device(rank) # defualt cuda's device-id

    @staticmethod
    def cleanup():
        torch.distributed.destroy_process_group()

    @staticmethod
    def spawn(func, world_size, func_args=(), *args, **kwargs):
        torch.multiprocessing.spawn(func, (world_size, ) + func_args, 
                                    nprocs=world_size, *args, **kwargs)
        
    @staticmethod
    def is_master():
        return torch.distributed.get_rank() == 0
