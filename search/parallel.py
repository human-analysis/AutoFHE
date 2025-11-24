import os
import errno
import queue
import time
import numpy as np
import multiprocessing as mp
from multiprocessing import Queue, Process
from evolution import rccde


class CPUParalCompute(object):
    def __init__(self, num_cpus: int, degrees: dict, prefix=None):
        super(CPUParalCompute, self).__init__()
        self.num_degrees = len(degrees) * 10
        num_cpus = min(num_cpus, self.num_degrees)
        self.task_queue = Queue()
        manager = mp.Manager()
        self.return_dict = manager.dict()
        self.deg_keys = []
        for k, deg in degrees.items():
            for i in range(10):
                self.task_queue.put((k, i, deg))
            self.deg_keys.append(k)
        self.size = self.task_queue.qsize()
        self.cpu_workers = [CoeffSolver(idx, self.task_queue, self.return_dict, prefix) for idx in range(num_cpus)]

    def __call__(self):
        for w in self.cpu_workers:
            w.start()
        for w in self.cpu_workers:
            w.join()
            if w.is_alive():
                w.terminate()
                w.close()

    def __len__(self):
        return self.size

    def get(self):
        res_dict = {}
        for deg_k in self.deg_keys:
            best_err = float("inf")
            com_poly = None
            for i in range(10):
                if deg_k+str(i) in self.return_dict.keys():
                    res = self.return_dict[deg_k+str(i)]
                    poly, err = res
                    if err < best_err:
                        best_err = err
                        com_poly = poly
                if com_poly is not None:
                    res_dict[deg_k] = com_poly
        return res_dict


class CoeffSolver(Process):
    def __init__(self, idx: int, task_queue: Queue, return_dict, prefix: str = None) -> None:
        super(CoeffSolver, self).__init__()
        self.task_queue = task_queue
        self.return_dict = return_dict
        self.name = 'CPU-Worker-{:0>3d}'.format(idx)
        if prefix is not None:
            self.name = '{:}| {:}'.format(prefix, self.name)

    def run(self) -> None:
        name = self.name
        pid = os.getpid()
        while True:
            try:
                deg_key, idx, deg = self.task_queue.get(False)
            except queue.Empty:
                break
            try:
                np.random.seed(int((time.time() // 10**5 + idx * 1000)))
                com_poly, err = rccde(deg)
                self.return_dict[deg_key+str(idx)] = (com_poly, err)
                print('{:}| PID: {:d}| Remaining: {:d}| L1-Error: {:.2e}'.format(name, pid, self.task_queue.qsize(), err))
            except IOError as e:
                if e.errno == errno.EPIPE:
                    print('{:}| PID: {:d}| PIPE BROKEN!'.format(name, pid))
                else:
                    print('{:}| PID: {:d}| Unknown multiprocessing error!'.format(name, pid))
            if self.task_queue.qsize() == 0:
                break
        print('{:}| PID: {:d}| exited'.format(name, pid))