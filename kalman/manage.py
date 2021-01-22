from typing import Dict

from kalman.tracker import Tracker

kmap: Dict[str, Tracker] = {}

def register(sid):
    kmap[sid] = Tracker(100, 100, 50, 10)

def delete(sid):
    del kmap[sid]