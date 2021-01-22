from redis import Redis
from redis.sentinel import Sentinel

from cache import config

if config.db_master_name is None:
    __redis = Redis(host=config.host, port=config.port, decode_responses=True)
else:
    db_sentinel = Sentinel(sentinels=config.db_urls, socket_timeout=0.1)
    __redis = db_sentinel.master_for(config.db_master_name, socket_timeout=0.1,
                                     password=config.db_password, decode_responses=True)


class ImageInfo(object):
    def __init__(self, id, box, label, ):
        pass


def save_info(id):
    pass