import argparse
import logging
import uuid
from typing import List

import numpy as np
import cv2
import requests
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from kalman.manage import delete as kalman_delete, register as kalman_register
from utils import webapi

app = FastAPI()

logger = logging.getLogger("uvicorn.error")

class Webcam(BaseModel):
    sid: str


class Task(BaseModel):
    # 数据id
    data_id: str
    # 数据类型
    type: str
    # 数据url
    url: str
    # 数据其他参数，摄像头id等
    param: Webcam = Webcam(sid=1)


class Tasks(BaseModel):
    # 任务id
    task_id: str
    # 回调地址
    callback: str = None
    # 数据
    data: List[Task]


# tracker = Tracker(100, 100, 50, 10)
# first = 0

@app.post("/detect")
def obj_detect(tasks: Tasks):
    logger.info(tasks.data)
    result = []
    for datum in tasks.data:
        request = requests.get(datum.url)
        img = cv2.imdecode(np.frombuffer(request.content, np.uint8), cv2.IMREAD_COLOR)
        sid = datum.param.sid or str(uuid.uuid4())
        objects, scores = webapi.detect_info(img, logger)
        result.append({
            "data_id": datum.data_id,
            "url": datum.url,
            "content": {
                "objects": objects,
                "score": scores
            }
        })
    return result

@app.delete("/tracker/{sid}")
def tracker_delete(sid: str):
    return kalman_delete(sid)

@app.get("/tracker/{sid}")
def tracker_register(sid: str):
    return kalman_register(sid)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="fastapi tracker")

    parser.add_argument("--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="Port for HTTP server (default: 8080)")

    parser.add_argument('--weights', nargs='+', type=str, default='yolov5/weights/yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')

    args = parser.parse_args()
    webapi.init(opt_in=args)

    uvicorn.run(app='fastapp:app', host=args.host, port=args.port)
