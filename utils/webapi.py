import logging

import cv2
import torch
import numpy as np
from numpy import random
import random

import settings
from kalman.tracker import Tracker
from kalman.kalman_object import Kobj
from kalman.line_segments_intersect import Point, doIntersect, angle
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import letterbox
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5.utils.plots import plot_one_box_chinese
from yolov5.utils.torch_utils import select_device, time_synchronized

def detect_info(img0, logger = logging.getLogger("webapi")):
    # Run inference
    # img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    # _ = model(img) if device.type != 'cpu' else None  # run once

    # Padded resize
    img = letterbox(img0, new_shape=imgsz)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = model(img, augment=opt.augment)[0]

    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
    t2 = time_synchronized()

    # Process detections
    det = pred[0]  # detections per image
    s = ""
    s += '%gx%g ' % img.shape[2:]  # print string
    objects = []
    scores = []
    if len(det):

        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
        # Print results
        for c in det[:, -1].unique():
            n = (det[:, -1] == c).sum()  # detections per class
            s += '%g %ss, ' % (n, names[int(c)])  # add to string

        # Write results
        for *xyxy, conf, cls in reversed(det):
            cX = ((int(xyxy[0]) + int(xyxy[2])) / 2.0)
            cY = ((int(xyxy[1]) + int(xyxy[3])) / 2.0)
            center = (int(cX), int(cY))

            objects.append({
                "box": (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])),
                "center": center,
                "title": names[int(cls)]
            })
            scores.append('%.2f' % conf)

    logger.info('%sDone. (%.3fs)' % (s, t2 - t1))
    return objects, scores

def detect(img0, first=0, ct = Tracker(25, 60, 1000, 10), fence = None):
    fence = fence or ((0, 240), (500, 240))

    p1 = Point(fence[0][0], fence[0][1])
    q1 = Point(fence[1][0], fence[1][1])
    # Run inference
    # img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    # _ = model(img) if device.type != 'cpu' else None  # run once

    # Padded resize
    img = letterbox(img0, new_shape=imgsz)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = model(img, augment=opt.augment)[0]

    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
    t2 = time_synchronized()

    # Process detections
    det = pred[0]  # detections per image
    s = ""
    s += '%gx%g ' % img.shape[2:]  # print string
    if len(det):

        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
        # Print results
        for c in det[:, -1].unique():
            n = (det[:, -1] == c).sum()  # detections per class
            s += '%g %ss, ' % (n, names[int(c)])  # add to string
        
        centers = []
        # Write results
        for *xyxy, conf, cls in reversed(det):
            if names[int(cls)] not in settings.tags_list:
                continue

            label = '%s %.2f' % (names[int(cls)], conf)
            
            cX = ((int(xyxy[0]) + int(xyxy[2])) / 2.0)
            cY = ((int(xyxy[1]) + int(xyxy[3])) / 2.0)

            kobj = Kobj(xyxy, (cX, cY), colors[int(cls)], label)

            centers.append(kobj)
        if len(centers) > 0:
            first=1
            # Track object using Kalman Filter
            ct.Update(centers,first)

            # For identified object tracks draw tracking line
            # Use various colors to indicate different track_id
            #print('NUM OF OBJECTS : ',len(ct.tracks))
            for i in range(len(ct.tracks)):
                clr = None
                if (len(ct.tracks[i].trace) > 1 and ct.tracks[i].skipped_frames == 0):
                    #print('NUM OF OBJECTS : ',tracker.tracks[i].trace)
                    color = ct.tracks[i].kobj.color
                    label = ct.tracks[i].kobj.label
                    xyxy = ct.tracks[i].kobj.xyxy
                    for j in range(len(ct.tracks[i].trace)-1):
                        # Draw trace line
                        x1 = ct.tracks[i].trace[j][0][0]
                        y1 = ct.tracks[i].trace[j][1][0]
                        x2 = ct.tracks[i].trace[j+1][0][0]
                        y2 = ct.tracks[i].trace[j+1][1][0]

                        cv2.line(img0, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                        p2 = Point(int(x1), int(y1))
                        q2 = Point(int(x2), int(y2))
                        if doIntersect(p1, q1, p2, q2) and (angle(p1, q1, p2, q2) < 180):
                            ct.tracks[i].inorout = 1
                        if doIntersect(p1, q1, p2, q2) and (angle(p1, q1, p2, q2) > 180):
                            ct.tracks[i].inorout = 0
                        # if intersect:
                        #     d = angle(p1, q1, p2, q2)
                        #     if d < 180:
                        #         intersect = False
                        
                    if ct.tracks[i].inorout == 1:
                        # 标红
                        cv2.line(img0, fence[0], fence[1], (0, 0, 255), 2)
                        img0 = plot_one_box_chinese(xyxy, img0, label=label + ";危险", color=(0, 0, 255), line_thickness=3)
                    else:
                        cv2.line(img0, fence[0], fence[1], (0, 0, 255), 2)
                        img0 = plot_one_box_chinese(xyxy, img0, label=label, color=color, line_thickness=3)
                    
            # Display the resulting tracking frame
            #cv2.imshow('Tracking', frame)
        elif first==1:
            ct.Update(centers,0)
            #print('NUM OF OBJECTSno : ',len(ct.tracks))
            for i in range(len(ct.tracks)):
                if (len(ct.tracks[i].trace) > 1):
                    #print('NUM OF OBJECTSnononono : ',len(ct.tracks[i].trace),)
                    #print('trace : ',ct.tracks[i].trace[len(ct.tracks[i].trace)-1],)
                    color = ct.tracks[i].kobj.color
                    for j in range(len(ct.tracks[i].trace)-1):
                        # Draw trace line
                        x1 = ct.tracks[i].trace[j][0][0]
                        y1 = ct.tracks[i].trace[j][1][0]
                        x2 = ct.tracks[i].trace[j+1][0][0]
                        y2 = ct.tracks[i].trace[j+1][1][0]

                        cv2.line(img0, (int(x1), int(y1)), (int(x2), int(y2)),
                                    color, 2)
    # Print time (inference + NMS)
    print('%sDone. (%.3fs)' % (s, t2 - t1))
    return first, img0

device = None
model = None
imgsz = None
names = []
colors = []
opt = None

def init(opt_in):

    global opt
    opt = opt_in
    global device
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    global model
    # Load model
    model = attempt_load(opt.weights, map_location=device)  # load FP32 model
    global imgsz
    imgsz = check_img_size(opt.img_size, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16
    
    # Get names and colors
    global names
    names = model.module.names if hasattr(model, 'module') else model.names
    names = settings.tags(names)
    print(names)
    global colors
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
