FROM ultralytics/yolov5

RUN apt-get update && apt-get install -y ffmpeg && apt-get clean

ENV PYTHONIOENCODING=utf-8
ENV PYTHONPATH=/server/

COPY object-tacking-fence /server

WORKDIR /server

RUN pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

CMD python yolov5/detect.py --source rtmp://58.200.131.2:1935/livetv/hunantv --weights /server/yolov5/models/yolov5s.pt --device 0
