import subprocess as sp


class Live():
    def __init__(self, fps, width, height):
        command = ['ffmpeg',
                   '-y',
                   '-f', 'rawvideo',
                   '-vcodec', 'rawvideo',
                   '-pix_fmt', 'bgr24',
                   '-s', "{}x{}".format(width, height),
                   '-r', str(fps),
                   '-i', '-',
                   '-c:v', 'libx264',
                   '-pix_fmt', 'yuv420p',
                   '-preset', 'ultrafast',
                   '-f', 'flv',
                   'rtmp://192.168.101.66:1935/live/test']
        self.proc = sp.Popen(command, stdin=sp.PIPE, shell=False)

    def send(self, img):
        self.proc.stdin.write(img.tostring())

    def close(self):
        self.proc.stdin.close()  # 关闭输入管道
        self.proc.communicate()