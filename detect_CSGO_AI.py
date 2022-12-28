# YOLOv3 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --weights yolov3.pt --source 0  # webcam
                                                             img.jpg  # image
                                                             vid.mp4  # video
                                                             path/  # directory
                                                             path/*.jpg  # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
"""

import argparse
import os
import sys
from pathlib import Path
import time

import cv2
import torch
import torch.backends.cudnn as cudnn

from PIL import ImageGrab
import  numpy as np

import pyautogui
import direct_input
import psutil
import win32con
import win32gui
import win32api
import win32process
import pynput

from utils.augmentations import letterbox

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


@torch.no_grad()
def run(weights=ROOT / 'yolov3.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        
        ):

    print('正在加载模型...')
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    side = input('选择识别阵营: CT-0 T-1 全部-2 ')

    print('模型加载成功')

    while 1:
        process_name,hwnd = get_process_name()

        if process_name == 'csgo.exe': # 判断是否是csgo窗口
            
            # 通过句柄值获取当前窗口的【左、上、右、下】四个方向的坐标位置

            left, top, right, bottom = win32gui.GetWindowRect(hwnd)
            break

    cv2.namedWindow("cs")
    screen_w,screen_h = ImageGrab.grab().size # 1920,1080
    csgo_w,csgo_h = int(right-left),int(bottom-top) # 1280 720
    # print(csgo_w,csgo_h)
    # cv2.resizeWindow("cs", int(screen_w),int(screen_h)) # 216 = 1080/5
    cv2.resizeWindow("cs", csgo_w,csgo_h)
    cv2.waitKey(1)

    while 1:
        
        process_name,hwnd = get_process_name()

        if process_name == 'csgo.exe': # 判断是否是csgo窗口 
            
            # 432 = 1080/5*2 648 = 1080/5*3
            frame_cv = ImageGrab.grab((left, top+int(csgo_h/5*2), right, top+int(csgo_h/5*3))) # (左x,上y,右x,下y)
            im0s = cv2.cvtColor(np.array(frame_cv), cv2.COLOR_RGB2BGR)
            im = letterbox(im0s, imgsz, stride=stride, auto=pt and not jit)[0]
            # Convert
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)

            # 显示图片
            # 创建一个窗口，名称cs
            cv2.imshow("cs",im0s)
            cv2.waitKey(1)

            # Run inference
            if pt and device.type != 'cpu':
                model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
            dt, seen = [0.0, 0.0, 0.0], 0

            t1 = time_sync()
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            # pred 形状 (n)(det)
            pred = model(im, augment=augment, visualize=visualize)

            t3 = time_sync()
            dt[1] += t3 - t2

            # NMS 
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            dt[2] += time_sync() - t3

            det = pred[0] # 只有一个批次

            # 判断CT,T逻辑
            if side == '0': # CT
                det = det[det[:,-1]==0,:]
                pass
            elif side == '1': # T
                det = det[det[:,-1]==1,:]
                pass

            

            if det.shape[0] != 0:
                # Process predictions
                seen += 1
                
                im0 =  im0s.copy()

                distance_list = []
                xy_list = []

                # det 形状: (左x,上y,右x,下y,置信度,类别(CT:0,T:1))
                
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                
                MouseX, MouseY = pyautogui.position()
            
                for j in range(det.shape[0]):

                    x = det[j][0]+left + det[j][2]+left
                    x = x / 2 - MouseX
                    x = int(x)
                    y = det[j][1]+top+csgo_h/5*2 + det[j][3]+top+csgo_h/5*2
                    y = y / 2 - MouseY
                    y = int(y)
                    distance = x**2+y**2
                    distance_list.append(distance)
                    xy_list.append([x,y])

                    # rectangle 左上角（x，y），右下角（x，y）
                    rectangle = cv2.rectangle(im0s, (int(det[j,0]),int(det[j,1])), (int(det[j,2]),int(det[j,3])),(0,255,0), 1)
                    cv2.imshow('cs', rectangle)
                    cv2.waitKey(1)  # 1 millisecond
                
                # print(det[:,-2])
                # print('distance',distance_list)
                # print('xy_list',xy_list)
                
                index = distance_list.index(min(distance_list))
                # print('index',index)
                x,y = xy_list[index]
                rectangle = cv2.rectangle(im0s, (int(det[index,0]),int(det[index,1])), (int(det[index,2]),int(det[index,3])),(0,0,255), 1)
                cv2.imshow('cs', rectangle)
                cv2.waitKey(1)  # 1 millisecond
                

                process_name,hwnd = get_process_name()

                if process_name == 'csgo.exe': # 判断是否是csgo窗口
                    while 1:
                        with pynput.mouse.Events() as event:

                            for i in event:
                                if isinstance(i, pynput.mouse.Events.Click):

                                    #鼠标点击事件。
                                    if i.button is pynput.mouse.Button.right :

                                        xy_rate_list = [[0.25,1.0],[0.45,1.0],[0.55,1.0]]

                                        if -int(csgo_w/2)<x<-int(csgo_w/10*3) or int(csgo_w/10*3)<x<int(csgo_w/2):
                                            x_rate,y_rate = xy_rate_list[0]
                                        elif -int(csgo_w/2)<x<-int(csgo_w/6) or int(csgo_w/6)<x<int(csgo_w/2):
                                            x_rate,y_rate = xy_rate_list[1]
                                        else:
                                            x_rate,y_rate = xy_rate_list[2]
                                            

                                        # 移动参数
                                        

                                        moved_x = x / x_rate
                                        moved_y = y / y_rate

                                        if moved_x<-int(csgo_w/2) :
                                            moved_x = -int(csgo_w/2)+1
                                        elif moved_x>int(csgo_w/2):
                                            moved_x = int(csgo_w/2)-1

                                        if moved_y<-int(csgo_h/2):
                                            moved_y = -int(csgo_h/2)+1
                                        elif moved_y>int(csgo_h/2):
                                            moved_y = int(csgo_h/2)-1

                                        direct_input.set_pos(int(moved_x), int(moved_y))
                                        # direct_input.shoot(2,0.15)
                                        break
                                else:
                                    break
                            break

                # time.sleep(2)

def get_process_name():
    # 获取当前鼠标【x y】坐标
    point = win32api.GetCursorPos()  
    # 通过坐标获取坐标下的【窗口句柄】
    hwnd = win32gui.WindowFromPoint(point)  # 请填写 x 和 y 坐标
    # 通过句柄获取【线程ID 进程ID】
    hread_id, process_id = win32process.GetWindowThreadProcessId(hwnd) 
    # 通过进程ID获取【进程名称】 列：weixin.exe
    process_name = psutil.Process(process_id).name()
    return process_name,hwnd

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'best.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images/CSGO_AI', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    
    
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
