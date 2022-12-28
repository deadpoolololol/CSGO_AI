
import ctypes
import time
import random



# time.sleep(5)
# # 获取当前鼠标位置
# while True:
#     currentMouseX, currentMouseY = pydirectinput.position()
#     print(f'当前鼠标: {(currentMouseX, currentMouseY)}')
#     xOffset,yOffset = 10,10
#     pydirectinput.moveRel(xOffset=xOffset, yOffset=yOffset,duration=2)
    
#     pydirectinput.click()
#     print(f'移动到: {(currentMouseX-xOffset, currentMouseY-yOffset)}')
#     time.sleep(0.2)


SendInput = ctypes.windll.user32.SendInput


W = 0x11
A = 0x1E
S = 0x1F
D = 0x20

# C struct redefinitions 
PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time",ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                 ("mi", MouseInput),
                 ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

# Actuals Functions

def PressKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
    print(f'按下: {hexKeyCode}') 

def ReleaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
    print(f'松开: {hexKeyCode}')

# def set_pos(x, y):
#     x = 1 + int(x * 65536./1600.)
#     y = 1 + int(y * 65536./1200.)
#     extra = ctypes.c_ulong(0)
#     ii_ = Input_I()
#     ii_.mi = MouseInput(x, y, 0, (0x0001 | 0x8000), 1, ctypes.pointer(extra))
#     command = Input(ctypes.c_ulong(0), ii_)
#     ctypes.windll.user32.SendInput(1, ctypes.pointer(command), ctypes.sizeof(command))
#     print(f'鼠标移动到: {(int(x*1600./65536.),int(y*1200./65536))}')

def set_pos(x_r, y_r):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.mi = MouseInput(x_r, y_r, 0, 0x0001, 1, ctypes.pointer(extra))
    command = Input(ctypes.c_ulong(0), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(command), ctypes.sizeof(command))
    print(f'鼠标移动了: {(x_r,y_r)}')

def left_click():
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.mi = MouseInput(0, 0, 0, 0x0002, 0, ctypes.pointer(extra))
    ipt = Input(ctypes.c_ulong(0), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(ipt), ctypes.sizeof(ipt))

    time.sleep(0.05)

    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.mi = MouseInput(0, 0, 0, 0x0004, 0, ctypes.pointer(extra))
    ipt = Input(ctypes.c_ulong(0), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(ipt), ctypes.sizeof(ipt))
    
    print(f'鼠标左键点击')

def right_click():
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.mi = MouseInput(0, 0, 0, 0x0008, 0, ctypes.pointer(extra))
    ipt = Input(ctypes.c_ulong(0), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(ipt), ctypes.sizeof(ipt))

    time.sleep(0.05)

    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.mi = MouseInput(0, 0, 0, 0x0010, 0, ctypes.pointer(extra))
    ipt = Input(ctypes.c_ulong(0), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(ipt), ctypes.sizeof(ipt))
    
    print(f'鼠标右键点击')

def shoot_scope(n):
    '''
    开镜射击 n 次
    '''
    # 开镜
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.mi = MouseInput(0, 0, 0, 0x0008, 0, ctypes.pointer(extra))
    ipt = Input(ctypes.c_ulong(0), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(ipt), ctypes.sizeof(ipt))
    time.sleep(0.5)
    for i in range(n):
        # 射击
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        ii_.mi = MouseInput(0, 0, 0, 0x0002, 0, ctypes.pointer(extra))
        ipt = Input(ctypes.c_ulong(0), ii_)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(ipt), ctypes.sizeof(ipt))
        time.sleep(0.05)
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        ii_.mi = MouseInput(0, 0, 0, 0x0004, 0, ctypes.pointer(extra))
        ipt = Input(ctypes.c_ulong(0), ii_)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(ipt), ctypes.sizeof(ipt))
        time.sleep(0.25)

    # 关镜
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.mi = MouseInput(0, 0, 0, 0x0010, 0, ctypes.pointer(extra))
    ipt = Input(ctypes.c_ulong(0), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(ipt), ctypes.sizeof(ipt))
    itv = random.uniform(0.5,1)
    time.sleep(itv)
    print(f'射击 {n} 次')

def shoot(n,sec=0.25):
    '''
    射击 n 次,间隔 sec 秒
    '''
    for i in range(n):
        # 射击
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        ii_.mi = MouseInput(0, 0, 0, 0x0002, 0, ctypes.pointer(extra))
        ipt = Input(ctypes.c_ulong(0), ii_)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(ipt), ctypes.sizeof(ipt))
        time.sleep(0.05)
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        ii_.mi = MouseInput(0, 0, 0, 0x0004, 0, ctypes.pointer(extra))
        ipt = Input(ctypes.c_ulong(0), ii_)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(ipt), ctypes.sizeof(ipt))
        time.sleep(sec)

    itv = random.uniform(0.5,1)
    time.sleep(itv)
    print(f'射击 {n} 次')


if __name__ == '__main__':
    time.sleep(3)

    while True:
        # PressKey(W)
        # time.sleep(1)
        # ReleaseKey(W)
        # currentMouseX, currentMouseY = pydirectinput.position()
        # print(f'当前鼠标: {(currentMouseX, currentMouseY)}')
        # # currentMouseX+=10
        # # currentMouseY+=10
        # set_pos(100,100)

        # shoot_scope(2) # 开镜射击
        shoot(2,0.25) # 射击
        itv = random.uniform(0.5,1)
        time.sleep(itv)