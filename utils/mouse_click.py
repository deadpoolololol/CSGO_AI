import pynput

while True:
    with pynput.mouse.Events() as event:

        for i in event:
        #迭代用法。

            if isinstance(i, pynput.mouse.Events.Click):
                #鼠标点击事件。
                if i.button is pynput.mouse.Button.middle and i.pressed:
                
                    pass
            break
        
