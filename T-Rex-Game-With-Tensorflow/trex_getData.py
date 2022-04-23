import keyboard
import uuid
import time
from PIL import Image
from mss import mss

"""
http://www.trex-game.skipser.com/
"""

mon = {"top":365, "left":745, "width":250, "height":100}
sct = mss()

i = 0

def record_screen(record_id, key):
    global i
    
    i += 1
    print("{}: {}".format(key, i))
    img = sct.grab(mon)
    im = Image.frombytes("RGB", img.size, img.rgb)
    im.save("./img1/{}_{}_{}.png".format(key, record_id, i))
    
is_exit = False

def exit():
    global is_exit
    is_exit = True
    
keyboard.add_hotkey("esc", exit)

record_id = uuid.uuid4()

while True:
    
    if is_exit: break

    try:
        if keyboard.is_pressed(keyboard.KEY_UP):
            record_screen(record_id, "up")
            time.sleep(0.2)
        elif keyboard.is_pressed(keyboard.KEY_DOWN):
            record_screen(record_id, "down")
            time.sleep(0.2)
        elif keyboard.is_pressed("right"):
            record_screen(record_id, "right")
            time.sleep(0.2)
    except RuntimeError: continue
            






















