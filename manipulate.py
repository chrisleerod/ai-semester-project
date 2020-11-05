import time
global letter1, framecount, held
letter1 = None
framecount = 0
held = False

def letter1store(letter):
    letter1 = letter
    time.sleep(1)
def letter1check():
    return letter1
def letter2check(letter):
    global letter1, framecount, held
    if letter == letter1:
        framecount += 1
    else:
        framecount = 0
    if framecount >= 24:
        held = True
        framecount = 0
def isheld():
    global held
    if held:
        return True
        held = False
    else:
        return False