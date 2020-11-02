import time
global letter1
letter1 = None

def letter1store(letter):
    letter1 = letter
    time.sleep(1)
def letter1check():
    return letter1