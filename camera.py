import cv2
import numpy as np
import onnxruntime as ort
import PySimpleGUI as sg

def main():
    # constants
    index_to_letter = list('ABCDEFGHIKLMNOPQRSTUVWXY')
    mean = 0.485 * 255.
    std = 0.229 * 255.

    # create runnable session with exported model
    ort_session = ort.InferenceSession("signlanguage.onnx")

    sg.theme("LightGreen")

    # Define the window layout
    layout = [
        [sg.Text("ASL Detection", size=(60, 1), justification="center")],
        [sg.Image(filename="", key="-IMAGE-")],
        [sg.Button("Exit", size=(10, 1))],
    ]

    # Create the window and show it without the plot
    window = sg.Window("ASL Translator", layout, location=(800, 400))

    cap = cv2.VideoCapture(0)
    while True:
        event, values = window.read(timeout=20)
        if event == "Exit" or event == sg.WIN_CLOSED:
            break

        # Capture frame-by-frame
        ret, frame = cap.read()

        # preprocess data

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        x = cv2.resize(frame, (28, 28))
        x = (x - mean) / std

        x = x.reshape(1, 1, 28, 28).astype(np.float32)
        y = ort_session.run(None, {'input': x})[0]

        index = np.argmax(y, axis=1)
        letter = index_to_letter[int(index)]

        cv2.putText(frame, letter, (50, 75), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), thickness=8)

        imgbytes = cv2.imencode(".png", frame)[1].tobytes()
        window["-IMAGE-"].update(data=imgbytes)

    window.close()

main()
