import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import threading
import screen_brightness_control as sbc
import speech_recognition as sr
import os
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)


screen_width, screen_height = pyautogui.size()


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
volRange = volume.GetVolumeRange()
minVol, maxVol = volRange[:2]


recognizer = sr.Recognizer()

def recognize_voice():
    global last_command
    while True:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            try:
                audio = recognizer.listen(source, timeout=5)
                command = recognizer.recognize_google(audio).lower()
                last_command = command
            except (sr.UnknownValueError, sr.RequestError, sr.WaitTimeoutError):
                pass
            time.sleep(1)

last_command = ""
voice_thread = threading.Thread(target=recognize_voice, daemon=True)
voice_thread.start()


cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            lm_list = [[id, int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])] for id, lm in enumerate(hand_landmarks.landmark)]

            if lm_list:
                x_thumb, y_thumb = lm_list[4][1], lm_list[4][2]
                x_index, y_index = lm_list[8][1], lm_list[8][2]
                x_middle, y_middle = lm_list[12][1], lm_list[12][2]
                x_ring, y_ring = lm_list[16][1], lm_list[16][2]
                x_pinky, y_pinky = lm_list[20][1], lm_list[20][2]

                distance_vol = np.hypot(x_index - x_thumb, y_index - y_thumb)
                vol = np.interp(distance_vol, [30, 200], [minVol, maxVol])
                volume.SetMasterVolumeLevel(vol, None)

                distance_scroll = np.hypot(x_ring - x_thumb, y_ring - y_thumb)
                if distance_scroll < 50:
                    pyautogui.scroll(50)
                elif distance_scroll > 150:
                    pyautogui.scroll(-60)

                distance_brightness = np.hypot(x_middle - x_thumb, y_middle - y_thumb)
                brightness = np.interp(distance_brightness, [30, 200], [0, 100])
                sbc.set_brightness(int(brightness))

                distance_screenshot = np.hypot(x_pinky - x_thumb, y_pinky - y_thumb)
                if distance_screenshot < 40:
                    screenshot_path = f"screenshot_{int(time.time())}.png"
                    pyautogui.screenshot(screenshot_path)

    if last_command:
        if "volume up" in last_command:
            pyautogui.press("volumeup")
        elif "volume down" in last_command:
            pyautogui.press("volumedown")
        elif "brightness up" in last_command:
            sbc.set_brightness(min(100, sbc.get_brightness()[0] + 10))
        elif "brightness down" in last_command:
            sbc.set_brightness(max(0, sbc.get_brightness()[0] - 10))
        elif "scroll up" in last_command:
            pyautogui.scroll(50)
        elif "scroll down" in last_command:
            pyautogui.scroll(-50)
        elif "screenshot" in last_command:
            screenshot_path = f"screenshot_{int(time.time())}.png"
            pyautogui.screenshot(screenshot_path)
        elif "minimize all" in last_command:
            pyautogui.hotkey("win", "d")
        elif "open folder" in last_command:
            os.system("explorer C:\\")
        last_command = ""

    cv2.imshow("Hand & Voice Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
