from keras.preprocessing.image import img_to_array
import imutils
import cv2
from gaze_tracking import GazeTracking
import time
import datetime
import numpy as np

gaze = GazeTracking()

ongaze = 0
offgaze = 0
absgaze = 0
sumtask = 0
focuspercent = 0
distractedpercent = 0
abspercent = 0
onscreen = 0
offscreen = 0
onscreenpercent = 0
offscreenpercent = 0
maxpresence = 0
att = ""

cap = cv2.VideoCapture("input.mp4")

if not cap.isOpened():
    print("Unable to read video")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out = cv2.VideoWriter('Output_Eyegaze_Input_Video.mp4', cv2.VideoWriter_fourcc(*"mp4v"), 20, (frame_width, frame_height))

while True:
   
    ret, frame = cap.read()

    if ret:
      
        gaze.refresh(frame)
        frame = gaze.annotated_frame()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        text = ""
        vertical_c = gaze.vertical_ratio()
        horizontal_c = gaze.horizontal_ratio()

        if vertical_c is None or horizontal_c is None:
            text = "Eyes Not Detected"
            absgaze += 1
        elif vertical_c <= 0.37 or vertical_c >= 0.80 or horizontal_c <= 0.44 or horizontal_c >= 0.74:
            text = "Eyes Not Focused"
            offgaze += 1
        elif 0.37 < vertical_c < 0.80 and 0.44 < horizontal_c < 0.74:
            text = "Eyes Focused"
            ongaze += 1

        sumtask = ongaze + offgaze + absgaze
        focuspercent = round((ongaze * 100 / sumtask), 2) if sumtask != 0 else 0
        abspercent = round((absgaze * 100 / sumtask), 2) if sumtask != 0 else 0
        onscreen = ongaze
        offscreen = offgaze + absgaze
        onscreenpercent = focuspercent
        offscreenpercent = round((offscreen * 100 / sumtask), 2) if sumtask != 0 else 0

        maxpresence = max(onscreenpercent, offscreenpercent, abspercent)
        if onscreenpercent == maxpresence:
            att = "On Screen"
        elif offscreenpercent == maxpresence and abspercent != 100:
            att = "Off Screen"
        elif abspercent == 100:
            att = "No Attendance (null)"

        cv2.putText(frame, text, (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.putText(frame, "Overall Attendance: " + str(att), (50, 490), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.putText(frame, "On Screen Percentage: " + str(onscreenpercent) + " %", (50, 530),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, "Off Screen Percentage: " + str(offscreenpercent) + " %", (50, 560),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, "Undetected Eyes Percentage: " + str(abspercent) + " %", (50, 590),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, "Number of Focused Eyes: " + str(ongaze), (50, 620), cv2.FONT_HERSHEY_DUPLEX, 0.9,
                    (0, 255, 0), 2)
        cv2.putText(frame, "Number of Unfocused Eyes: " + str(offgaze), (50, 650), cv2.FONT_HERSHEY_DUPLEX, 0.9,
                    (0, 255, 0), 2)
        cv2.putText(frame, "Number of Undetected Eyes: " + str(absgaze), (50, 680), cv2.FONT_HERSHEY_DUPLEX, 0.9,
                    (0, 255, 0), 2)

        out.write(frame)

        f = open("Result_Eyegaze_Input_Video.txt", "w") 
        f.write("TEAM GLADIATORS \n\n")
        f.write("MachineKnight Season 2 \n\n")
        f.write("EyeGaze On Screen-Off Screen Detection \n")
        f.write("Overall Attendance: " + str(att) + " \n\n")
        f.write("On Screen Percentage: " + str(onscreenpercent) + " %\n")
        f.write("Off Screen Percentage: " + str(offscreenpercent) + " % \n")
        f.write("Undetected Eyes Percentage: " + str(abspercent) + " %\n \n")
        f.write("Number of Focused Eyes: " + str(ongaze) + "\n")
        f.write("Number of Unfocused Eyes: " + str(offgaze) + "\n")

        f.close()

        cv2.imshow("EyeGaze On Screen-Off Screen Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()

cv2.destroyAllWindows()
