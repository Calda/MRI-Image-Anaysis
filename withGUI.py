import io
import time
import threading
import picamera
from PIL import Image
import pygame
import sys
from pygame.locals import *
import numpy as np
import cv2
import scipy.ndimage.filters as scipy
import Buttons
import RPi.GPIO as GPIO
from time import sleep
import os
import math

def saveCalibration(scaleInMM):
    global calibratedScale
    calibratedScale = scaleInMM
    try:
        os.remove("MRI_SCALE.conf")
    except OSError:
        pass
    f = open('MRI_SCALE.conf', 'w')
    f.write(str(scaleInMM))
    f.close()

def getCalibrationScale():
    try:
        calibration = open("MRI_SCALE.conf")
        scale = float(calibration.read())
    except (IOError, ValueError):
        return 1.0
    return scale

calibratedScale = getCalibrationScale()
motionThreshold = 10

timeStart = time.time()

#Create the pool of image processors
done = False
tracking = False
lock = threading.Lock()
pool = []

numberOfFrames = 10
minDiff = 50
previousData = None

imageScale = 2

first = True
centroid1 = (0,0)
centroidPrevious = (0,0)
distanceMoved = 0
farthestDistance = 0
deltaDistance = []
for x in range(1, 60):
    deltaDistance.append(0)
frameCount = 0
thresholdCurrent = 0
saveString = ""

pygame.init()
screen = pygame.display.set_mode((720,420), 0, 32)
background = pygame.Surface(screen.get_size())
background.fill((5,5,5))
screen.blit(background, (0,0))
stream = io.BytesIO()

pygame.display.set_caption("MRI Image Analysis by Cal Stephens")

font = pygame.font.SysFont("Calibri", int(20))
cameraViewLabel = font.render("Camera View", 1, (255,255,255))
screen.blit(cameraViewLabel, (80, 402))
edgesViewLabel = font.render("Edges and Centroid", 1, (255,255,255))
screen.blit(edgesViewLabel, (310, 402))
graphViewLabel = font.render("Absolute Centroid Distance vs Time", 1, (255,255,255))
screen.blit(graphViewLabel, (485, 190))
calibratedScaleLabel = font.render("1 pixel = " + str(round(calibratedScale, 1)) + " mm", 1, (255,255,255))
screen.blit(calibratedScaleLabel, (485, 205))
thresholdLabel = font.render("Motion Threshold: " + str(motionThreshold) + " mm", 1, (255, 255, 255))
screen.blit(thresholdLabel, (485, 240))

#draw empty graph
pygame.draw.rect(screen, (15, 15, 15), pygame.Rect(480, 0, 400, 185))
for x in range(0, 59):
    if x < 58:
        pygame.draw.line(screen, (155, 0, 0), (481 + x * 4, 181), (481 + (x + 1) * 4, 181))
    pygame.draw.rect(screen, (255, 100, 100), pygame.Rect(480 + x * 4, 180, 3, 3))

    
startButton = Buttons.Button()
startButton.create_button(screen, (80,80,80), 487, 355, 70, 20, 0, "Start", (255,255,255))
stopButton = Buttons.Button()
stopButton.create_button(screen, (80,80,80), 565, 355, 70, 20, 0, "Stop", (255,255,255))
quitButton = Buttons.Button()
quitButton.create_button(screen, (80,80,80), 643, 355, 70, 20, 0, "Quit", (255,255,255))
saveButton = Buttons.Button()
saveButton.create_button(screen, (80,80,80), 250, 4002, 70, 20, 0, "Save", (255,255,255))
lightButton = Buttons.Button()
lightButton.create_button(screen, (80,80,80), 487, 380, 148, 20, 0, "Light on", (255,255,255))
calibrateButton = Buttons.Button()
calibrateButton.create_button(screen, (80,80,80), 643, 380, 70, 20, 0, "Calibrate", (255,255,255))
upThresholdButton = Buttons.Button()
upThresholdButton.create_button(screen, (80,80,80), 683, 237, 25, 20, 0, "+1", (255,255,255))
downThresholdButton = Buttons.Button()
downThresholdButton.create_button(screen, (80,80,80), 653, 237, 25, 20, 0, "-1", (255,255,255))

font = pygame.font.SysFont("Calibri", 20)

GPIO.setmode(GPIO.BOARD)
GPIO.setup(13, GPIO.OUT)
GPIO.setup(11, GPIO.OUT)
GPIO.output(11,True)
p = GPIO.PWM(13, 500)
p.start(0)
p.ChangeDutyCycle(0)
lightDutyCycle = 0

calibrateMode = False
failedCalibration = False

def calibrate(edges):
    global failedCalibration
    failedCalibration = False
    pixelArea = 0
    
    pygame.draw.rect(screen,(5,5,5), (485,205,200,15), 0)
    calibratedScaleLabel = font.render("CALIBRATION IN PROGRESS...", 1, (255,255,255))
    screen.blit(calibratedScaleLabel, (485, 205))
    pygame.display.update()

    static = np.array(edges)
    print "starting calibration"
    pixelArea = interior_rec(edges, (30,50))
    print "PIXEL AREA=" + str(pixelArea)
    if not failedCalibration:
        calibrateMode = False
        pygame.draw.rect(screen,(5,5,5), (485,205,200,15), 0)
        calibratedScaleLabel = font.render("1 pixel = " + str(round(calibratedScale, 1)) + " mm", 1, (255,255,255))
        screen.blit(calibratedScaleLabel, (485, 205))
        saveCalibration(76.2 / math.sqrt(pixelArea))
    else:
        pygame.draw.rect(screen,(5,5,5), (485,205,200,15), 0)
        calibratedScaleLabel = font.render("FAILED CALIBRATION.", 1, (255,255,255))
        screen.blit(calibratedScaleLabel, (485, 205))
    return edges

def interior_rec(image, coords):
    global bottomLeft, bottomRight, topLeft, topRight, failedCalibration
    pixelCount = 0
    if coords[0] is -1 or coords[1] is -1 or coords[0] is 60 or coords[1] is 100:
        print "NOT INSIDE INDEX CARD!!!"
        failedCalibration = True
        return pixelCount
    if iscolor(image, coords, (255,0,0)):
        return pixelCount
    if iscolor(image, coords, (255, 255, 255)):
        return pixelCount
    if iscolor(image, coords, (0,0,0)):
        setpixel(image, coords, (255, 0, 0))
        pixelCount += 1
        pixelCount += interior_rec(image, (coords[0], coords[1] + 1))
        if not failedCalibration:
            pixelCount += interior_rec(image, (coords[0], coords[1] - 1))
        if not failedCalibration:
            pixelCount += interior_rec(image, (coords[0] + 1, coords[1]))
        if not failedCalibration:
            pixelCount += interior_rec(image, (coords[0] - 1, coords[1]))
    return pixelCount

def interior(image, coords, xdir, ydir):
    foundWhite = False
    while not foundWhite and coords[0] is not -1 and coords[1] is not -1 and coords[0] is not 60 and coords[1] is not 100:
        if not iscolor(image, coords, (0,0,0)):
            return True
        coords = (coords[0] + xdir, coords[1] + ydir)
    return False
    
def setpixel(image, coords, color):
    image[coords[0], coords[1], 0] = color[0]
    image[coords[0], coords[1], 1] = color[1]
    image[coords[0], coords[1], 2] = color[2]

def iscolor(image, coords, color):
    red = image[coords[0], coords[1], 0] == color[0]
    if not red:
        return False
    blue = image[coords[0], coords[1], 1] == color[1]
    if not blue:
        return False
    green = image[coords[0], coords[1], 2] == color[2]
    return green

with picamera.PiCamera() as camera:
    camera.resolution = (200, 120)
    camera.framerate = 30
    camera.contrast = 100
    while not done:
        for event in pygame.event.get():
            if event.type in (QUIT, KEYDOWN):
                done = True
            if event.type == MOUSEBUTTONDOWN:
                if startButton.pressed(pygame.mouse.get_pos()):
                    tracking = True
                    first = True
                    centroid1 = (0,0)
                    centroidPrevious = (0,0)
                    distanceMoved = 0
                    farthestDistance = 0
                    frameCount = 0
                    thresholdCurrent = 0
                    deltaDistance = []
                    for x in range(1, 60):
                        deltaDistance.append(0)
                elif stopButton.pressed(pygame.mouse.get_pos()):
                    tracking = False
                elif quitButton.pressed(pygame.mouse.get_pos()):
                    done = True
                #if saveButton.pressed(pygame.mouse.get_pos()):
                    #print saveString
                elif lightButton.pressed(pygame.mouse.get_pos()):
                    if lightDutyCycle is 100:
                        p.ChangeDutyCycle(0)
                        lightDutyCycle = 0
                        lightButton.create_button(screen, (80,80,80), 487, 380, 148, 20, 0, "Light on", (255,255,255))
                    else:
                        p.ChangeDutyCycle(100)
                        lightDutyCycle = 100
                        lightButton.create_button(screen, (80,80,80), 487, 380, 148, 20, 0, "Light off", (255,255,255))
                elif calibrateButton.pressed(pygame.mouse.get_pos()):
                    calibrateMode = True
                elif upThresholdButton.pressed(pygame.mouse.get_pos()):
                    motionThreshold = motionThreshold + 1
                    pygame.draw.rect(screen,(5,5,5), (485,240,165,15), 0)
                    thresholdLabel = font.render("Motion Threshold: " + str(motionThreshold) + " mm", 1, (255, 255, 255))
                    screen.blit(thresholdLabel, (485, 240))
                    pygame.display.update()
                elif downThresholdButton.pressed(pygame.mouse.get_pos()):
                    if motionThreshold > 1:
                        motionThreshold = motionThreshold - 1
                    pygame.draw.rect(screen,(5,5,5), (485,240,165,15), 0)
                    thresholdLabel = font.render("Motion Threshold: " + str(motionThreshold) + " mm", 1, (255, 255, 255))
                    screen.blit(thresholdLabel, (485, 240))
                    pygame.display.update()
        camera.capture(stream, format='jpeg', use_video_port=True)
        stream.seek(0)
        timeStartNumpy = time.time()
        data = np.fromstring(stream.getvalue(), dtype=np.uint8);
        currentData = cv2.imdecode(data, 1)
        currentData = currentData[:, :, ::-1]
        currentData = currentData.astype(np.int16)

        averageColor = (currentData.sum(axis=2) / 3)
        sobelHorizontal = scipy.sobel(averageColor, 0)
        sobelVertical = scipy.sobel(averageColor, 1)
        magnitude = np.hypot(sobelHorizontal, sobelVertical)
        final = magnitude * (255 / np.max(magnitude))
        final = final.astype(np.int16)

        threshold = 40
        final[final < threshold] = 0
        edges = np.transpose(np.nonzero((final >= threshold)))
        count = edges.shape[0]
        totals = edges.sum(axis=0)
        xTotal = totals[0]
        yTotal = totals[1]

        if count is 0:
            count = 1
        centroid = (xTotal / count, yTotal / count)
        frameCount += 1
        #print str(count) + " pixels over threshold of " + str(threshold)
        
        frameDistance = 0
        absoluteDistance = 0
        
        if first is True:
            first = False
            centroid1 = centroid
            centroidPrevious = centroid
        elif tracking:
            distance = ((centroid1[0] - centroid[0]) ** 2 + (centroid1[1] - centroid[1]) ** 2) ** 0.5
            distance = round(distance)  
            #print "Centroid is currently " + str(distance) + " pixels from the original position."
            absoluteDistance = distance
            if distance > farthestDistance:
                farthestDistance = distance
            deltaDistance.remove(deltaDistance[0])
            deltaDistance.append(distance)
            
            distance = ((centroidPrevious[0] - centroid[0]) ** 2 + (centroidPrevious[1] - centroid[1]) ** 2) ** 0.5
            distance = round(distance)
            #print "Centroid moved " + str(distance) + " pixels from its previous position."
            frameDistance = distance
            distanceMoved += distance

            ##change to distance from current and distance to original##
            if distanceMoved > 10:
                GPIO.output(11, True)
            else:
                GPIO.output(11, False)
            centroidPrevious = centroid
            #frameInfo = "Frame " + str(frameCount - 1) + "{\n"
            #frameInfo += "Time: " + str(time.time()) + "\n"
            #frameInfo += "Centroid position: (" + str(centroid[0]) + ", " + str(centroid[1]) + ")\n"
            #frameInfo += "Frame movement: " + str(distance) + "\n"
            #frameInfo += "Total movement " + str(distanceMoved) + "\n}\n"
            #saveString += frameInfo

        final = np.dstack((final, final, final))
        black = pygame.Surface((480,480))
        pygame.draw.rect(black,(0,0,0), (0,0,480,480), 0)
        screen.blit(black, (0,440))
        if calibrateMode == True:
            final = calibrate(final)
        if tracking:
            text = font.render(("Centroid moved " + str(frameDistance) + " pixels"), 1, (255,255,255))
            screen.blit(text, (20, 440))
            text = font.render(("Centroid is " + str(absoluteDistance) + " pixels from start"), 1, (255,255,255))
            screen.blit(text, (20, 460))
            for x in range(4):
                x2 = x - 2
                for y in range(4):
                    y2 = y - 2
                    final[centroid[0] + x2,centroid[1] + y2] = (200,0,0)
                    final[centroid1[0] + x2,centroid1[1] + y2] = (100,0,0)
            #updateStatusBox
            boxColor = (100,190,100)
            boxText = "Within threshold"
            if absoluteDistance * calibratedScale > motionThreshold:
                boxColor = (255, 100, 100)
                boxText = "Over threshold"
            elif absoluteDistance * calibratedScale > (motionThreshold * 0.75):
                boxColor = (255, 190, 100)
                boxText = "Nearing threshold"
            pygame.draw.rect(screen, boxColor, pygame.Rect(500, 267, 200, 75))
            text = font.render(boxText + ": " + str(round(absoluteDistance * calibratedScale, 2)) + " mm", 1, (20,20,20))
            screen.blit(text, (505, 327))
        #draw graph
        if frameCount % 1 == 0:
            pygame.draw.rect(screen, (15, 15, 15), pygame.Rect(480, 0, 400, 185))
            maxValue = (motionThreshold / calibratedScale)
            yPerTick = float(180) / 5
            for scaleTick in range(0,5):
                pygame.draw.rect(screen, (30,30,30), pygame.Rect(480, 180 - (scaleTick * yPerTick), 400, 2))
            for value in deltaDistance:
                if value > maxValue:
                    maxValue = value
            valuePerPixel = float(maxValue) / float(180)
            for x in range(0, 59):
                if x < 58:
                    pygame.draw.line(screen, (155, 0, 0), (481 + x * 4, 181 - int(float(deltaDistance[x]) / valuePerPixel)), (481 + (x + 1) * 4, 181 - int(float(deltaDistance[x + 1]) / valuePerPixel)))
                pixelColor = (100, 190, 100)
                if float(deltaDistance[x]) * calibratedScale > motionThreshold:
                    pixelColor = (255, 100, 100)
                elif float(deltaDistance[x]) * calibratedScale > (motionThreshold * 0.75):
                    pixelColor = (255, 190, 100)
                pygame.draw.rect(screen, pixelColor, pygame.Rect(480 + x * 4, 180 - int(float(deltaDistance[x]) / valuePerPixel), 3, 3))
            for scaleTick in range(0,5):
                scaleMark = font.render(str(round(scaleTick * yPerTick * valuePerPixel * calibratedScale, 1)), 1, (255, 255, 255))
                screen.blit(scaleMark, (483, 166 - (scaleTick * yPerTick)))
            thresholdLineY = 181 - int(motionThreshold / calibratedScale / valuePerPixel)
            pygame.draw.rect(screen, (255, 75, 75), pygame.Rect(480, thresholdLineY, 400, 2))
               
        originalOutput = pygame.Surface((240,400))
        sobelOutput = pygame.Surface((240,400))
        pygame.surfarray.blit_array(originalOutput, np.repeat(np.repeat(currentData, 2, axis=0), 2, axis=1))
        pygame.surfarray.blit_array(sobelOutput, np.repeat(np.repeat(final, 2, axis=0), 2, axis=1))
        screen.blit(originalOutput, (0,0))
        screen.blit(sobelOutput, (240,0))
        
        pygame.display.update()
        #print str(time.time() - timeStartNumpy) + " for one frame"
        if calibrateMode == True:
            sleep(5)
            calibrateMode = False

print str(time.time() - timeStart) + " complete time"
print "Centroid moved a total of " + str(distanceMoved) + " pixels and was " + str(farthestDistance) + " pixels away at the farthest."

pygame.quit()
p.stop()
GPIO.cleanup()
