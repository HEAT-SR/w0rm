#!/bin/env python

import cv2 as cv
import numpy as np
import threading

DESC = "A simple program preprocessing training video data into images"
RESOLUTIONS = [
        (48,32)
        ]

def arger() -> tuple[list[str], str]:
def arger() -> tuple[list[str], list[str], str]:
    """Parse command line arguments"""
    import argparse

    parser = argparse.ArgumentParser(
            prog = "w0rm",
            description = DESC
            )

    help_src = "Path to a video file to preprocess"
    help_src2 = "Path to a pictures directory to preprocess"
    help_out = "Path to the directory into which to put the processed data"

    parser.add_argument("-s", "--source", action="append", help=help_src)
    parser.add_argument("-p", "--pictures", action="append", help=help_src2)
    parser.add_argument(
            "-o",
            "--output",
            default="training_data",
            action="store",
            help=help_out
            )

    args = parser.parse_args()
    sourcev = []
    sourcep = []

    if args.source == None and args.pictures == None:
        print("Nothing to process; exiting...")
        exit(0)
    if args.source != None:
            sourcev = [x for x in set(args.source)]
    if args.pictures != None:
            sourcep = [x for x in set(args.pictures)]

    return sourcev, sourcep, args.output


def frame_extractor(target: str):
    """Extract the frames from a video into a ndarray"""
    video = cv.VideoCapture(target)

    count = 0
    success = 1
    frames = []

    while success:
        success, frame = video.read()
        if frame is not None:
            grayframe = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            frames.append(grayframe)
        count += 1

    return frames[0].shape[:2][::-1], np.array(frames)


def frame_dump(frames, r: tuple[int, int], dest: str, src: str) -> None:
    """Write the processed frames to disk"""

    import os
    filename = (f"{dest}/{os.path.basename(src)}/{r[0]}x{r[1]}")
    os.makedirs(filename)

    for i, frame in enumerate(frames):
        cv.imwrite(f"{filename}/frame_{i}_{r[0]}x{r[1]}.png", frame)


def crop3by2(frames, shape: tuple[int, int]):
    """Crop the frames to fit the aspect ratio of the ir sensors"""

    w,h = shape
    major = max(w,h)

    if major == w:
        crop_w = h * 3/2
        crop_h = h 
    else:
        crop_w = w
        crop_h = w * 2/3

    center = (int(w/2), int(h/2))
    offset_w, offset_h = int(crop_w/2), int(crop_h/2)

    base_w, top_w = center[0]-offset_w,center[0]+offset_w
    base_h, top_h = center[1]-offset_h,center[1]+offset_h

    cropped = []

    for frame in frames:
        cropped.append(frame[base_w:top_w][base_h:top_h])

    return np.array(cropped)


def downscale(frames, shape: tuple[int, int], dest: str, src: str):
    """Downscale the frames into various resolution"""
    starter_rez = (48,32)

    ry = [x for x in range(shape[1])[starter_rez[1]:] if x%starter_rez[1] == 0]
    rx = [int(x * 3/2) for x in ry]

    rezzos = zip(rx,ry)
    threads = []

    for rez in rezzos:
        frame_list = []
        for frame in frames:
            frame = cv.resize(frame, rez, interpolation=cv.INTER_LANCZOS4)
            frame_list.append(frame)
        targs = [np.array(frame_list), rez, dest, src]
        t = threading.Thread(target=frame_dump, args=targs)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()


def processorv_kernel(dest:str, vid_src: str) -> None:
    """
    Processing steps needed to be performed on the source files
    Used to create threads
    """

    frame_size, frame_list = frame_extractor(vid_src)
    frame_list = crop3by2(frame_list, frame_size)
    downscale(frame_list, frame_size, dest, vid_src)


def processorp_kernel(dest:str, pic_src: str) -> None:
    """
    Processing steps needed to be performed on the source folders
    Used to create threads
    """
    frame_size = (1920, 1080)
    frame_list = frame_loader(pic_src)
    # frame_list = place3by2(frame_list)
    # downscale(frame_list, frame_size, dest, pic_src)
    pass


def main() -> None:
    """w0rm main function"""
    srcv, srcp, dest = arger()

    threads = []

    for s in srcv:
        print(f"Processing {s}")
        t = threading.Thread(target=processorv_kernel, args=[dest,s])
        t.start()
        threads.append(t)
    for s in srcp:
        print(f"Processing {s}")
        t = threading.Thread(target=processorp_kernel, args=[dest,s])
        t.start()
        threads.append(t)

    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
