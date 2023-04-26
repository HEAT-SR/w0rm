#!/bin/env python

import cv2
import numpy as np

DESC = "A simple program preprocessing training video data into images"
RESOLUTIONS = [
        (48,32)
        ]

def arger() -> tuple[list[str], str]:
    """Parse command line arguments"""
    import argparse

    parser = argparse.ArgumentParser(
            prog = "w0rm",
            description = DESC
            )

    help_src = "Path to a video file to preprocess"
    help_out = "Path to the directory into which to put the processed data"

    parser.add_argument("-s", "--source", action="append", help=help_src)
    parser.add_argument(
            "-o",
            "--output",
            default="training_data",
            action="store",
            help=help_out
            )

    args = parser.parse_args()

    match args.source:
        case None:
            print("Nothing to process; exiting...")
            exit(1)
        case some:
            source = [x for x in set(some)]

    return source, args.output


def frame_extractor(target: str) -> list:
    """Extract the frames from a video into a list"""
    video = cv2.VideoCapture(target)

    count = 0
    success = 1
    frames = []

    while success:
        success, frame = video.read()
        if frame is not None:
            frames.append(frame)
        count += 1

    return frames


def frame_dump(frames: list, r: tuple[int, int], dest: str, src: str) -> None:
    """Write the processed frames to disk"""

    import os
    filename = (f"{dest}/{os.path.basename(src)}")
    os.makedirs(filename)

    for i, frame in enumerate(frames):
        cv2.imwrite(f"{filename}/frame_{i}_{r[0]}x{r[1]}.png", frame)


def processor_kernel(dest:str, vid_src: str) -> None:
    """
    Processing steps needed to be performed on the source files
    Used to create threads
    """
    frame_list = frame_extractor(vid_src)
    frame_dump(frame_list, (1920,1080), dest, vid_src)


def main() -> None:
    """w0rm main function"""
    src, dest = arger()

    threads = []

    import threading
    for s in src:
        print(f"Processing {s}")
        t = threading.Thread(target=processor_kernel, args=[dest,s])
        t.start()
        threads.append(t)

    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
