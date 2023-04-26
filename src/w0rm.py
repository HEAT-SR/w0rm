#!/bin/env python

DESC = "A simple program preprocessing training video data into images"

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

    print(source)

    return source, args.output


def main() -> None:
    """w0rm main function"""
    src, dest = arger()


if __name__ == "__main__":
    main()
