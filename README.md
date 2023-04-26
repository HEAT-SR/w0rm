# W0RM
The HEAT-SR training data preprocessor

## Installation
W0rm depends on OpenCV and numpy.
The following command installs all the dependencies:
```
conda install -c conda-forge opencv numpy
```

## Usage
1. Add w0rm.py to PATH
2. Execute it.
    - `-s` is used to specify a source video file
    - `-o` is used to specify the directory into which the processed images will go

### Example
```
./w0rm.py -s ./example_video_file.mp4 -o data
```
