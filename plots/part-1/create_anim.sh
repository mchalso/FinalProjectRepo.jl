#!/bin/bash

ffmpeg -f image2 -framerate 10 -i plot_%03d.png -loop -1 diffusion.mp4
