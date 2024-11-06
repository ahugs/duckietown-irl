#!/bin/sh 
export DISPLAY=:1
Xvfb :1 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &> /scratch/xvfb.log & 
exec "$@"