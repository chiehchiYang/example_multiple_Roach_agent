#!/bin/sh
killall -9 -r CarlaUE4-Linux 
../../CarlaUE4.sh -RenderOffScreen &
# ../../CarlaUE4.sh  &
sleep 5
python obstacle_scenario.py --town Town03



# Town05


# Town10HD -- > done 
