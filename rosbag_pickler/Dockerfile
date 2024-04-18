# Use the official ROS base image 
FROM ros:noetic 

# Install Python3, pip, and necessary packages 
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-opencv \
    python3-matplotlib

# install numpy and other Python dependencies via pip 
RUN pip3 install numpy 

# Install additional ROS packages 
RUN apt-get install -y \ 
    ros-noetic-cv-bridge \ 
    ros-noetic-rosbag \ 
    ros-noetic-image-transport \ 
    ros-noetic-sensor-msgs 

# Set the working directory 
WORKDIR /app 
