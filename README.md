


# MQTT Broker and Object Detection with YOLOv11


This project uses a Docker container to run the object detection model and compose it whit a MQTT broker service.

## Table of Contents
-----------------

* [Overview](#overview)
* [Requirements](#requirements)
* [Usage](#usage)
* [Configuration](#configuration)
* [Troubleshooting](#troubleshooting)

## Overview
------------

The vision docker service consists of three main components:

* The Dockerfile that uses as base image ultralytics/ultralytics:latest (for YOLOv11) and then adds the Orbbec dependencies in order to have a plug and play vision system.
* An _inference.py_ script that runs the object detection model and sends the results to the MQTT broker.
* The trained model weights for object detection _best.pt_

Docker Compose is used to run the vision service with the MQTT broker.

## Requirements
------------

* Docker installed on your system
* Python 3.x installed on your system
* Paho MQTT library installed on your system
* RealSense Camera connected to your system

## Usage
-----

### Building the Docker Image

To build the Docker image, run the following command in the root directory of the project:
```bash
docker compose build
```
During the building phase, inference.py and best.pt will be copied inside the image.
### Running the Docker Container

To run the Docker container, use the following command:
```bash
docker compose up
```
### Running the MQTT Subscriber

To read the result from the vision service, run the MQTT subscriber. Use the following command:
```
python subscriber.py
```
### Configuring the MQTT Broker

To configure the MQTT broker settings, such as the listener ports and password file, modify the `mosquitto.conf` file.

In /auth/pwd.txt there are the access credentials for the MQTT broker.
In order to add others use:
```
mosquitto_passwd -b <filename> <username> <password>
```
this will create a filename with the sha256 hash of the password for the user. Copy and paste inside pwd.txt.

## Configuration of Inference
-------------

In the _inference.py_ file you can configure the parsing of the results from object detection (in function _parse_). So if you have trained a model with other class or feature, along with substitude best.pt with the new one you can configure the parsing to match your needs.


## Troubleshooting
--------------

* At the moment ip address of the broker is hardcoded in inference.py. The mqtt services uses the ip address of the host.

* If you want to see Yolo results on sceen (calling method _model.predict(image, stream=True, conf=min_confidence, show=**True**, verbose=False)_ inside _inference.py_) you must have permissions to let docker use the display. To do that run command: 
```
xhost +local:root
````

* This project uses mosquitto broker so stop eventually service running on host with command: 
```
sudo service stop mosquitto
```

* If you see the error `Error starting userland proxy: listen tcp4 0.0.0.0:80: bind: address already in use` probably there is apache2 running on host that is using port 80. To stop apache2 run
```
sudo service apache2 stop
```