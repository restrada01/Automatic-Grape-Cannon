# Code

This script will use OpenCV and a Triangle Similarity algorithm to measure how far away a face is from the camera.

Knowing the distance, I can then find the cooresponding vector to the object's mouth. By measuring the vector from the grape launcher to the camera,
I should be able to calculate the angles needed for the servos to point the projectile at the object's mouth.

The second part will be to run an optimization algorithm for the controller to choose the best angle to shoot the grape so that the object can catch it without moving.
First, I have to make this work.
