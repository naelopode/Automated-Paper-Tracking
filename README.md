
# Automated Paper Tracking

This project aims to automatize an experiment: free falling paper trajectory analysis.

## Goal
Track accurately 6 Degrees of freedom of a falling paper and be able to reproduce the experiment hundreds of times to find global patterns. Attempt to extract meaningful data from the experiments.

## Setup
The experiment make use of:
- An Arduino to command a stepper motor to drop and grab a paper.
- A paper (with two different april tags to track the trajectory).
- A reference tag, to compute a relative distance.
- 3 GoPro Cameras (More is better but more computation per experiment).
- A screen (phone or computer) using QR precision time to sync internal clocks of the cameras.

![Setup](images/setup.png)
### More information on the Setup
- The cameras record at 2.7K, 250fps, allowing our algorithm to detect a small tag and to capture a good amount of data points.
- Each cameras needs to have it's deformation matrix calculated with a Checkerboard.

## Experiment pipeline
About every 5 to 10 experiment, it's advised to sync the internal clock to avoid any offset between the cameras.
The experiment program controlling the Arduino and camera can the be enabled. The grabber allows some time to be reloaded with a paper. The cameras are the enabled and start recording. The paper is dropped. Finally, the cameras stop recording and the files are downloaded 
![Experiment pipeline](images/exp_pipeline.png)

## Processing pipeline
We extract each frame from the videos, compute the offset between each video with milliseconds precision using the metadata. We then compute the position of each tags (reference tag and falling tag), evaluate the reference position for each camera. We then export the relative position and plot the 6 degrees of freedom.


# Tools used
- [GoPro Precision Date and Time QR for Lab enabled cameras, used for calibration](https://gopro.github.io/labs/control/precisiontime/)
- [Apriltag for video, some code reused and used for calibration](https://github.com/yanshil/video-apriltags)
