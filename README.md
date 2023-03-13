## Overview

The software is designed to train an agent via Deep Reinforcement Learning to control an aircraft in a simulated environment. The code uses a Proximal Policy Optimization (PPO) algorithm for the learning process that is implemented using Tensorflow. The agent trains in the environment, the JSBSim Flight Dynamics Model, until it reaches a control policy by optimizing a set reward function (having a stable plane).

## Dependencies

Overview of major dependencies and software:
* [Tensorflow](https://www.tensorflow.org/)
* [XPlaneConnect](https://github.com/nasa/XPlaneConnect)
* [XPlane 11](https://www.x-plane.com/)
* [JSBSim](https://github.com/JSBSim-Team/jsbsim)
* [Flightgear](https://www.flightgear.org/)

### Tested Versions

|Software | Version|
|-----|-----|
|XPlane11 | 11.50r3|
|JSBSim | 1.1.5|
|Flightgear | 2020.3.6|
|XPlaneConnect | 1.3-rc.2|
|Python | 3.9.13|
|numpy | 1.22.3|
|Tensorflow | 2.10.0|
|Miniconda | 22.9.0|
|MacOS | 13.2.1|

## Dependency Installation

Installing Tensorflow of MacOS is a nontrivial process, especially on the Apple Silicon hardware that was used. For installing Tensorflow on MacOS, the following tutorial was used successfully:
* [Apple Tensorflow Installation Guide](https://developer.apple.com/metal/tensorflow-plugin/)

For Windows installations follow the official Tensorflow Guide:
* [Official Tensorflow Installation Guide](https://www.tensorflow.org/install)

Please note that the software was never tested on a Windows system, so some unknown dependency errors may exist on this platform. The MacOS tutorial includes the instructions for installing Miniconda as well, however on Windows download the appropriate installer from Miniconda.
* [Miniconda Installation Guide](https://docs.conda.io/en/latest/miniconda.html)

Other dependencies such as matplotlib, time, os and socket are used in the software; however, these libraries should be included in the listed Miniconda installation. If errors regarding these or other modules occur, please install them individually using either conda or pip.

Furthermore, as XPlane is a commercial flight simulation software which requires a license, the instructions will mostly cover JSBSim with visualisation in FlightGear. Basic instructions for XPlane installation are still provided, but please note that the code only works with XPlane 11 and not XPlane 12.

### JSBSim & FlightGear installation

Installing `jsbsim` using `pip` can be achieved with:

```bash
> pip install jsbsim
```

FlightGear can be downloaded at the following link. Follow instructions in installer for completing installation (5GB of storage is required):
* [FlightGear Download](https://www.flightgear.org/download/)

As installation for FlightGear does take a considerable amount of storage space just for visualising the training process, an example of an untrained and trained agent is provided along with the project video.
* [Example Video](https://youtube.com/playlist?list=PLDBfEt2X6tiz8CO5XBbErqCcoeztl3HyU)

### XPlane Installation

1. To use XPlane, purchase a lisence and download the game from Steam at the link below (20GB of storage is required):
* [XPlane 11 Download](https://store.steampowered.com/app/269950/XPlane_11/)

2. XPlane requires the XPlane Connect plugin to communicate with the code that can be installed by:
* Download the `XPlaneConnect.zip` file from the latest release on the [releases](https://github.com/nasa/XPlaneConnect/releases) page.
* Copy the contents of the .zip archive to the plugin directory (`[X-Plane Directory]/Resources/plugins/`)

Cloning the XPlane Connect repository will not work as only the zip file contains the compiled version of the software.

### Repository Installation

1. Clone the repository or download zipped code file.

   ```bash
   > git clone https://github.com/NA1384/RL-UAV
   ```
2. After installing dependencies (other versions might work)
    * Clone the JSBsim repo into `src/environments/jsbsim`
    
      ```bash
      > git clone https://github.com/JSBSim-Team/jsbsim
      ```
    * For visualizing JSBSim download the c172r plane model in the FlightGear Menu
    * FlightGear scenery can be enabled by turning on TerraSync under Downloads in Settings.

## Testing

Once downloaded and installed, simply execute the `PPO-Plane.ipynb` file to run and test the code.
* For the rendering with XPlane, XPlane just needs to be running.
* For JSBSim with rendering, Flightgear needs to run with the following flags under Additional Settings in the Settings Menu:

```bash
--fdm=null --native-fdm=socket,in,60,localhost,5550,udp --aircraft=c172r --airport=RKJJ
```

Please note that by default rendering and loading the trained model are not enabled in the file. Loading the example trained model provided and visualising in FlightGear can easily be enabled by changing the following variables to `True` in `PPO-Plane.ipynb` like below:

```bash
loadModel = True  # will load trained model for tf if True
jsbRender = True # will send UDP data to flight gear for rendering if True
jsbRealTime = True  # will slow down the physics to portrait real time rendering
```

Some utility files are also included under `/utils`, but they are mostly used for graphing data and aren't used in the `PPO-Plane.ipynb` itself, but included for review nonetheless.
