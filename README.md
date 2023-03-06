

### Built With

This project is built with these frameworks, libraries, repositories and software:
* [tensorflow](https://www.tensorflow.org/)
* [XPlaneConnect](https://github.com/nasa/XPlaneConnect)
* [XPlane 11](https://www.x-plane.com/)
* [JSBSim](https://github.com/JSBSim-Team/jsbsim)
* [Flightgear](https://www.flightgear.org/)



<!-- GETTING STARTED -->
## Getting Started

Simple clone this repository to your local filesystem:
```sh
git clone https://github.com/JDatPNW/QPlane
```

### Prerequisites

Tested and running with:

|Software | Version|
|-----|-----|
|XPlane11 Version: | 11.50r3 (build 115033 64-bit, OpenGL)|
|JSBSim Version: | 1.1.5 (GitHub build 277)|
|Flightgear Version: | 2020.3.6|
|XPlaneConnect Version: | 1.3-rc.2|
|Python Version: | 3.8.2|
|numpy Version: | 1.19.4|
|tensorflow Version: | 2.3.0|
|Anaconda Version: | 4.9.2|
|Windows Version: | 1909|


### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/JDatPNW/QPlane
   ```
2. Install the above listed software (other versions might work)
    * For JSBSim clone the JSBsim repo into `src/environments/jsbsim`
    * For visualizing JSBSim download the c172r plane model in the Flightgear Menu

<!-- USAGE EXAMPLES -->
## Usage

Once downloaded and installed, simply execute the `QPlane.py` file to run and test the code.
* For the XPlane Environment, XPlane (the game) needs to run.
* For JSBSim with rendering, Flightgear needs to run with the following flags `--fdm=null --native-fdm=socket,in,60,localhost,5550,udp --aircraft=c172r --airport=RKJJ`

<!-- LICENSE -->
## License

Distributed under the MIT License. See `misc/LICENSE` for more information.
