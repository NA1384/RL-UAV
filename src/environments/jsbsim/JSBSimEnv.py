import numpy as np
import jsbsim
import os
import time

class Env():

    def __init__(self, scene, orig, dest, n_acts, usePredefinedSeeds, dictObservation, dictAction, dictRotation, speed,
                 pause, qID, render, realTime):
        self.scenario = scene
        self.startingPosition = orig
        self.destinationPosition = dest
        self.previousPosition = orig
        self.startingOrientation = []
        self.desiredState = []
        self.n_actions = n_acts
        self.dictObservation = dictObservation
        self.dictAction = dictAction
        self.dictRotation = dictRotation
        self.startingVelocity = speed
        self.pauseDelay = pause
        self.qID = qID
        self.fsToMs = 0.3048  # conversion from feet per sec to meter per sec
        self.msToFs = 3.28084  # conversion from meter per sec to feet per sec
        self.radToDeg = 57.2957795  # conversion from radians to degree
        self.degToRad = 0.0174533  # conversion from deg to rad
        self.realTime = realTime
        self.id = "JSBSim"
        self.epochs = 0
        self.change = 25

        if(usePredefinedSeeds):
            np.random.seed(42)

        os.environ["JSBSIM_DEBUG"] = str(0)  # set this before creating fdm to stop debug print outs
        self.fdm = jsbsim.FGFDMExec('./src/environments/jsbsim/jsbsim/')  # declaring the sim and setting the path
        self.physicsPerSec = int(1 / self.fdm.get_delta_t())  # default by jsb. Each physics step is a 120th of 1 sec
        self.realTimeDelay = self.fdm.get_delta_t()
        self.fdm.load_model('c172r')  # loading cessna 172
        if render:  # only when render is True
            # Open Flight gear and enter: --fdm=null --native-fdm=socket,in,60,localhost,5550,udp --aircraft=c172r --airport=RKJJ
            self.fdm.set_output_directive('./data_output/flightgear.xml')  # loads xml that initates udp transfer
        self.fdm.run_ic()  # init the sim
        self.fdm.print_simulation_configuration()
        self.fdm["propulsion/active_engine"] = True  # starts the engine?
        self.fdm['propulsion/set-running'] = -1  # starts the engine?
        self.fdm["atmosphere/turb-type"] = 5  # Atmosphere model selection
        self.fdm["atmosphere/turbulence/milspec/windspeed_at_20ft_AGL-fps"] = 75  # Atmosphere model selection
        self.fdm["atmosphere/turbulence/milspec/severity"] = 6  # Atmosphere model selection

    def send_posi(self, posi, rotation):
        position = posi[:]
        position[self.dictObservation["pitch"]] = rotation[self.dictRotation["pitch"]]
        position[self.dictObservation["roll"]] = rotation[self.dictRotation["roll"]]
        position[self.dictObservation["yaw"]] = rotation[self.dictRotation["yaw"]]

        self.fdm["ic/lat-gc-deg"] = position[self.dictObservation["lat"]]  # Latitude initial condition in degrees
        self.fdm["ic/long-gc-deg"] = position[self.dictObservation["long"]]  # Longitude initial condition in degrees
        self.fdm["ic/h-sl-ft"] = position[self.dictObservation["alt"]]  # Height above sea level initial condition in feet

        self.fdm["ic/theta-deg"] = position[self.dictObservation["pitch"]]  # Pitch angle initial condition in degrees
        self.fdm["ic/phi-deg"] = position[self.dictObservation["roll"]]  # Roll angle initial condition in degrees
        self.fdm["ic/psi-true-deg"] = position[self.dictObservation["yaw"]]  # Heading angle initial condition in degrees


    def send_velo(self, rotation):

        self.fdm["ic/ve-fps"] = rotation[self.dictRotation["eastVelo"]] * self.msToFs  # Local frame y-axis (east) velocity initial condition in feet/second
        self.fdm["ic/vd-fps"] = -rotation[self.dictRotation["verticalVelo"]] * self.msToFs  # Local frame z-axis (down) velocity initial condition in feet/second
        self.fdm["ic/vn-fps"] = -rotation[self.dictRotation["northVelo"]] * self.msToFs  # Local frame x-axis (north) velocity initial condition in feet/second
        # self.fdm["propulsion/refuel"] = True  # refuels the plane?
        # self.fdm["propulsion/active_engine"] = True  # starts the engine?
        # self.fdm['propulsion/engine/set-running'] = 1  # starts the engine?
        # self.fdm['propulsion/tank[0]/contents-lbs'] = 200  # refuels the plane?

        if self.epochs % self.change == 0 and self.epochs != 0:
            self.fdm["atmosphere/turbulence/milspec/windspeed_at_20ft_AGL-fps"] = np.random.randint(75)  # Atmosphere model selection
            self.fdm["atmosphere/turbulence/milspec/severity"] = np.random.randint(6)  # Atmosphere model selection

        '''        
        The Milspec and Tustin models are described in the Yeager report cited below. They both use a Dryden 
        spectrum model whose parameters (scale lengths and intensities) are modelled according to MIL-F-8785C. 
        Parameters are modelled differently for altitudes below 1000ft and above 2000ft, for altitudes in between 
        they are interpolated linearly. The two models differ in the implementation of the transfer functions 
        described in the milspec. 
        
        To use one of these two models, set atmosphere/turb-type to 4 resp. 5, and specify values for 
        atmosphere/turbulence/milspec/windspeed_at_20ft_AGL-fps and atmosphere/turbulence/milspec/severity (the 
        latter corresponds to the probability of exceedence curves from Fig. 7 of the milspec, allowable range is 0 (
        disabled) to 7). atmosphere/psiw-rad is respected as well; note that you have to specify a positive wind 
        magnitude to prevent psiw from being reset to zero. 
        
        Link to Documentation:https://jsbsim-team.github.io/jsbsim/classJSBSim_1_1FGWinds.html
        '''

        self.fdm["ic/q-rad_sec"] = 0  # Pitch rate initial condition in radians/second
        self.fdm["ic/p-rad_sec"] = 0  # Roll rate initial condition in radians/second
        self.fdm["ic/r-rad_sec"] = 0  # Yaw rate initial condition in radians/second

        # client.sendDREF("sim/flightmodel/position/local_ax", 0)  # The acceleration in local OGL coordinates +ax=E -ax=W
        # client.sendDREF("sim/flightmodel/position/local_ay", 0)  # The acceleration in local OGL coordinates +=Vertical (up)
        # client.sendDREF("sim/flightmodel/position/local_az", 0)  # The acceleration in local OGL coordinates -az=S +az=N

    def getVelo(self):

        P = self.fdm["velocities/p-rad_sec"] * self.radToDeg  # The roll rotation rates
        Q = self.fdm["velocities/q-rad_sec"] * self.radToDeg  # The pitch rotation rates
        R = self.fdm["velocities/r-rad_sec"] * self.radToDeg  # The yaw rotation rates
        AoA = self.fdm["aero/alpha-deg"]  # The angle of Attack
        AoS = self.fdm["aero/beta-deg"]  # The angle of Slip
        values = [P, Q, R, AoA, AoS]

        return values

    def send_Ctrl(self, ctrl):
        '''
        ctrl[0]: + Stick in (elevator pointing down) / - Stick back (elevator pointing up)
        ctrl[1]: + Stick right (right aileron up) / - Stick left (left aileron up)
        ctrl[2]: + Peddal (Rudder) left / - Peddal (Rudder) right
        '''
        self.fdm["fcs/elevator-cmd-norm"] = -ctrl[0]  # Elevator control (stick in/out)?
        self.fdm["fcs/aileron-cmd-norm"] = ctrl[1]  # Aileron control (stick left/right)? might need to switch
        self.fdm["fcs/rudder-cmd-norm"] = -ctrl[2]  # Rudder control (pedals)
        self.fdm["fcs/throttle-cmd-norm"] = ctrl[3]  # throttle

    def get_Posi(self):
        lat = self.fdm["position/lat-gc-deg"]  # Latitude
        long = self.fdm["position/long-gc-deg"]  # Longitude
        alt = self.fdm["position/h-sl-ft"]  # altitude

        pitch = self.fdm["attitude/theta-deg"]  # pitch
        roll = self.fdm["attitude/phi-deg"]  # roll
        heading = self.fdm["attitude/psi-deg"]  # yaw

        r = [lat, long, alt, pitch, roll, heading]

        return r

    def getControl(self, action, observation):
        ctrl, actions_binary = self.scenario.getControl(action, observation)

        return ctrl, actions_binary

    def getDeepState(self, observation):
        velocities = self.getVelo()
        positions = observation[3:]
        state = self.scenario.getDeepState(velocities, positions)
        return state

    def getState(self, observation):
        state = self.scenario.getState(observation)
        return state

    def rewardFunction(self, action, newObservation):
        alt = self.fdm["position/h-agl-ft"]
        alpha = self.fdm["aero/alpha-deg"]
        reward, done = self.scenario.rewardFunction(action, newObservation, alt, alpha)
        return reward, done

    def step(self, action):
        position = self.get_Posi()

        newCtrl, actions_binary = self.getControl(action, position)

        self.send_Ctrl(newCtrl)
        for i in range(int(self.pauseDelay * self.physicsPerSec)):  # will mean that the input will be applied for pauseDelay seconds
            # If realTime is True, then the sim will slow down to real time, should only be used for viewing/debugging, not for training
            if(self.realTime):
                self.send_Ctrl(newCtrl)
                self.fdm.run()
                time.sleep(self.realTimeDelay)
            # Non realTime code: this is default
            else:
                self.send_Ctrl(newCtrl)
                self.fdm.run()

        position = self.get_Posi()
        if self.qID == "deep" or self.qID == "doubleDeep":
            state = self.getDeepState(position)
        else:
            state = self.getState(position)

        done = False
        reward = 0
        reward, done = self.rewardFunction(action, position)

        info = [position, actions_binary, newCtrl]

        return state, reward, done, info

    def reset(self):
        resetPosition, desintaionState = self.scenario.resetStartingPosition()
        self.startingOrientation = resetPosition
        self.desiredState = desintaionState
        self.send_posi(self.startingPosition, resetPosition)
        self.send_velo(resetPosition)
        self.epochs = self.epochs + 1

        self.fdm.run_ic()
        self.scenario.resetStateDepth()
        self.send_Ctrl([0, 0, 0, 0, 0, 0, 1])  # this means it will not control the stick during the reset
        new_posi = self.get_Posi()
        if self.qID == "deep" or self.qID == "doubleDeep":
            state = self.getDeepState(new_posi)
        else:
            state = self.getState(new_posi)

        return state
