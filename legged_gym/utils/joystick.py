from dataclasses import dataclass
import numpy as np
import pygame

@dataclass
class JoystickInput:
    """Adjustable parameters for the camera control

        Super neat description

        Args:
            maxRadiusVel :  The linear velocity with which to zoom in and out 
                            manually using the square and circle button
            
            autoPhiVel   :  The angular velocity with which the cinematic mode circles
                            around the robot. Trigger Cinematic mode with L1 or R1 button.
                            To turn it off, press both buttons at the same time
            
            maxPhiVel    :  The angular velocity with which to circle around the robot
                            manually with the L2 and R2 button
            
            maxThetaVel  :  The angular velocity with which to move up and down 
                            manually using the x and triangle button
    """
    # Variables to adjust
    maxRadiusVel = 2 # m/s speed radial
    autoPhiVel = np.pi/4 # rad/s speed horizontally in cinematic mode
    maxPhiVel = np.pi # rad/s speed horizontally
    maxThetaVel = np.pi / 4 # rad/s speed vertically
    dt = 0.02 # inference time

    # helpers
    dRadius = 0.0 # m
    ldPhi = 0.0 # rad (l/rdphi = left/right dphi, to not get weird interference while pressing ...
    rdPhi = 0.0 # rad   ... left and right button concurrently)
    dTheta = 0.0 # rad

    rotate_automatically = False # cinematic mode
    counterClockwise = True

    xVel = 0.0
    yVel = 0.0
    yawVel = 0.0

    switchRobot = False
    resetCamera = False

class ViewerCamera:
    """Utility for computing camera position for isaac gym viewer."""

    def __init__(self, radius0=2, phi0=45 * np.pi / 180, theta0=80 * np.pi / 180):
        self.radius = radius0 # meter distance from robot
        self.phi = phi0 # rotates over yaw axis of robot  
        self.theta = theta0 # rotates up and down 
        self.initState = [radius0, phi0, theta0]

    def reset(self):
        [self.radius, self.phi, self.theta] = self.initState

    def get_camera_position(self, robotRoot, joystickInput: JoystickInput):
        if joystickInput.resetCamera:
            self.reset()
        cameraPos = np.zeros(3)
        if joystickInput.rotate_automatically:
            self.phi += (joystickInput.counterClockwise - 0.5) * 2 * joystickInput.autoPhiVel * joystickInput.dt
        self.phi += joystickInput.rdPhi + joystickInput.ldPhi # rotates horizontally
        self.theta += joystickInput.dTheta  # rotates vertically
        self.radius += joystickInput.dRadius

        # sphere formula https://en.wikipedia.org/wiki/Sphere
        cameraPos[0] = robotRoot[0] + self.radius * np.sin(self.theta) * np.cos(self.phi)
        cameraPos[1] = robotRoot[1] + self.radius * np.sin(self.theta) * np.sin(self.phi)
        cameraPos[2] = robotRoot[2] + self.radius * np.cos(self.theta)
        return cameraPos

class Joystick:
    """Class to use a joystick-controller in isaac gym

    Robot Steering
    --------------
        Left joystick   :   move the robot around in x-y-direction.
        Right joystick  :   move the robot around its yaw axis.

    Camera Steering
    ---------------
        L1 button       :   cinematic mode clockwise
        R1 button       :   cinematic mode counter clockwise
        L1 + R1 button  :   stop cinematic mode
        L2 button       :   manually rotate the camera clockwise
        R2 button       :   manually rotate the camera counter clockwise
        Square button   :   zoom in
        Circle button   :   zoom out
        X button        :   move camera down
        Triangle button :   move camera up

        Options Button  :   randomly switch to another robot
        Share button    :   reset camera to initial state
        
    """ 

    def __init__(self):
        self.joystickInput = JoystickInput()
        pygame.init()
        pygame.joystick.init()
        # Buffers to store toggle states
        # This allows using L1/R1 keys to toggle cinematic mode
        self.toggle_states = dict.fromkeys(["L1", "R1"], False)

        # Find all joysticks
        self.joysticks = [pygame.joystick.Joystick(x) for x in range(pygame.joystick.get_count())]
        # Use last connected joystick
        self.hasJoystick = False
        if len(self.joysticks) > 0:
            self.hasJoystick = True
            self.joysticks[-1].init() # use last connected joystick

    def has_joystick(self):
        """Returns true if a joystick is connected"""
        return self.hasJoystick

    def update(self):
        self.joystickInput.resetCamera = False
        self.joystickInput.switchRobot = False
        leftUp = False
        rightUp = False
        for event in pygame.event.get():
            # axis movement of joystick
            if event.type == pygame.JOYAXISMOTION:
                if event.axis == 1: # left joystick up / down
                    self.joystickInput.xVel = -event.value
                elif event.axis == 0: # left joystick left / right
                    self.joystickInput.yVel = -event.value
                elif event.axis == 3: # right joystick left / right
                    self.joystickInput.yawVel = -event.value
                elif event.axis == 5: # R2 button
                    self.joystickInput.rdPhi = (event.value + 1) / 2 * self.joystickInput.maxPhiVel * self.joystickInput.dt
                elif event.axis == 2: # L2 button
                    self.joystickInput.ldPhi = -(event.value + 1) / 2 * self.joystickInput.maxPhiVel * self.joystickInput.dt
            # button pressed down
            if event.type == pygame.JOYBUTTONDOWN:
                if event.button == 0: # x button
                    self.joystickInput.dTheta = self.joystickInput.maxThetaVel * self.joystickInput.dt
                elif event.button == 2: # triangle button
                    self.joystickInput.dTheta = -self.joystickInput.maxThetaVel * self.joystickInput.dt
                elif event.button == 1: # circle button
                    self.joystickInput.dRadius = self.joystickInput.maxRadiusVel * self.joystickInput.dt
                elif event.button == 3: # square button
                    self.joystickInput.dRadius = -self.joystickInput.maxRadiusVel * self.joystickInput.dt
                elif event.button == 4: # L1 button
                    # update toggle state
                    self.toggle_states["L1"] = not self.toggle_states["L1"]
                    # update joystick input
                    leftUp = True
                    self.joystickInput.counterClockwise = False
                    self.joystickInput.rotate_automatically = self.toggle_states["L1"]
                elif event.button == 5: # R1 button
                    # update toggle state
                    self.toggle_states["R1"] = not self.toggle_states["R1"]
                    rightUp = True
                    # update joystick input
                    self.joystickInput.counterClockwise = True
                    self.joystickInput.rotate_automatically = self.toggle_states["R1"]
                elif event.button == 9: # Options Button
                    self.joystickInput.switchRobot = True
                elif event.button == 8: # Share Button
                    self.joystickInput.resetCamera = True
                    self.joystickInput.rotate_automatically = False
            # button released
            if event.type == pygame.JOYBUTTONUP:
                if event.button == 0 or event.button == 2: # x and triangle button
                    self.joystickInput.dTheta = 0.0
                if event.button == 1 or event.button == 3: # x and triangle button
                    self.joystickInput.dRadius = 0.0
        if rightUp and leftUp:
            self.joystickInput.rotate_automatically = False # stop rotating
        return self.joystickInput