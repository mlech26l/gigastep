import time

import numpy as np
import time


class JoystickInput:
    def __init__(self, pygame):
        self.pygame = pygame
        self.pygame.joystick.init()

        joysticks = [
            self.pygame.joystick.Joystick(x)
            for x in range(self.pygame.joystick.get_count())
        ]
        joy = None
        self.map_fn = self._map_default
        # Check if there is a known joystick
        for joy in joysticks:
            name = joy.get_name()
            if "Xbox" in name:
                self.map_fn = self._map_xbox
                break
            # if "Logitech" in name:
            #     self.map_fn = self._map_logitech
            #     break
        self.joy = joy

    def is_connected(self):
        return self.joy is not None

    def poll(self):
        # self.pygame.event.pump()
        action, pause, reset, quit = self.map_fn()

        return action, pause, reset, quit

    def _map_default(self):
        heading = -self.joy.get_axis(1)  # invert axis
        dive = 0.5 * (self.joy.get_axis(5) - self.joy.get_axis(2))
        thrust = self.joy.get_axis(4)
        pause = self.joy.get_button(7) > 0
        reset = self.joy.get_button(6) > 0
        quit = self.joy.get_button(0) > 0

        action = np.array([heading, dive, thrust])

        return action, pause, reset, quit

    def _map_xbox(self):
        heading = -self.joy.get_axis(1)  # invert axis
        dive = 0.5 * (self.joy.get_axis(5) - self.joy.get_axis(2))
        thrust = self.joy.get_axis(4)
        pause = self.joy.get_button(7) > 0
        reset = self.joy.get_button(6) > 0
        quit = self.joy.get_button(0) > 0

        action = np.array([heading, dive, thrust])

        return action, pause, reset, quit

    def vibrate(self, left, right, duration):
        if self.joy is not None:
            self.joy.rumble(left, right, duration)


if __name__ == "__main__":
    joy = JoystickInput()
    while True:
        print("joy", joy.get_discrete_action()[0])
        time.sleep(0.5)
    # for i in range(17):
    #     joy.vibrate(1, 1, 100)
    # time.sleep(1)
    # breakpoint()
    # print("joy._error", joy._error)
    # joy.stop()
    # time.sleep(500)
    # breakpoint()
    # while joy._error is None:
    #     time.sleep(1)
    # breakpoint()
    while True:
        # print(joy.read())
        # pass
        print(joy)
    #     # events = get_gamepad()
    #     state = inputs.devices.gamepads[0].read()
    #     print(state)
    #     # for e in events:
    #     #     print("code: ",e.code,", state: ",e.state)