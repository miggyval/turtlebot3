#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# TurtleBot3 Joystick Teleop (pygame)
# - Publishes geometry_msgs/Twist (Humble) or TwistStamped (other)
# - Xbox controller via pygame
#
# Based on TurtleBot3 teleop_keyboard (BSD-2-Clause) and user-provided joystick logic.

import os
import sys
import select
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.qos import QoSProfile

from geometry_msgs.msg import Twist, TwistStamped
from std_msgs.msg import Bool

import pygame
import numpy as np

# -------- Controller mapping (adjust if needed) --------
# Axes (Xbox on macOS via pygame typically):
#   0: LS X, 1: LS Y, 2: RS X, 3: RS Y, 4: LT, 5: RT   (ranges [-1,1])
FORWARD_AXIS = 1        # Left stick vertical (up is negative in many drivers)
STEER_AXIS   = 2        # Right stick horizontal
LEFT_TRIGGER = 4        # LT axis
RIGHT_TRIGGER = 5       # RT axis

# Buttons (these indices can vary; using your previous values)
LEFT_BUMPER  = 9
RIGHT_BUMPER = 10
BUTTON_X     = 3        # Press to publish /reset_sim=True (optional)

# -------- Speed / shaping --------
LIN_VEL_STEP = 0.01     # only used in printed hints if you add keys later
ANG_VEL_STEP = 0.1

REVERSE_RATIO = 0.5     # reverse max speed fraction vs forward
SLOW_MULT = 0.4         # hold LB for precise moves
TURBO_MULT = 1.2        # hold RB for a small boost (capped at model max)
DEADZONE = 0.05         # ignore tiny stick noise
CURVE_EXP = 4           # your sign(x)*|x|^4 shaping

# Low-pass filter (smoothing)
CTRL_RATE_HZ = 50.0
TAU = 0.25              # seconds, smaller = snappier

class TB3JoystickTeleop(Node):
    def __init__(self):
        super().__init__('teleop_joystick')

        qos = QoSProfile(depth=10)
        self.ros_distro = os.environ.get('ROS_DISTRO', 'humble').lower()
        self.cmd_topic = self.declare_parameter('cmd_vel_topic', 'cmd_vel').get_parameter_value().string_value

        # Pick Twist vs TwistStamped like the TB3 keyboard node did
        if self.ros_distro == 'humble':
            self.pub = self.create_publisher(Twist, self.cmd_topic, qos)
        else:
            self.pub = self.create_publisher(TwistStamped, self.cmd_topic, qos)

        # Optional reset pulse (kept from your node)
        self.pub_reset = self.create_publisher(Bool, '/reset_sim', 1)

        # Model limits from environment
        model = os.environ.get('TURTLEBOT3_MODEL', 'burger').lower()
        if model == 'burger':
            self.max_lin = 0.22
            self.max_ang = 2.84
        else:
            # waffle / waffle_pi
            self.max_lin = 0.26
            self.max_ang = 1.82

        # Init pygame/joystick
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            self.get_logger().error('No joystick detected by pygame.')
            raise RuntimeError('No joystick found')
        self.js = pygame.joystick.Joystick(0)
        self.js.init()
        self.get_logger().info(f'Using joystick: {self.js.get_name()} '
                               f'({self.js.get_numaxes()} axes, {self.js.get_numbuttons()} buttons)')

        # State
        self.v_target = 0.0
        self.w_target = 0.0
        self.v_ctrl = 0.0
        self.w_ctrl = 0.0

        self.dt = 1.0 / CTRL_RATE_HZ
        self.timer = self.create_timer(self.dt, self._on_timer)

    # ---- helpers ----
    def _shape_axis(self, val: float) -> float:
        # deadzone + sign(x)*|x|^CURVE_EXP
        if abs(val) < DEADZONE:
            return 0.0
        return math.copysign(abs(val) ** CURVE_EXP, val)

    def _axis(self, idx: int) -> float:
        try:
            return float(self.js.get_axis(idx))
        except Exception:
            return 0.0

    def _button(self, idx: int) -> bool:
        try:
            return bool(self.js.get_button(idx))
        except Exception:
            return False

    def _trig01(self, raw: float) -> float:
        # Convert trigger from [-1,1] to [0,1]
        return (raw + 1.0) / 2.0

    def _publish_reset_if_needed(self):
        if self._button(BUTTON_X):
            msg = Bool()
            msg.data = True
            self.pub_reset.publish(msg)

    def _publish_cmd(self, v: float, w: float):
        v = float(np.clip(v, -self.max_lin, self.max_lin))
        w = float(np.clip(w, -self.max_ang, self.max_ang))

        if self.ros_distro == 'humble':
            msg = Twist()
            msg.linear.x = v
            msg.angular.z = w
            self.pub.publish(msg)
        else:
            msg = TwistStamped()
            msg.header.stamp = Clock().now().to_msg()
            msg.header.frame_id = ''
            msg.twist.linear.x = v
            msg.twist.angular.z = w
            self.pub.publish(msg)

    # ---- main loop ----
    def _on_timer(self):
        # Keep pygame events moving
        pygame.event.pump()

        # Read sticks
        raw_forward = -self._axis(FORWARD_AXIS)   # up is negative on many drivers
        raw_steer   =  self._axis(STEER_AXIS)

        throttle = self._shape_axis(raw_forward)
        steering = self._shape_axis(raw_steer)

        # Combine so strong steering reduces forward, like your scale idea
        weight_v = 0.7
        weight_w = 2.0
        scale = max(1.0, weight_v * abs(throttle) + weight_w * abs(steering))

        # Base max lin/ang; reduce reverse top speed
        max_lin_eff = self.max_lin if throttle >= 0.0 else self.max_lin * REVERSE_RATIO
        v_cmd = (max_lin_eff * throttle) / scale
        w_cmd = (self.max_ang * steering) / scale

        # Triggers: push to do in-place rotation (fine steering); zero linear while using triggers
        lt = self._trig01(self._axis(LEFT_TRIGGER))
        rt = self._trig01(self._axis(RIGHT_TRIGGER))
        if lt > 0.1 or rt > 0.1:
            v_cmd = 0.0
            w_cmd = (rt - lt) * self.max_ang  # RT: +z, LT: -z

        # Bumpers = speed modifiers (hold)
        if self._button(LEFT_BUMPER):
            v_cmd *= SLOW_MULT
            w_cmd *= SLOW_MULT
        if self._button(RIGHT_BUMPER):
            v_cmd *= TURBO_MULT
            w_cmd *= TURBO_MULT

        # Optional reset on X
        self._publish_reset_if_needed()

        # First-order smoothing (like your tau filter)
        alpha = self.dt / max(TAU, 1e-6)
        self.v_ctrl += (v_cmd - self.v_ctrl) * alpha
        self.w_ctrl += (w_cmd - self.w_ctrl) * alpha

        self._publish_cmd(self.v_ctrl, self.w_ctrl)

def main():
    rclpy.init()
    node = None
    try:
        node = TB3JoystickTeleop()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(e, file=sys.stderr)
    finally:
        if node is not None:
            # stop the robot before shutdown
            try:
                if os.environ.get('ROS_DISTRO', 'humble').lower() == 'humble':
                    stop = Twist()
                else:
                    stop = TwistStamped()
                    stop.header.stamp = Clock().now().to_msg()
                    stop.twist = Twist()
                node.pub.publish(stop)
            except Exception:
                pass
            node.destroy_node()
        rclpy.shutdown()
        try:
            pygame.joystick.quit()
            pygame.quit()
        except Exception:
            pass

if __name__ == '__main__':
    main()
