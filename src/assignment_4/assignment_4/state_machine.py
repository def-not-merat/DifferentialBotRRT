from .path_finder import RRT
from ament_index_python.packages import get_package_share_directory
import json
from enum import Enum
import math
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Pose, Point, Quaternion
from nav_msgs.msg import Odometry

def euler_from_quaternion(quaternion):
    """
    Converts quaternion (w in last place) to euler roll, pitch, yaw
    quaternion = [x, y, z, w]
    """
    x = quaternion.x
    y = quaternion.y
    z = quaternion.z
    w = quaternion.w

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


class FSM_STATES(Enum):
    AT_START = 'AT STart',
    HEADING_TO_TASK = 'Heading to Task',
    DO_TASK = 'Doing Task',
    RETURNING_FROM_TASK = 'Returning from Task',
    TASK_DONE = 'Task Done'

class LAWN_STATES(Enum):
    DO_ROW = 'Row',
    TURN_1 = 'turn 1',
    DO_OFFSET = 'offset',
    TURN_2 = 'turn 2',
    RETURN = 'return'

class State_Machine(Node):

    def __init__(self, my_RRT):
        super().__init__('FSM')
        self.get_logger().info(f'{self.get_name()} created')
        self._RRT = my_RRT
        self._RRT._set_logger(self.get_logger())
        self._vis_path = None

        self.max_lin_vel = 0.3
        self.max_ang_vel = 0.4
        self.lin_err = 0.1
        self.ang_err = math.pi/30

        self._subscriber = self.create_subscription(Odometry, "/odom", self._listener_callback, 1)
        self._publisher = self.create_publisher(Twist, "/cmd_vel", 1)

        self._task_x = 3.0
        self._task_y = 1.5
        self._task_theta = math.pi/3
        self._path = []
        self._cur_goal = -1

        self._row_length = 1.0 
        self._row_offset = 0.5
        self._rows_til_now = 1

        self._start_x = 0.0
        self._start_y = 0.0
        self._new_head = 0.0
        self._dir = 1

        # the blackboard
        self._cur_x = 0.0
        self._cur_y = 0.0
        self._cur_theta = 0.0
        self._lawn_states = LAWN_STATES
        self._deploy_x = 0.0
        self._deploy_y = 0.0
        self._cur_state = FSM_STATES.AT_START
        self._start_time = self.get_clock().now().nanoseconds * 1e-9

    def _shortest_turn(self, goal_yaw):
        #figuring out the shortest turn to get us facing goal
        this_dir = goal_yaw-self._cur_theta
        if this_dir>0:
            other_dir = this_dir - 2*math.pi
        else:
            other_dir = this_dir + 2*math.pi

        if abs(this_dir)>abs(other_dir):
            turn = other_dir
        else:
            turn = this_dir
        return turn

    def _drive_to_goal(self, goal_x, goal_y, goal_theta=None):
        twist = Twist()

        x_diff = goal_x - self._cur_x
        y_diff = goal_y - self._cur_y
        dist = x_diff * x_diff + y_diff * y_diff

        
        # turn to the goal
        heading = math.atan2(y_diff, x_diff)
        turn = self._shortest_turn(heading) 
        if dist > self.lin_err**2 and abs(turn) > self.ang_err:
            if turn>0:
                twist.angular.z = min(5*turn, self.max_ang_vel)
            else:
                twist.angular.z = max(5*turn, -self.max_ang_vel)
            self._publisher.publish(twist)
            return False

        
        # pointing the right direction, so go there
        if dist > self.lin_err**2:
            twist.angular.z = turn
            twist.linear.x = min(self.max_lin_vel, 5*dist)
            self._publisher.publish(twist)
            return False

        # we are there, set the correct angle
        if goal_theta is not None:
            turn = self._shortest_turn(goal_theta)
            if abs(turn) > self.ang_err:
                if turn>0:
                    twist.angular.z = min(5*turn, self.max_ang_vel)
                else:
                    twist.angular.z = max(5*turn, -self.max_ang_vel)
                self._publisher.publish(twist)
                return False
        return True


    def _lawnMower_state_machine(self, row_length, row_offset, num_rows):
        twist = Twist()

        self.get_logger().info(f'DOING TASK: {self._lawn_states}')

        if self._lawn_states == LAWN_STATES.DO_ROW:
            dist = math.sqrt((self._cur_x-self._start_x)**2 + (self._cur_y-self._start_y)**2)
            
            if abs(dist - row_length)>self.lin_err:
                twist.linear.x = min(self.max_lin_vel, 5*dist)
                self._publisher.publish(twist)
            else:
                if self._rows_til_now == num_rows:
                    self._lawn_states = LAWN_STATES.RETURN
                    return False
                self._lawn_states = LAWN_STATES.TURN_1
               
                self._new_head = self._new_head + self._dir*np.pi/2
                if self._new_head > np.pi:
                    self._new_head -=2*np.pi
                if self._new_head < -np.pi:
                    self._new_head +=2*np.pi
            return False

        elif self._lawn_states == LAWN_STATES.TURN_1:
            turn = self._shortest_turn(self._new_head)
            if abs(turn) > self.ang_err: 
                if turn > 0:
                    twist.angular.z = min(5*turn, self.max_ang_vel)
                else:
                    twist.angular.z = max(5*turn, -self.max_ang_vel)
                self._publisher.publish(twist)
                return False
            else:
                self._lawn_states = LAWN_STATES.DO_OFFSET
                self._start_x = self._cur_x
                self._start_y = self._cur_y
            return False

        elif self._lawn_states == LAWN_STATES.DO_OFFSET:
            dist = math.sqrt((self._cur_x-self._start_x)**2 + (self._cur_y-self._start_y)**2)
            
            if abs(dist - row_offset)>self.lin_err:
                twist.linear.x = min(self.max_lin_vel, 5*dist)
                self._publisher.publish(twist)
            else:
                self._lawn_states = LAWN_STATES.TURN_2
                
                self._new_head = self._new_head + self._dir*np.pi/2
                if self._new_head > np.pi:
                    self._new_head -=2*np.pi
                if self._new_head < -np.pi:
                    self._new_head +=2*np.pi
            return False
        
        elif self._lawn_states == LAWN_STATES.TURN_2:
            turn = self._shortest_turn(self._new_head)
            if abs(turn) > self.ang_err: 
                if turn > 0:
                    twist.angular.z = min(5*turn, self.max_ang_vel)
                else:
                    twist.angular.z = max(5*turn, -self.max_ang_vel)
                self._publisher.publish(twist)
                return False
            else:
                self._lawn_states = LAWN_STATES.DO_ROW
                self._rows_til_now += 1
                self._start_x = self._cur_x
                self._start_y = self._cur_y
                self._dir *= -1
            return False

        elif self._lawn_states == LAWN_STATES.RETURN:
            if self._drive_to_goal(self._task_x, self._task_y):
                return True

        return False


    def _do_state_at_start(self):
        self.get_logger().info(f'{self._cur_state}')
        now = self.get_clock().now().nanoseconds * 1e-9
        if now > (self._start_time + 1):
            self._cur_state = FSM_STATES.HEADING_TO_TASK
            self._vis_path, self._path = self._RRT.find_best_path([self._cur_x,self._cur_y],
                                                [self._task_x, self._task_y])
            self._cur_goal = 1

    def _do_state_heading_to_task(self):
        size_path = len(self._path)
        self.get_logger().info(f'{self._cur_state} : {self._cur_goal} / {size_path}')
        if self._cur_goal != size_path:
            theta = None
            if self._cur_goal==size_path-1:
                theta = self._task_theta
            goal_x, goal_y = self._path[self._cur_goal]

            if self._drive_to_goal(goal_x, goal_y, theta):
                self._cur_goal+=1
        else:
            self._cur_state = FSM_STATES.DO_TASK

            self._rows_til_now = 1
            self._start_x = self._cur_x
            self._start_y = self._cur_y
            self._lawn_states = LAWN_STATES.DO_ROW
            self._new_head = self._cur_theta
            if self._row_offset>0:
                self._dir = 1
            else: 
                self._dir = -1

    def _do_state_do_task(self):
        if self._lawnMower_state_machine(self._row_length, self._row_offset, 3):
            self._cur_state = FSM_STATES.RETURNING_FROM_TASK
            self._cur_goal = len(self._path)-2

    def _do_state_returning_from_task(self):
        self.get_logger().info(f'{self._cur_state}')
        size_path = len(self._path)
        self.get_logger().info(f'{self._cur_state} : {size_path - self._cur_goal} / {size_path}')
        if self._cur_goal != -1:
            theta = None
            if self._cur_goal==0:
                theta = self._deploy_theta
            goal_x, goal_y = self._path[self._cur_goal]

            if self._drive_to_goal(goal_x, goal_y, theta):
                self._cur_goal-=1
        else:
            self._cur_state = FSM_STATES.TASK_DONE

    def _do_state_task_done(self):
        self.get_logger().info(f'{self._cur_state}')
        twist = Twist()
        self._publisher.publish(twist)

    def _state_machine(self):
        if self._cur_state == FSM_STATES.AT_START:
            self._do_state_at_start()
            self._deploy_x = self._cur_x
            self._deploy_y = self._cur_y
            self._deploy_theta = self._cur_theta
        elif self._cur_state == FSM_STATES.HEADING_TO_TASK:
            self._do_state_heading_to_task()
        elif self._cur_state == FSM_STATES.DO_TASK:
            self._do_state_do_task()
        elif self._cur_state == FSM_STATES.RETURNING_FROM_TASK:
            self._do_state_returning_from_task()
        elif self._cur_state == FSM_STATES.TASK_DONE:
            self._do_state_task_done()
        else:
            self.get_logger().info(f'{self.get_name()} bad state {state_cur_state}')
        if self._vis_path is not None:
            cur = self._RRT.scale_point([self._cur_x, self._cur_y])
            show = cv2.circle(self._vis_path, (cur[1], cur[0]), radius=3, color=(0, 0, 255), thickness=-1)
            cv2.imshow("map", show)
            cv2.waitKey(5)

    def _listener_callback(self, msg):
        pose = msg.pose.pose

        roll, pitch, yaw = euler_from_quaternion(pose.orientation)
        self._cur_x = pose.position.x
        self._cur_y = pose.position.y
        self._cur_theta = yaw
        self._state_machine()


def main(args=None):
    map_name = "default.json"
    package_path = get_package_share_directory('assignment_4')

    try:
        with open(f"{package_path}/{map_name}") as fd:
            obs = json.load(fd)
    except Exception as e:
        node.get_logger().error(f"Unable to find/parse map in {package_path}/{map_name}")
        sys.exit(1) 

    my_RRT = RRT(1000, obs, 0.3)

    rclpy.init(args=args)
    node = State_Machine(my_RRT)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()
