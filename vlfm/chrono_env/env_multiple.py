import time
import math
import numpy as np
import torch
import sys
import os

import pychrono as chrono
import pychrono.sensor as sens
import pychrono.irrlicht as chronoirr

# Set up project paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(project_root)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

TARGET_OBJECT = "toilet"

class ChronoEnvMulti:
    def __init__(self, target_object: str = TARGET_OBJECT, num_agents: int = 1):
        # Allow choosing the number of agents for debugging.
        self.num_agents = num_agents

        # Output directory for sensor outputs (if needed)
        self.out_dir = "SENSOR_OUTPUT/"
        self.lens_model = sens.PINHOLE

        # Simulation settings
        self.update_rate = 30  # Hz
        self.image_width = 640
        self.image_height = 480
        self.fov = 1.408  # horizontal field of view in radians
        self.lag = 0
        self.exposure_time = 0

        # Physics simulation timestep & control frequency
        self.timestep = 0.001
        self.control_frequency = 10
        self.steps_per_control = round(1 / (self.timestep * self.control_frequency))
        self.step_number = 0
        self.render_frame = 0

        self.target_object = target_object

        # Holders for multiple agents and their sensors
        self.virtual_robots = []   # list of chrono bodies for agents
        self.lidar_list = []       # lidar sensors for each agent
        self.cam_list = []         # camera sensors for each agent

        self.manager = None  # A single sensor manager for all agents
        self.vis = None      # Shared visual system for the simulation
        self.observations = None

    def reset(self):
        # Create the Chrono simulation system.
        self.my_system = chrono.ChSystemSMC()
        self.my_system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)

        patch_mat = chrono.ChContactMaterialSMC()

        # Instead of hard-coding two positions, we choose based on self.num_agents.
        # For testing with one agent, you might choose a single fixed starting point.
        if self.num_agents == 1:
            start_positions = [chrono.ChVector3d(-1.25, -1.25, 0.25)]
        else:
            # If more than one agent is desired, use some predefined positions.
            start_positions = [
                chrono.ChVector3d(-1.25, -1.25, 0.25),
                chrono.ChVector3d(1.25, 1.25, 0.25),
            ]
            # You could extend this list if needed.

        # Create a single sensor manager for all sensors.
        self.manager = sens.ChSensorManager(self.my_system)
        
        for pos in start_positions:
            robot = chrono.ChBodyEasyBox(0.25, 0.25, 0.5, 100, True, True, patch_mat)
            robot.SetPos(pos)
            robot.SetFixed(True)
            self.my_system.Add(robot)
            self.virtual_robots.append(robot)

            # Define a common sensor offset.
            offset_pose = chrono.ChFramed(chrono.ChVector3d(0.3, 0, 0.25), chrono.QUNIT)

            # Create and add a lidar sensor.
            lidar = sens.ChLidarSensor(
                robot,              # attach to this robot
                30,                 # scanning rate in Hz
                offset_pose,
                self.image_width,   # horizontal samples
                self.image_height,  # vertical channels
                self.fov,           # horizontal FOV
                chrono.CH_PI/6,     # vertical FOV upper bound
                -chrono.CH_PI/6,    # vertical FOV lower bound
                3.66,               # maximum lidar range
                sens.LidarBeamShape_RECTANGULAR,
                1,                  # sample radius
                0, 0,               # divergence angles
                sens.LidarReturnMode_STRONGEST_RETURN
            )
            lidar.SetName("Lidar Sensor")
            lidar.SetLag(0)
            lidar.SetCollectionWindow(1/20)
            lidar.PushFilter(sens.ChFilterVisualize(self.image_width, self.image_height, "depth camera"))
            lidar.PushFilter(sens.ChFilterDIAccess())
            self.manager.AddSensor(lidar)
            self.lidar_list.append(lidar)

            # Create and add a camera sensor.
            cam = sens.ChCameraSensor(
                robot,
                self.update_rate,
                offset_pose,
                self.image_width,
                self.image_height,
                self.fov
            )
            cam.SetName("Camera Sensor")
            cam.SetLag(self.lag)
            cam.SetCollectionWindow(self.exposure_time)
            cam.PushFilter(sens.ChFilterVisualize(self.image_width, self.image_height, "rgb camera"))
            cam.PushFilter(sens.ChFilterRGBA8Access())
            self.manager.AddSensor(cam)
            self.cam_list.append(cam)

        # Load the static environment mesh.
        mmesh = chrono.ChTriangleMeshConnected()
        mesh_path = os.path.join(project_root, "data/chrono_environment/new_flat_3.obj")
        mmesh.LoadWavefrontMesh(mesh_path, False, True)

        trimesh_shape = chrono.ChVisualShapeTriangleMesh()
        trimesh_shape.SetMesh(mmesh)
        trimesh_shape.SetName("ENV MESH")
        trimesh_shape.SetMutable(False)
        mesh_body = chrono.ChBody()
        mesh_body.SetPos(chrono.ChVector3d(0, 0, 0))
        mesh_body.SetRot(chrono.Q_ROTATE_Y_TO_Z)
        mesh_body.AddVisualShape(trimesh_shape)
        mesh_body.SetFixed(True)
        self.my_system.Add(mesh_body)

        # Initialize the visualization system.
        self.vis = chronoirr.ChVisualSystemIrrlicht(self.my_system)
        self.vis.SetCameraVertical(chrono.CameraVerticalDir_Z)
        self.vis.AddLightWithShadow(chrono.ChVector3d(2, 2, 2),
                                    chrono.ChVector3d(0, 0, 0),
                                    5, 1, 11, 55)
        self.vis.EnableAbsCoordsysDrawing(True)
        self.vis.Initialize()
        self.vis.AddSkyBox()
        self.vis.AddCamera(chrono.ChVector3d(-7/3, 0, 4.5/3),
                           chrono.ChVector3d(0, 0, 0))

        # Get initial observations from all agents.
        self.observations = [self._get_observations(i) for i in range(len(self.virtual_robots))]
        return self.observations

    def step(self, actions):
        for i, action in enumerate(actions):
            self._do_action(action, self.virtual_robots[i])
        for _ in range(self.steps_per_control):
            self.manager.Update()
            self.my_system.DoStepDynamics(self.timestep)

        self.vis.BeginScene()
        self.vis.Render()
        self.vis.EndScene()

        new_observations = [self._get_observations(i) for i in range(len(self.virtual_robots))]
        stop = any(self._get_stop(action) for action in actions)
        return new_observations, stop

    def _get_stop(self, action):
        if isinstance(action, torch.Tensor) and action.numel() == 1 and action.item() == 0:
            print("STOPPED")
            time.sleep(5)
            return True
        return False

    def _get_observations(self, idx):
        lidar = self.lidar_list[idx]
        cam = self.cam_list[idx]
        robot = self.virtual_robots[idx]

        depth_buffer = lidar.GetMostRecentDIBuffer()
        if depth_buffer.HasData():
            depth_data = torch.tensor(depth_buffer.GetDIData()[:, :, 0], dtype=torch.float32)
            depth_data = torch.flip(depth_data, dims=[0, 1])
            MIN_DEPTH = 0
            MAX_DEPTH = 5.5
            depth_data = np.clip((depth_data - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH), 0, 1)
            depth_data[depth_data == 0] = 1
        else:
            depth_data = torch.zeros(self.image_height, self.image_width, dtype=torch.float32)

        camera_buffer = cam.GetMostRecentRGBA8Buffer()
        if camera_buffer.HasData():
            camera_data = torch.tensor(camera_buffer.GetRGBA8Data(), dtype=torch.uint8)[:, :, :3]
            camera_data = torch.flip(camera_data, dims=[0])
        else:
            camera_data = torch.zeros(self.image_height, self.image_width, 3, dtype=torch.uint8)

        robot_x = torch.tensor(robot.GetPos().x, dtype=torch.float32)
        robot_y = torch.tensor(robot.GetPos().y, dtype=torch.float32)
        quat_list = [robot.GetRot().e0, robot.GetRot().e1, robot.GetRot().e2, robot.GetRot().e3]
        yaw = self.quaternion_to_yaw(quat_list)
        robot_yaw = torch.tensor(yaw, dtype=torch.float32)

        obs_dict = {
            "rgb": camera_data,
            "depth": depth_data,
            "gps": torch.stack((robot_x, robot_y)),
            "compass": robot_yaw,
            "objectgoal": self.target_object
        }
        return obs_dict

    def _do_action(self, action, robot):
        if len(action[0]) > 1:
            target_x = float(action[0][0])
            target_y = float(action[0][1])
            cur_pos = robot.GetPos()
            heading = math.atan2(target_y - cur_pos.y, target_x - cur_pos.x)
            heading = (heading + math.pi) % (2 * math.pi) - math.pi
            current_heading = robot.GetRot().GetCardanAnglesXYZ().z
            turn_angle = heading - current_heading
            turn_angle = (turn_angle + math.pi) % (2 * math.pi) - math.pi
            print("TURN ANGLE: ", turn_angle)
            new_rotation = chrono.QuatFromAngleZ(turn_angle) * robot.GetRot()
            robot.SetRot(new_rotation)
            robot.SetPos(chrono.ChVector3d(target_x, target_y, 0.25))
            print("Moved Pos: ", robot.GetPos())
        else:
            action_id = action.item()
            if action_id == 1:  # MOVE_FORWARD
                rot_state = robot.GetRot().GetCardanAnglesXYZ()
                robot.SetPos(robot.GetPos() + chrono.ChVector3d(
                    0.01 * np.cos(rot_state.z),
                    0.01 * np.sin(rot_state.z),
                    0))
            elif action_id == 2:  # TURN_LEFT
                robot.SetRot(chrono.QuatFromAngleZ(np.pi/12) * robot.GetRot())
            elif action_id == 3:  # TURN_RIGHT
                robot.SetRot(chrono.QuatFromAngleZ(-np.pi/12) * robot.GetRot())

    def quaternion_to_yaw(self, quaternion):
        w, x, y, z = quaternion
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
        return yaw

if __name__ == "__main__":
    # For debugging the shared obstacle map, we will run with one agent.
    # Import the shared mapping class and policy.
    from vlfm.mapping.obstacle_map import ObstacleMap
    from vlfm.policy.chrono_policies_multiple import ChronoITMPolicyV2

    # Define parameters as in your single-agent implementation.
    min_obstacle_height = 0.61
    max_obstacle_height = 0.88
    agent_radius = 0.18
    obstacle_map_area_threshold = 1.5
    hole_area_thresh = 100000

    # Create the shared obstacle map with all required parameters.
    shared_map = ObstacleMap(
        min_height=min_obstacle_height,
        max_height=max_obstacle_height,
        area_thresh=obstacle_map_area_threshold,
        agent_radius=agent_radius,
        hole_area_thresh=hole_area_thresh,
    )

    # Instantiate a single policy instance.
    camera_height = 0.5
    min_depth = 0
    max_depth = 5.5
    camera_fov = 80.67  # in degrees
    image_width = 640
    text_prompt = "Seems like there is a target_object ahead."
    use_max_confidence = False
    pointnav_policy_path = "data/pointnav_weights.pth"
    depth_image_shape = (480, 640)
    pointnav_stop_radius = 0.5
    object_map_erosion_size = 5

    policy = ChronoITMPolicyV2(
        camera_height=camera_height,
        min_depth=min_depth,
        max_depth=max_depth,
        camera_fov=camera_fov,
        image_width=image_width,
        shared_map=shared_map,
        text_prompt=text_prompt,
        use_max_confidence=use_max_confidence,
        pointnav_policy_path=pointnav_policy_path,
        depth_image_shape=depth_image_shape,
        pointnav_stop_radius=pointnav_stop_radius,
        object_map_erosion_size=object_map_erosion_size,
        obstacle_map_area_threshold=obstacle_map_area_threshold,
        min_obstacle_height=min_obstacle_height,
        max_obstacle_height=max_obstacle_height,
        hole_area_thresh=hole_area_thresh,
        use_vqa=False,
        vqa_prompt="Is this ",
        coco_threshold=0.8,
        non_coco_threshold=0.4,
        agent_radius=agent_radius
    )

    # Create the Chrono environment with one agent by passing num_agents=1.
    env = ChronoEnvMulti(target_object="toilet", num_agents=1)
    observations = env.reset()

    # Optionally, initialize a dummy mask.
    masks = torch.zeros(1, 1)
    
    # Simulation loop.
    end_time = 10  # seconds
    control_timestep = 0.1
    time_count = 0
    stop = False

    while time_count < end_time and not stop:
        # Get the agent's action from its policy.
        action, _ = policy.act(observations[0], None, None, masks)
        actions = [action]

        masks = torch.ones(1, 1)
        observations, stop = env.step(actions)
        if stop:
            break

        # For debugging, print a summary of the observation:
        print(f"Step {time_count:.2f}: GPS = {observations[0]['gps']}  | Compass = {observations[0]['compass']}")

        # Optionally, visualize the RGB and depth images.
        import matplotlib.pyplot as plt
        annotated_depth = observations[0]["depth"].numpy()
        rgb_image = observations[0]["rgb"].numpy()

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.title("Depth")
        plt.imshow(annotated_depth, cmap='gray')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.title("RGB")
        plt.imshow(rgb_image)
        plt.axis('off')
        plt.tight_layout()
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        plt.savefig(f"tmp_vis_2/single_agent_policy_info_{timestamp}.png")
        plt.close()

        time_count += control_timestep

    # Optionally, you can output the visualized obstacle map:
    # obstacle_vis = shared_map.visualize()
    # cv2.imwrite("tmp_vis_2/shared_obstacle_map.png", obstacle_vis)
