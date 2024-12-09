import math
import numpy as np
import pychrono.sensor as sens
import pychrono.vehicle as veh
import pychrono.irrlicht as chronoirr
import pychrono as chrono
import sys
import os
import torch
# Assuming the script is located in the 'experiments/apartment' directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(project_root)
# Add the parent directory of 'models' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import vlfm.policy.chrono_policies

class ChronoEnv:
    def __init__(self, target_object: str = "tv"):
        self.my_system = None

        # Output directory
        self.out_dir = "SENSOR_OUTPUT/"

        # Camera lens model
        self.lens_model = sens.PINHOLE

        # Update rate in Hz
        self.update_rate = 30

        # Image width and height
        self.image_width = 640
        self.image_height = 480

        # Camera's horizontal field of view
        self.fov = 1.408

        # Lag (in seconds) between sensing and when data becomes accessible
        self.lag = 0

        # Exposure (in seconds) of each image
        self.exposure_time = 0

        self.manager = None

        self.lidar = None
        self.cam = None

        self.vis = None
        self.rt_timer = None
        self.timestep = 0.001
        self.control_frequency = 10  # Control frequency of the simulation
        self.steps_per_control = round(
            1 / (self.timestep * self.control_frequency))
        self.step_number = 0
        self.render_frame = 0
        self.observations = None
        self.target_object = target_object  # Target object

    def reset(self):
        self.my_system = chrono.ChSystemSMC()
        self.my_system.SetCollisionSystemType(
            chrono.ChCollisionSystem.Type_BULLET)

        patch_mat = chrono.ChContactMaterialSMC()
        # patch_mat.SetFriction(0.1)
        # patch_mat.SetRollingFriction(0.001)
        # terrain = veh.RigidTerrain(my_system)
        # patch = terrain.AddPatch(patch_mat,
        #     chrono.ChCoordsysd(chrono.ChVector3d(0, -0.09, 0), chrono.Q_ROTATE_Z_TO_Y),
        #     100, 100)
        # patch.SetColor(chrono.ChColor(0.0, 0.0, 0.0))
        # terrain.Initialize()

        mmesh = chrono.ChTriangleMeshConnected()
        mmesh.LoadWavefrontMesh(
            project_root + '/data/chrono_environment/flat_env.obj', False, True)

        # scale to a different size
        # mmesh.Transform(chrono.ChVector3d(0, 0, 0), chrono.ChMatrix33d(2))

        trimesh_shape = chrono.ChVisualShapeTriangleMesh()
        trimesh_shape.SetMesh(mmesh)
        trimesh_shape.SetName("ENV MESH")
        trimesh_shape.SetMutable(False)

        mesh_body = chrono.ChBody()
        mesh_body.SetPos(chrono.ChVector3d(0, 0, 0))
        mesh_body.AddVisualShape(trimesh_shape)
        mesh_body.SetFixed(True)
        self.my_system.Add(mesh_body)

        # ---------------------------------------

        # Add camera sensor

        # -----------------
        # Camera parameters
        # -----------------

        self.manager = sens.ChSensorManager(self.my_system)

        intensity = 1.0
        self.manager.scene.AddAreaLight(chrono.ChVector3f(0, 0, 4), chrono.ChColor(
            intensity, intensity, intensity), 500.0, chrono.ChVector3f(1, 0, 0), chrono.ChVector3f(0, -1, 0))

        rotation_1 = chrono.QuatFromAngleAxis(
            np.pi/2, chrono.ChVector3d(0, 0, 1))
        rotation_2 = chrono.QuatFromAngleAxis(
            np.pi/2, chrono.ChVector3d(0, 1, 0))
        rotation_3 = chrono.QuatFromAngleAxis(
            np.pi/2, chrono.ChVector3d(0, 0, 1))

        rotation_quat = rotation_1 * rotation_2 * rotation_3

        offset_pose = chrono.ChFramed(
            chrono.ChVector3d(0, 1, 0), rotation_quat)

        self.lidar = sens.ChLidarSensor(
            mesh_body,             # body lidar is attached to
            20,                     # scanning rate in Hz
            offset_pose,            # offset pose
            self.image_width,                   # number of horizontal samples
            self.image_height,                    # number of vertical channels
            1.6,                    # horizontal field of view
            chrono.CH_PI/6,         # vertical field of view
            -chrono.CH_PI/6,
            5.0,                  # max lidar range
            sens.LidarBeamShape_RECTANGULAR,
            1,          # sample radius
            0,       # divergence angle
            0,       # divergence angle
            sens.LidarReturnMode_STRONGEST_RETURN)

        self.lidar.SetName("Lidar Sensor")
        self.lidar.SetLag(0)
        self.lidar.SetCollectionWindow(1/20)
        self.lidar.PushFilter(sens.ChFilterVisualize(
            self.image_width, self.image_height, "depth camera"))
        self.lidar.PushFilter(sens.ChFilterDIAccess())
        self.manager.AddSensor(self.lidar)

        self.cam = sens.ChCameraSensor(
            mesh_body,              # body camera is attached to
            self.update_rate,            # update rate in Hz
            offset_pose,            # offset pose
            self.image_width,            # image width
            self.image_height,           # image height
            self.fov                    # camera's horizontal field of view
        )

        self.cam.SetName("Camera Sensor")
        self.cam.SetLag(self.lag)
        self.cam.SetCollectionWindow(self.exposure_time)
        self.cam.PushFilter(sens.ChFilterVisualize(
            self.image_width, self.image_height, "rgb camera"))
        # Provides the host access to this RGBA8 buffer
        self.cam.PushFilter(sens.ChFilterRGBA8Access())

        # add sensor to manager
        self.manager.AddSensor(self.cam)

        # ---------------------------------------

        # Create visualization for the gripper fingers
        self.vis = chronoirr.ChVisualSystemIrrlicht(
            self.my_system, chrono.ChVector3d(-2, 1, -1))
        self.vis.SetCameraVertical(chrono.CameraVerticalDir_Z)
        self.vis.AddLightWithShadow(chrono.ChVector3d(6, 6, 6),  # point
                                    chrono.ChVector3d(0, 6, 0),  # aimpoint
                                    5,                       # radius (power)
                                    1, 11,                     # near, far
                                    55)                       # angle of FOV

        # vis.EnableShadows()
        self.vis.EnableCollisionShapeDrawing(True)

        self.observations = self._get_observations()

        return self.observations

    def step(self, action):
        for i in range(0, self.steps_per_control):
            self.manager.Update()
            sim_time = self.my_system.GetChTime()
            self.my_system.DoStepDynamics(self.timestep)

        self.vis.BeginScene()
        self.vis.Render()
        self.vis.EndScene()

        self.observations = self._get_observations()

        self.stop = self._get_stop()
        return self.observations, self.stop

    def _get_stop(self):
        return False

    def _get_observations(self):
        depth_buffer = self.lidar.GetMostRecentDIBuffer()
        if depth_buffer.HasData():
            depth_data = depth_buffer.GetDIData()
            # Removes the 2nd column which is intensity
            depth_data = torch.tensor(depth_data[:, :, 0], dtype=torch.float32)
            print('shape of depth data: ', depth_data.shape)
        else:
            depth_data = torch.zeros(
                self.image_height, self.image_width, dtype=torch.float32)

        camera_buffer = self.cam.GetMostRecentRGBA8Buffer()
        if camera_buffer.HasData():
            camera_data = camera_buffer.GetRGBA8Data()
            camera_data = torch.tensor(camera_data, dtype=torch.uint8)
            # Remove the 4th column which is transparency
            camera_data = camera_data[:, :, :3]
            print('shape of camera data: ', camera_data.shape)
        else:
            camera_data = torch.zeros(
                self.image_height, self.image_width, 3, dtype=torch.uint8)

        robot_x = torch.tensor(0, dtype=torch.float32)
        robot_y = torch.tensor(0, dtype=torch.float32)
        robot_yaw = torch.tensor(0, dtype=torch.float32)

        obs_dict = {
            "rgb": camera_data,
            "depth": depth_data,
            "gps": torch.stack((robot_x, robot_y)),
            "compass": robot_yaw,
            "objectgoal": self.target_object  # Target object
        }

        return obs_dict


if __name__ == "__main__":
    env = ChronoEnv()
    env.reset()




    ### VLFM

    # sensor params
    camera_height = 0.88
    min_depth = 0.5
    max_depth = 5.0
    camera_fov = 79
    image_width = 640

    # kwargs for itm policy
    # name = "ChronoITMPolicy"
    text_prompt = "Seems like there is a target_object ahead."
    use_max_confidence = False
    pointnav_policy_path = "data/pointnav_weights.pth"
    depth_image_shape = (224, 224)
    pointnav_stop_radius = 0.9
    object_map_erosion_size = 5
    exploration_thresh = 0.0
    obstacle_map_area_threshold = 1.5  # in square meters
    min_obstacle_height = 0.61
    max_obstacle_height = 0.88
    hole_area_thresh = 100000
    use_vqa = False
    vqa_prompt = "Is this "
    coco_threshold = 0.8
    non_coco_threshold = 0.4
    agent_radius = 0.18


    vlfm_policy = vlfm.policy.chrono_policies.ChronoITMPolicy(
        camera_height=camera_height,
        min_depth=min_depth,
        max_depth=max_depth,
        camera_fov=camera_fov,
        image_width=image_width,
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
        use_vqa=use_vqa,
        vqa_prompt=vqa_prompt,
        coco_threshold=coco_threshold,
        non_coco_threshold=non_coco_threshold,
        agent_radius=agent_radius
    )

    end_time = 10
    control_timestep = 0.01
    time = 0
    while time < end_time:
        obs, stop = env.step(0)
        print(obs)

        time += control_timestep
