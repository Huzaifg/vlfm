from dataclasses import dataclass
from typing import Any, Dict, Union, Tuple
import numpy as np
import torch
from torch import Tensor

# Import helper functions and mapping modules.
from depth_camera_filtering import filter_depth
from vlfm.utils.geometry_utils import xyz_yaw_to_tf_matrix
from vlfm.mapping.obstacle_map import ObstacleMap
from vlfm.vlm.grounding_dino import ObjectDetections

# Import base policies (assumed to be available in the package structure)
from .base_objectnav_policy import BaseObjectNavPolicy, VLFMConfig
from .itm_policy import ITMPolicy, ITMPolicyV2, ITMPolicyV3

# Define a simple enumeration for action IDs.
class TorchActionIDs:
    STOP = torch.tensor([[0]], dtype=torch.long)
    MOVE_FORWARD = torch.tensor([[1]], dtype=torch.long)
    TURN_LEFT = torch.tensor([[2]], dtype=torch.long)
    TURN_RIGHT = torch.tensor([[3]], dtype=torch.long)
    GOAL = torch.tensor([[4]], dtype=torch.float32)

class ChronoMixin(BaseObjectNavPolicy):
    """
    Mixin for running VLFM in a Chrono environment.
    Expects a shared mapping object to be provided.
    """
    _stop_action: Tensor = TorchActionIDs.STOP
    _start_yaw: Union[float, None] = None  # Must be set in reset
    _observations_cache: Dict[str, Any] = {}
    _policy_info: Dict[str, Any] = {}
    _compute_frontiers: bool = True
    _visualize: bool = True

    def __init__(self, camera_height: float, min_depth: float, max_depth: float,
                 camera_fov: float, image_width: int, shared_map: ObstacleMap, *args: Any, **kwargs: Any) -> None:
        # Pass any extra arguments to the parent.
        super().__init__(*args, **kwargs)
        self.shared_map = shared_map  # Shared obstacle and value map instance
        self._camera_height = camera_height
        self._min_depth = min_depth
        self._max_depth = max_depth
        # Convert provided field of view to radians.
        self._camera_fov = np.deg2rad(camera_fov)
        self._image_width = image_width
        # Calculate focal lengths (fx and fy) from camera intrinsics.
        camera_fov_rad = np.radians(camera_fov)
        self._fx = self._fy = image_width / (2 * np.tan(camera_fov_rad / 2))

    def act(self, observations, rnn_hidden_states: Any, prev_actions: Any, masks: Tensor,
            deterministic: bool = False) -> Tuple[Tensor, Any]:
        parent_cls: BaseObjectNavPolicy = super()
        try:
            action, rnn_hidden_states = parent_cls.act(
                observations, rnn_hidden_states, prev_actions, masks, deterministic)
        except StopIteration:
            action = self._stop_action
        return action, rnn_hidden_states

    def _initialize(self) -> Tensor:
        """Spin (turn left) repeatedly at reset for an initial 360° scan."""
        # Turning left 30° 12 times gives a full 360° rotation.
        self._done_initializing = not self._num_steps < 24  # type: ignore
        return TorchActionIDs.TURN_LEFT

    def _reset(self) -> None:
        parent_cls: BaseObjectNavPolicy = super()  # type: ignore
        parent_cls._reset()
        self._start_yaw = None

    def _get_policy_info(self, detections: ObjectDetections) -> Dict[str, Any]:
        """Get policy info for logging."""
        parent_cls: BaseObjectNavPolicy = super()  # type: ignore
        info = parent_cls._get_policy_info(detections)
        if not self._visualize:
            return info
        if self._start_yaw is None:
            self._start_yaw = self._observations_cache.get("habitat_start_yaw", 0)
        info["start_yaw"] = self._start_yaw
        return info

    def _cache_observations(self, observations) -> None:
        """
        Cache observations from the environment. Updates the shared
        obstacle map using the agent's RGB-D data.
        """
        if len(self._observations_cache) > 0:
            return

        rgb = observations["rgb"].cpu().numpy()
        depth = observations["depth"].cpu().numpy()
        x, y = observations["gps"].cpu().numpy()
        camera_yaw = observations["compass"].cpu().item()

        # Optionally filter or pre-process the depth map.
        depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)

        camera_position = np.array([x, y, self._camera_height])
        tf_camera_to_episodic = xyz_yaw_to_tf_matrix(camera_position, camera_yaw)

        # Update the shared obstacle map based on the new depth observation.
        self.shared_map.update_map(
            depth,
            tf_camera_to_episodic,
            self._min_depth,
            self._max_depth,
            self._fx,
            self._fy,
            self._camera_fov,
        )
        # Also update agent trajectory on the map.
        robot_xy = camera_position[:2]
        self.shared_map.update_agent_traj(robot_xy, camera_yaw)

        self._observations_cache = {
            "frontier_sensor": self.shared_map.frontiers,
            "nav_depth": observations["depth"],
            "robot_xy": robot_xy,
            "robot_heading": camera_yaw,
            "object_map_rgbd": [
                (rgb, depth, tf_camera_to_episodic, self._min_depth, self._max_depth, self._fx, self._fy)
            ],
            "value_map_rgbd": [
                (rgb, depth, tf_camera_to_episodic, self._min_depth, self._max_depth, self._camera_fov)
            ],
            "habitat_start_yaw": observations["compass"].item(),
        }

# Define multi-agent policy classes by mixing in ChronoMixin with your ITM variants.
class ChronoITMPolicy(ChronoMixin, ITMPolicy):
    pass

class ChronoITMPolicyV2(ChronoMixin, ITMPolicyV2):
    pass

class ChronoITMPolicyV3(ChronoMixin, ITMPolicyV3):
    pass

# Optionally, if using dataclass-based configuration, you might add a VLFMPolicyConfig here.
# @dataclass
# class VLFMPolicyConfig(VLFMConfig, PolicyConfig):
#     pass
