from dataclasses import dataclass
from typing import Any, Dict, Union

import numpy as np
import torch
from depth_camera_filtering import filter_depth
from frontier_exploration.base_explorer import BaseExplorer
from vlfm.utils.geometry_utils import closest_point_within_threshold
from torch import Tensor

from vlfm.utils.geometry_utils import xyz_yaw_to_tf_matrix
from vlfm.vlm.grounding_dino import ObjectDetections

from ..mapping.obstacle_map import ObstacleMap
from .base_objectnav_policy import BaseObjectNavPolicy, VLFMConfig
from .itm_policy import ITMPolicy, ITMPolicyV2, ITMPolicyV3
import os
from typing import Any, Dict, Union, Tuple


class TorchActionIDs:
    STOP = torch.tensor([[0]], dtype=torch.long)
    MOVE_FORWARD = torch.tensor([[1]], dtype=torch.long)
    TURN_LEFT = torch.tensor([[2]], dtype=torch.long)
    TURN_RIGHT = torch.tensor([[3]], dtype=torch.long)
    GOAL = torch.tensor([[4]], dtype=torch.float32)


class ChronoMixin:
    """Class for running VLFM in a Chrono environment."""
    _stop_action: Tensor = TorchActionIDs.STOP
    _start_yaw: Union[float, None] = None  # must be set by _reset() method
    _observations_cache: Dict[str, Any] = {}
    _policy_info: Dict[str, Any] = {}
    _compute_frontiers: bool = True
    _visualize: bool = True

    def __init__(self, camera_height: float, min_depth: float, max_depth: float, camera_fov: float, image_width: int, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._camera_height = camera_height
        self._min_depth = min_depth
        self._max_depth = max_depth
        self._camera_fov = np.deg2rad(camera_fov)
        self._image_width = image_width
        # Convert to radians if in degrees
        camera_fov_rad = np.radians(camera_fov)
        self._fx = self._fy = image_width / (2 * np.tan(camera_fov_rad / 2))

    def act(self, observations, rnn_hidden_states: Any, prev_actions: Any, masks: Tensor, deterministic: bool = False) -> Tuple[Tensor, Any]:
        parent_cls: BaseObjectNavPolicy = super()

        try:
            action, rnn_hidden_states = parent_cls.act(
                observations, rnn_hidden_states, prev_actions, masks, deterministic)
        except StopIteration:
            action = self._stop_action

        return action, rnn_hidden_states

    def _initialize(self) -> Tensor:
        """Turn left 30 degrees 12 times to get a 360 view at the beginning"""
        self._done_initializing = not self._num_steps < 24  # type: ignore
        return TorchActionIDs.TURN_LEFT

    def _reset(self) -> None:
        parent_cls: BaseObjectNavPolicy = super()  # type: ignore
        parent_cls._reset()
        self._start_yaw = None

    def _get_policy_info(self, detections: ObjectDetections) -> Dict[str, Any]:
        """Get policy info for logging"""
        parent_cls: BaseObjectNavPolicy = super()  # type: ignore
        info = parent_cls._get_policy_info(detections)

        if not self._visualize:  # type: ignore
            return info

        if self._start_yaw is None:
            self._start_yaw = self._observations_cache["habitat_start_yaw"]
        info["start_yaw"] = self._start_yaw
        return info

    def _cache_observations(self: Union["ChronoMixin", BaseObjectNavPolicy], observations) -> None:
        """Caches the rgb, depth, and camera transform from the observations.

        Args:
           observations: The observations from the current timestep.
        """
        if len(self._observations_cache) > 0:
            return
        rgb = observations["rgb"].cpu().numpy()
        depth = observations["depth"].cpu().numpy()
        x, y = observations["gps"].cpu().numpy()
        camera_yaw = observations["compass"].cpu().item()

        # Print shape of all of above
        depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
        camera_position = np.array([x, y, self._camera_height])
        robot_xy = camera_position[:2]
        tf_camera_to_episodic = xyz_yaw_to_tf_matrix(
            camera_position, camera_yaw)

        self._obstacle_map: ObstacleMap
        if self._compute_frontiers:
            self._obstacle_map.update_map(
                depth,
                tf_camera_to_episodic,
                self._min_depth,
                self._max_depth,
                self._fx,
                self._fy,
                self._camera_fov,
            )
            frontiers = self._obstacle_map.frontiers
            self._obstacle_map.update_agent_traj(robot_xy, camera_yaw)
        else:
            if "frontier_sensor" in observations:
                frontiers = observations["frontier_sensor"][0].cpu().numpy()
            else:
                frontiers = np.array([])

        self._observations_cache = {
            "frontier_sensor": frontiers,
            "nav_depth": observations["depth"],  # for pointnav
            "robot_xy": robot_xy,
            "robot_heading": camera_yaw,
            "object_map_rgbd": [
                (
                    rgb,
                    depth,
                    tf_camera_to_episodic,
                    self._min_depth,
                    self._max_depth,
                    self._fx,
                    self._fy,
                )
            ],
            "value_map_rgbd": [
                (
                    rgb,
                    depth,
                    tf_camera_to_episodic,
                    self._min_depth,
                    self._max_depth,
                    self._camera_fov,
                )
            ],
            "habitat_start_yaw": observations["compass"].item(),
        }

# @baseline_registry.register_policy
# class OracleFBEPolicy(ChronoMixin, BaseObjectNavPolicy):
#     def _explore(self, observations: TensorDict) -> Tensor:
#         explorer_key = [k for k in observations.keys() if k.endswith("_explorer")][0]
#         pointnav_action = observations[explorer_key]
#         return pointnav_action


# @baseline_registry.register_policy
# class SuperOracleFBEPolicy(ChronoMixin, BaseObjectNavPolicy):
#     def act(
#         self,
#         observations: TensorDict,
#         rnn_hidden_states: Any,  # can be anything because it is not used
#         *args: Any,
#         **kwargs: Any,
#     ) -> PolicyActionData:
#         return PolicyActionData(
#             actions=observations[BaseExplorer.cls_uuid],
#             rnn_hidden_states=rnn_hidden_states,
#             policy_info=[self._policy_info],
#         )


class ChronoITMPolicy(ChronoMixin, ITMPolicy):
    pass


class ChronoITMPolicyV2(ChronoMixin, ITMPolicyV2):
    pass


class ChronoITMPolicyV3(ChronoMixin, ITMPolicyV3):
    pass


# @dataclass
# class VLFMPolicyConfig(VLFMConfig, PolicyConfig):
#     pass

    # def _explore(self, observations: TensorDict) -> Tensor:
    #     frontiers = self._observations_cache["frontier_sensor"]
    #     if np.array_equal(frontiers, np.zeros((1, 2))) or len(frontiers) == 0:
    #         print("No frontiers found during exploration, stopping.")
    #         return self._stop_action
    #     best_frontier, best_value = self._get_best_frontier(
    #         observations, frontiers)
    #     os.environ["DEBUG_INFO"] = f"Best value: {best_value*100:.2f}%"
    #     print(f"Best value: {best_value*100:.2f}%")

    #     # TODO: Here now we should actually in Chrono we should just go to this frontier with a SetPosition action
    #     pointnav_action = self._pointnav(best_frontier, stop=False)

    #     return pointnav_action

    # def _get_best_frontier(
    #     self,
    #     observations: TensorDict,
    #     frontiers: np.ndarray,
    # ) -> Tuple[np.ndarray, float]:
    #     """Returns the best frontier and its value based on self._value_map.

    #     Args:
    #         observations (TensorDict): The observations from the environment.
    #         frontiers (np.ndarray): The frontiers to choose from, array of 2D points.

    #     Returns:
    #         Tuple[np.ndarray, float]: The best frontier and its value.
    #     """
    #     # The points and values will be sorted in descending order
    #     sorted_pts, sorted_values = self._sort_frontiers_by_value(
    #         observations, frontiers)
    #     robot_xy = self._observations_cache["robot_xy"]
    #     best_frontier_idx = None
    #     top_two_values = tuple(sorted_values[:2])

    #     os.environ["DEBUG_INFO"] = ""
    #     # If there is a last point pursued, then we consider sticking to pursuing it
    #     # if it is still in the list of frontiers and its current value is not much
    #     # worse than self._last_value.
    #     if not np.array_equal(self._last_frontier, np.zeros(2)):
    #         curr_index = None

    #         for idx, p in enumerate(sorted_pts):
    #             if np.array_equal(p, self._last_frontier):
    #                 # Last point is still in the list of frontiers
    #                 curr_index = idx
    #                 break

    #         if curr_index is None:
    #             closest_index = closest_point_within_threshold(
    #                 sorted_pts, self._last_frontier, threshold=0.5)

    #             if closest_index != -1:
    #                 # There is a point close to the last point pursued
    #                 curr_index = closest_index

    #         if curr_index is not None:
    #             curr_value = sorted_values[curr_index]
    #             if curr_value + 0.01 > self._last_value:
    #                 # The last point pursued is still in the list of frontiers and its
    #                 # value is not much worse than self._last_value
    #                 print("Sticking to last point.")
    #                 os.environ["DEBUG_INFO"] += "Sticking to last point. "
    #                 best_frontier_idx = curr_index

    #     # If there is no last point pursued, then just take the best point, given that
    #     # it is not cyclic.
    #     if best_frontier_idx is None:
    #         for idx, frontier in enumerate(sorted_pts):
    #             cyclic = self._acyclic_enforcer.check_cyclic(
    #                 robot_xy, frontier, top_two_values)
    #             if cyclic:
    #                 print("Suppressed cyclic frontier.")
    #                 continue
    #             best_frontier_idx = idx
    #             break

    #     if best_frontier_idx is None:
    #         print("All frontiers are cyclic. Just choosing the closest one.")
    #         os.environ["DEBUG_INFO"] += "All frontiers are cyclic. "
    #         best_frontier_idx = max(
    #             range(len(frontiers)),
    #             key=lambda i: np.linalg.norm(frontiers[i] - robot_xy),
    #         )

    #     best_frontier = sorted_pts[best_frontier_idx]
    #     best_value = sorted_values[best_frontier_idx]
    #     self._acyclic_enforcer.add_state_action(
    #         robot_xy, best_frontier, top_two_values)
    #     self._last_value = best_value
    #     self._last_frontier = best_frontier
    #     os.environ["DEBUG_INFO"] += f" Best value: {best_value*100:.2f}%"

    #     return best_frontier, best_value
