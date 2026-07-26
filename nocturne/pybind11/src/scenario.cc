// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "scenario.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <string>
#include <variant>

#include "geometry/geometry_utils.h"
#include "numpy_utils.h"
#include "object.h"

namespace py = pybind11;

namespace nocturne {

using geometry::utils::kHalfPi;

void DefineScenario(py::module& m) {
  py::class_<Scenario, std::shared_ptr<Scenario>>(m, "Scenario")
      .def(py::init<const std::string&,
                    const std::unordered_map<
                        std::string, std::variant<bool, int64_t, float>>&>(),
           "Constructor for Scenario", py::arg("scenario_path"),
           py::arg("config"))

      // Properties
      .def_property_readonly("name", &Scenario::name)

      // Methods
      .def("vehicles", &Scenario::vehicles, py::return_value_policy::reference)
      .def("pedestrians", &Scenario::pedestrians,
           py::return_value_policy::reference)
      .def("cyclists", &Scenario::cyclists, py::return_value_policy::reference)
      .def("objects", &Scenario::objects, py::return_value_policy::reference)
      .def("moving_objects", &Scenario::moving_objects,
           py::return_value_policy::reference)
      .def("remove_object", &Scenario::RemoveObject)
      .def("road_lines", &Scenario::road_lines)

      .def("ego_state",
           [](const Scenario& scenario, const Object& src) {
             // EgoState is pure C++ (no Python objects); release the GIL for
             // the computation, re-acquire before building the numpy array.
             py::gil_scoped_release release;
             const auto state = scenario.EgoState(src);
             py::gil_scoped_acquire acquire;
             return utils::AsNumpyArray(state);
           })
      .def(
          "visible_state",
          [](const Scenario& scenario, const Object& src, float view_dist,
             float view_angle, float head_angle, bool padding) {
            py::gil_scoped_release release;
            const auto result =
                scenario.VisibleState(src, view_dist, view_angle, head_angle, padding);
            py::gil_scoped_acquire acquire;
            return utils::AsNumpyArrayDict(result);
          },
          py::arg("object"), py::arg("view_dist") = 60,
          py::arg("view_angle") = kHalfPi, py::arg("head_angle") = 0.0, py::arg("padding") = false)
      .def(
          "visible_objects_state",
          [](const Scenario& scenario, const Object& src, float view_dist,
             float view_angle, float head_angle, bool padding) {
            py::gil_scoped_release release;
            const auto result = scenario.VisibleObjectsState(
                src, view_dist, view_angle, head_angle, padding);
            py::gil_scoped_acquire acquire;
            return utils::AsNumpyArray(result);
          },
          py::arg("object"), py::arg("view_dist") = 60,
          py::arg("view_angle") = kHalfPi, py::arg("head_angle") = 0.0, py::arg("padding") = false)
      .def(
          "flattened_visible_state",
          [](const Scenario& scenario, const Object& src, float view_dist,
             float view_angle, float head_angle) {
            py::gil_scoped_release release;
            const auto result = scenario.FlattenedVisibleState(
                src, view_dist, view_angle, head_angle);
            py::gil_scoped_acquire acquire;
            return utils::AsNumpyArray(result);
          },
          py::arg("object"), py::arg("view_dist") = 60,
          py::arg("view_angle") = kHalfPi, py::arg("head_angle") = 0.0)
      .def(
          "all_object_states",
          [](const Scenario& scenario) {
            py::gil_scoped_release release;
            const auto result = scenario.AllObjectStates();
            py::gil_scoped_acquire acquire;
            return utils::AsNumpyArray(result);
          },
          "Return (N, 6) float32 array of (id, x, y, speed, heading, type) "
          "for all vehicles/pedestrians/cyclists. type: 0=vehicle, "
          "1=pedestrian, 2=cyclist.")
      .def(
          "road_edge_points",
          [](const Scenario& scenario) {
            py::gil_scoped_release release;
            const auto result = scenario.RoadEdgePoints();
            py::gil_scoped_acquire acquire;
            return utils::AsNumpyArray(result);
          },
          "Return (M, 2) float32 array of (x, y) for all road-edge geometry "
          "points.")
      .def("save_snapshot",
           [](Scenario& scenario) {
             py::gil_scoped_release release;
             scenario.SaveSnapshot();
           },
           "Save post-warmup object state for fast reset.")
      .def("restore_snapshot",
           [](Scenario& scenario) {
             py::gil_scoped_release release;
             scenario.RestoreSnapshot();
           },
           "Restore object state from a previously saved snapshot.")
      .def(
          "occupancy_grid",
          [](const Scenario& scenario, const Object& ego, int64_t rows,
             int64_t cols, float forward_dist, float backward_dist,
             float lateral_dist, float vehicle_weight, float vru_weight,
             float road_edge_weight, int64_t ego_id) {
            py::gil_scoped_release release;
            const auto result = scenario.OccupancyGrid(
                ego, rows, cols, forward_dist, backward_dist, lateral_dist,
                vehicle_weight, vru_weight, road_edge_weight, ego_id);
            py::gil_scoped_acquire acquire;
            return utils::AsNumpyArray(result);
          },
          "Return (3, rows, cols) float32 occupancy grid (occ, rel_vx, rel_vy).",
          py::arg("ego"), py::arg("rows"), py::arg("cols"),
          py::arg("forward_dist"), py::arg("backward_dist"),
          py::arg("lateral_dist"), py::arg("vehicle_weight"),
          py::arg("vru_weight"), py::arg("road_edge_weight"),
          py::arg("ego_id") = -1)
      .def("expert_position", &Scenario::ExpertPosition)
      .def("expert_heading", &Scenario::ExpertHeading)
      .def("expert_speed", &Scenario::ExpertSpeed)
      .def("expert_velocity", &Scenario::ExpertVelocity)
      .def("expert_action", &Scenario::ExpertAction)
      .def("expert_pos_shift", &Scenario::ExpertPosShift)
      .def("expert_heading_shift", &Scenario::ExpertHeadingShift)

      // TODO: Deprecate the legacy interfaces below.
      .def("getVehicles", &Scenario::vehicles,
           py::return_value_policy::reference)
      .def("getPedestrians", &Scenario::pedestrians,
           py::return_value_policy::reference)
      .def("getCyclists", &Scenario::cyclists,
           py::return_value_policy::reference)
      .def("getObjectsThatMoved", &Scenario::moving_objects,
           py::return_value_policy::reference)
      .def("getObjects", &Scenario::objects, py::return_value_policy::reference)
      .def("getRoadLines", &Scenario::road_lines)
      .def(
          "getImage",
          [](Scenario& scenario, uint64_t img_height, uint64_t img_width,
             bool draw_target_positions, float padding, Object* source,
             uint64_t view_height, uint64_t view_width,
             bool rotate_with_source) {
            return utils::AsNumpyArray<unsigned char>(scenario.Image(
                img_height, img_width, draw_target_positions, padding, source,
                view_height, view_width, rotate_with_source));
          },
          "Return a numpy array of dimension (img_height, img_width, 4) "
          "representing an image of the scene.",
          py::arg("img_height") = 1000, py::arg("img_width") = 1000,
          py::arg("draw_target_positions") = true, py::arg("padding") = 50.0f,
          py::arg("source") = nullptr, py::arg("view_height") = 200,
          py::arg("view_width") = 200, py::arg("rotate_with_source") = true)
      .def(
          "getFeaturesImage",
          [](Scenario& scenario, const Object& source, float view_dist,
             float view_angle, float head_angle, uint64_t img_height,
             uint64_t img_width, float padding, bool draw_target_position) {
            return utils::AsNumpyArray<unsigned char>(
                scenario.EgoVehicleFeaturesImage(
                    source, view_dist, view_angle, head_angle, img_height,
                    img_width, padding, draw_target_position));
          },
          "Return a numpy array of dimension (img_height, img_width, 4) "
          "representing an image of what is returned by getVisibleState(?).",
          py::arg("source"), py::arg("view_dist") = 120.0f,
          py::arg("view_angle") = geometry::utils::kPi * 0.8f,
          py::arg("head_angle") = 0.0f, py::arg("img_height") = 1000,
          py::arg("img_width") = 1000, py::arg("padding") = 0.0f,
          py::arg("draw_target_position") = true)
      .def(
          "getConeImage",
          [](Scenario& scenario, const Object& source, float view_dist,
             float view_angle, float head_angle, uint64_t img_height,
             uint64_t img_width, float padding, bool draw_target_position) {
            return utils::AsNumpyArray<unsigned char>(
                scenario.EgoVehicleConeImage(source, view_dist, view_angle,
                                             head_angle, img_height, img_width,
                                             padding, draw_target_position));
          },
          "Return a numpy array of dimension (img_height, img_width, 4) "
          "representing a cone of what the agent sees.",
          py::arg("source"), py::arg("view_dist") = 120.0f,
          py::arg("view_angle") = geometry::utils::kPi * 0.8f,
          py::arg("head_angle") = 0.0f, py::arg("img_height") = 1000,
          py::arg("img_width") = 1000, py::arg("padding") = 0.0f,
          py::arg("draw_target_position") = true)
      .def("removeVehicle", &Scenario::RemoveObject)
      .def("getExpertAction", &Scenario::ExpertAction)
      .def("getExpertSpeeds", &Scenario::ExpertVelocity)
      .def("getMaxNumVisibleObjects", &Scenario::getMaxNumVisibleObjects)
      .def("getMaxNumVisibleRoadPoints", &Scenario::getMaxNumVisibleRoadPoints)
      .def("getMaxNumVisibleStopSigns", &Scenario::getMaxNumVisibleStopSigns)
      .def("getMaxNumVisibleTrafficLights",
           &Scenario::getMaxNumVisibleTrafficLights)
      .def("getObjectFeatureSize", &Scenario::getObjectFeatureSize)
      .def("getRoadPointFeatureSize", &Scenario::getRoadPointFeatureSize)
      .def("getTrafficLightFeatureSize", &Scenario::getTrafficLightFeatureSize)
      .def("getStopSignsFeatureSize", &Scenario::getStopSignsFeatureSize)
      .def("getEgoFeatureSize", &Scenario::getEgoFeatureSize);
}

}  // namespace nocturne
