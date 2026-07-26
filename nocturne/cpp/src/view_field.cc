// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "view_field.h"

#include <algorithm>
#include <functional>
#include <iterator>
#include <limits>
#include <numeric>
#include <optional>
#include <type_traits>
#include <unordered_set>
#include <utility>

#include "geometry/geometry_utils.h"
#include "geometry/intersection.h"
#include "geometry/line_segment.h"
#include "geometry/polygon.h"
#include "utils/data_utils.h"

namespace nocturne {

namespace {

using geometry::Circle;
using geometry::CircleLike;
using geometry::CircularSector;
using geometry::ConvexPolygon;
using geometry::LineSegment;
using geometry::Vector2D;
using geometry::utils::kTwoPi;

// TODO: Find a better eps.
constexpr float kRotationEps = 1e-4f;

void AppendSightCandidates(const CircleLike& vision,
                           const geometry::Vector2D& p,
                           std::vector<Vector2D>& candidates) {
  const geometry::Vector2D& o = vision.center();
  const float r = vision.radius();
  const geometry::Vector2D d = p - o;
  const geometry::Vector2D d0 = d / d.Norm() * r;
  const geometry::Vector2D d1 = d0.Rotate(kRotationEps);
  const geometry::Vector2D d2 = d0.Rotate(-kRotationEps);
  const geometry::Vector2D p0 = o + d0;
  const geometry::Vector2D p1 = o + d1;
  const geometry::Vector2D p2 = o + d2;
  candidates.push_back(p0);
  if (vision.Contains(p1)) {
    candidates.push_back(p1);
  }
  if (vision.Contains(p2)) {
    candidates.push_back(p2);
  }
}

std::vector<geometry::utils::MaskType> VisibleObjectsImpl(
    const std::vector<const ObjectBase*>& objects, const Vector2D& o,
    const std::vector<Vector2D>& points) {
  const int64_t n = objects.size();
  const int64_t m = points.size();
  auto [x, y] = geometry::utils::PackCoordinates(points);

  std::vector<float> dis(m, 1.0f);
  std::vector<int64_t> idx(m, -1);
  for (int64_t i = 0; i < n; ++i) {
    if (!objects[i]->can_block_sight()) {
      continue;
    }

    const std::vector<float> cur_dis = geometry::BatchParametricIntersection(
        o, x, y, objects[i]->BoundingPolygon());
    for (int64_t j = 0; j < m; ++j) {
      if (cur_dis[j] < dis[j]) {
        dis[j] = cur_dis[j];
        idx[j] = i;
      }
    }
  }

  std::vector<geometry::utils::MaskType> mask(n, 0);
  for (int64_t i = 0; i < m; ++i) {
    const geometry::Vector2D p = LineSegment(o, points[i]).Point(dis[i]);
    x[i] = p.x();
    y[i] = p.y();
    if (idx[i] != -1) {
      mask[idx[i]] = 1;
    }
  }
  for (int64_t i = 0; i < n; ++i) {
    const std::vector<geometry::utils::MaskType> cur_mask =
        BatchIntersects(objects[i]->BoundingPolygon(), o, x, y);
    // Use bitwise operation to get better performance.
    mask[i] |= std::accumulate(cur_mask.cbegin(), cur_mask.cend(),
                               geometry::utils::MaskType(0),
                               std::bit_or<geometry::utils::MaskType>());
  }

  return mask;
}

bool IsVisibleNonblockingObject(const CircleLike& vision,
                                const ObjectBase* obj) {
  const auto edges = obj->BoundingPolygon().Edges();
  for (const LineSegment& edge : edges) {
    // Check one endpoint should be enough, the othe one will be checked in
    // the next edge.
    const Vector2D& x = edge.Endpoint0();
    if (vision.Contains(x)) {
      return true;
    }
    const auto [p, q] = vision.Intersection(edge);
    if (p.has_value() || q.has_value()) {
      return true;
    }
  }
  return false;
}

}  // namespace

ViewField::ViewField(const geometry::Vector2D& center, float radius,
                     float heading, float theta)
    : panoramic_view_(theta >= kTwoPi) {
  if (panoramic_view_) {
    vision_ = std::make_unique<Circle>(center, radius);
  } else {
    vision_ = std::make_unique<CircularSector>(center, radius, heading, theta);
  }
}

// O(N^2) algorithm.
// TODO: Implment O(NlogN) algorithm when there are too many objects.
std::vector<const ObjectBase*> ViewField::VisibleObjects(
    const std::vector<const ObjectBase*>& objects) const {
  const int64_t n = objects.size();
  if (n <= 1) {
    return objects;
  }
  // For small N the overhead of the angular sweep setup exceeds the savings.
  // The crossover is empirically around 8-12 objects within the view field.
  if (n < 10) {
    const Vector2D& o = vision_->center();
    std::vector<Vector2D> sight_endpoints = ComputeSightEndpoints(objects);
    const std::vector<geometry::utils::MaskType> mask =
        VisibleObjectsImpl(objects, o, sight_endpoints);
    std::vector<const ObjectBase*> ret;
    for (int64_t i = 0; i < n; ++i) {
      if (mask[i]) ret.push_back(objects[i]);
    }
    return ret;
  }
  return VisibleObjectsAngularSweep(objects);
}

std::vector<const ObjectBase*> ViewField::VisibleObjectsAngularSweep(
    const std::vector<const ObjectBase*>& objects) const {
  // O(N log N) angular-sweep visibility:
  // 1. For each blocker, compute the angular interval [a_min, a_max] it
  //    subtends from the observer, and its minimum distance.
  // 2. Sort blockers by min distance (closest first).
  // 3. Sweep in distance order, maintaining a covered-interval set. A
  //    blocker is visible iff its angular interval is NOT fully covered by
  //    the union of intervals of closer blockers.
  // This replaces the O(N^2) BatchParametricIntersection + BatchIntersects
  // approach with angular arithmetic + interval operations.
  const int64_t n = objects.size();
  const Vector2D& o = vision_->center();
  const float radius = vision_->radius();

  struct BlockerInfo {
    const ObjectBase* ptr;
    float min_dist;
    float a_min;
    float a_max;
  };

  std::vector<BlockerInfo> blockers;
  blockers.reserve(n);
  for (int64_t i = 0; i < n; ++i) {
    if (!objects[i]->can_block_sight()) continue;
    const auto poly = objects[i]->BoundingPolygon();
    const auto vertices = poly.Vertices();
    if (vertices.empty()) continue;
    float a_min = std::numeric_limits<float>::max();
    float a_max = std::numeric_limits<float>::lowest();
    float min_dist = std::numeric_limits<float>::max();
    for (const auto& v : vertices) {
      const Vector2D d = v - o;
      const float dist = d.Norm();
      if (dist < 1e-6f) continue;
      if (!vision_->Contains(v)) continue;
      const float angle = std::atan2(d.y(), d.x());
      a_min = std::min(a_min, angle);
      a_max = std::max(a_max, angle);
      min_dist = std::min(min_dist, dist);
    }
    if (a_min > a_max) continue;  // No visible vertices
    blockers.push_back({objects[i], min_dist, a_min, a_max});
  }

  // Sort by min distance (closest first) so closer blockers occlude farther.
  std::sort(blockers.begin(), blockers.end(),
            [](const BlockerInfo& a, const BlockerInfo& b) {
              return a.min_dist < b.min_dist;
            });

  // Covered angular intervals (sorted, non-overlapping). A blocker is
  // visible iff its [a_min, a_max] is not fully contained in the union.
  std::vector<std::pair<float, float>> covered;

  std::vector<const ObjectBase*> ret;
  for (const auto& b : blockers) {
    // Check if [b.a_min, b.a_max] is fully covered by `covered`.
    bool fully_covered = false;
    for (const auto& c : covered) {
      if (c.first <= b.a_min && c.second >= b.a_max) {
        fully_covered = true;
        break;
      }
    }
    if (!fully_covered) {
      ret.push_back(b.ptr);
      // Insert [b.a_min, b.a_max] into the covered set and merge overlaps.
      covered.emplace_back(b.a_min, b.a_max);
      std::sort(covered.begin(), covered.end());
      std::vector<std::pair<float, float>> merged;
      for (const auto& iv : covered) {
        if (!merged.empty() && iv.first <= merged.back().second) {
          merged.back().second = std::max(merged.back().second, iv.second);
        } else {
          merged.push_back(iv);
        }
      }
      covered = std::move(merged);
    }
  }
  return ret;
}

void ViewField::FilterVisibleObjects(
    std::vector<const ObjectBase*>& objects) const {
  const int64_t n = objects.size();
  if (n <= 1) return;
  if (n < 10) {
    const Vector2D& o = vision_->center();
    std::vector<Vector2D> sight_endpoints = ComputeSightEndpoints(objects);
    const std::vector<geometry::utils::MaskType> mask =
        VisibleObjectsImpl(objects, o, sight_endpoints);
    const int64_t pivot = utils::MaskedPartition(mask, objects);
    objects.resize(pivot);
  } else {
    std::vector<const ObjectBase*> visible = VisibleObjectsAngularSweep(objects);
    std::unordered_set<const ObjectBase*> visible_set(visible.begin(),
                                                       visible.end());
    auto pivot = std::partition(objects.begin(), objects.end(),
                                 [&](const ObjectBase* o) {
                                   return visible_set.count(o) > 0;
                                 });
    objects.resize(std::distance(objects.begin(), pivot));
  }
}

std::vector<const ObjectBase*> ViewField::VisibleNonblockingObjects(
    const std::vector<const ObjectBase*>& objects) const {
  std::vector<const ObjectBase*> ret;
  const CircleLike* vptr = vision_.get();
  std::copy_if(objects.cbegin(), objects.cend(), std::back_inserter(ret),
               [vptr](const ObjectBase* o) {
                 return IsVisibleNonblockingObject(*vptr, o);
               });
  return ret;
}

void ViewField::FilterVisibleNonblockingObjects(
    std::vector<const ObjectBase*>& objects) const {
  const CircleLike* vptr = vision_.get();
  auto pivot = std::partition(objects.begin(), objects.end(),
                              [vptr](const ObjectBase* o) {
                                return IsVisibleNonblockingObject(*vptr, o);
                              });
  objects.resize(std::distance(objects.begin(), pivot));
}

std::vector<const geometry::PointLike*> ViewField::VisiblePoints(
    const std::vector<const geometry::PointLike*>& objects) const {
  std::vector<const geometry::PointLike*> ret;
  const std::vector<geometry::utils::MaskType> mask =
      vision_->BatchContains(objects);
  const int64_t n = objects.size();
  for (int64_t i = 0; i < n; ++i) {
    if (mask[i]) {
      ret.push_back(objects[i]);
    }
  }
  return ret;
}

void ViewField::FilterVisiblePoints(
    std::vector<const geometry::PointLike*>& objects) const {
  const std::vector<geometry::utils::MaskType> mask =
      vision_->BatchContains(objects);
  const int64_t pivot = utils::MaskedPartition(mask, objects);
  objects.resize(pivot);
}

std::vector<Vector2D> ViewField::ComputeSightEndpoints(
    const std::vector<const ObjectBase*>& objects) const {
  std::vector<Vector2D> ret;
  const Vector2D& o = vision_->center();
  if (!panoramic_view_) {
    const CircularSector* vptr = dynamic_cast<CircularSector*>(vision_.get());
    ret.push_back(o + vptr->Radius0());
    ret.push_back(o + vptr->Radius1());
  }
  for (const ObjectBase* obj : objects) {
    const auto edges = obj->BoundingPolygon().Edges();
    for (const LineSegment& edge : edges) {
      // Check one endpoint should be enough, the othe one will be checked in
      // the next edge.
      const Vector2D& x = edge.Endpoint0();
      if (vision_->Contains(x)) {
        AppendSightCandidates(*vision_, x, ret);
      }
      const auto [p, q] = vision_->Intersection(edge);
      if (p.has_value()) {
        AppendSightCandidates(*vision_, *p, ret);
      }
      if (q.has_value()) {
        AppendSightCandidates(*vision_, *q, ret);
      }
    }
  }
  // Remove duplicate endpoints.
  std::sort(ret.begin(), ret.end());
  auto it = std::unique(ret.begin(), ret.end());
  ret.resize(std::distance(ret.begin(), it));
  return ret;
}

}  // namespace nocturne
