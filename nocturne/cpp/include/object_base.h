// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <SFML/Graphics.hpp>
#include <cstdint>
#include <string>

#include "geometry/aabb.h"
#include "geometry/aabb_interface.h"
#include "geometry/polygon.h"
#include "geometry/vector_2d.h"

namespace nocturne {

enum class CollisionType {
  kNotCollided = 0,
  kVehicleVehicleCollision = 1,
  kVehicleRoadEdgeCollision = 2,
};

class ObjectBase : public sf::Drawable, public geometry::AABBInterface {
 public:
  ObjectBase() = default;

  explicit ObjectBase(const geometry::Vector2D& position)
      : position_(position) {}

  ObjectBase(const geometry::Vector2D& position, bool can_block_sight,
             bool can_be_collided, bool check_collision)
      : position_(position),
        can_block_sight_(can_block_sight),
        can_be_collided_(can_be_collided),
        check_collision_(check_collision) {}

  const geometry::Vector2D& position() const { return position_; }
  void set_position(const geometry::Vector2D& position) {
    position_ = position;
  }
  void set_position(float x, float y) { position_ = geometry::Vector2D(x, y); }

  bool can_block_sight() const { return can_block_sight_; }
  bool can_be_collided() const { return can_be_collided_; }
  bool check_collision() const { return check_collision_; }
  // Allow runtime toggling so the env can scope collision checks to only
  // controlled agents (non-controlled objects get check_collision=false,
  // so the O(N^2) all-pairs loop skips pairs where neither side is
  // controlled). Both sides stay correct: if ego (check_collision=true)
  // hits a non-ego (check_collision=false), the check still runs and
  // both get collided=true.
  void set_check_collision(bool check_collision) {
    check_collision_ = check_collision;
  }

  bool collided() const { return collided_; }
  void set_collided(bool collided) { collided_ = collided; }

  CollisionType collision_type() const { return collision_type_; }
  void set_collision_type(CollisionType collision_type) {
    collision_type_ = collision_type;
  }

  void ResetCollision() {
    collided_ = false;
    collision_type_ = CollisionType::kNotCollided;
  }

  virtual float Radius() const = 0;

  virtual geometry::ConvexPolygon BoundingPolygon() const = 0;

  geometry::AABB GetAABB() const override {
    return BoundingPolygon().GetAABB();
  }

 protected:
  geometry::Vector2D position_;

  const bool can_block_sight_ = false;
  const bool can_be_collided_ = false;
  bool check_collision_ = false;
  bool collided_ = false;
  CollisionType collision_type_ = CollisionType::kNotCollided;
};

}  // namespace nocturne
