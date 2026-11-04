//! Kinematic bicycle car model at the rear axle.
//!
//! Used identically by headless training and the renderer, so a policy that
//! drives in the game drives in training.

use crate::math::{Vec2, clampf};

#[derive(Clone, Copy, Debug)]
pub struct CarSpec {
    /// Distance between rear and front axles (m).
    pub wheelbase: f32,
    /// Body extents relative to the rear axle (m).
    pub rear_overhang: f32,
    pub front_overhang: f32,
    pub half_width: f32,
    /// Maximum road-wheel angle (rad).
    pub max_steer: f32,
    /// Steering slew rate (rad/s).
    pub steer_rate: f32,
    pub engine_accel: f32,
    pub brake_accel: f32,
    /// Linear drag coefficient (1/s).
    pub drag: f32,
    pub rolling: f32,
    pub max_speed: f32,
    pub max_reverse: f32,
    /// Cornering grip: maximum lateral acceleration (m/s^2). At speed the
    /// available steering angle shrinks, so fast cars must slow for corners.
    pub max_lat_accel: f32,
}

impl Default for CarSpec {
    fn default() -> Self {
        Self {
            wheelbase: 2.45,
            rear_overhang: 0.95,
            front_overhang: 3.35,
            half_width: 1.05,
            max_steer: 0.56,
            steer_rate: 3.2,
            engine_accel: 12.5,
            brake_accel: 9.0,
            drag: 0.30,
            rolling: 0.7,
            max_speed: 35.0,
            max_reverse: 7.0,
            max_lat_accel: 16.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Car {
    /// Rear-axle position.
    pub pos: Vec2,
    pub heading: f32,
    pub speed: f32,
    pub steer: f32,
}

impl Car {
    pub fn new(pos: Vec2, heading: f32) -> Self {
        Self {
            pos,
            heading,
            speed: 0.0,
            steer: 0.0,
        }
    }

    #[inline]
    pub fn forward(&self) -> Vec2 {
        Vec2::from_angle(self.heading)
    }

    #[inline]
    pub fn yaw_rate(&self, spec: &CarSpec) -> f32 {
        self.speed / spec.wheelbase * self.steer.tan()
    }

    /// Advance physics one fixed tick. `throttle` and `steer_input` are in
    /// [-1, 1]; positive steering turns left (counter-clockwise in world space).
    pub fn step(&mut self, throttle: f32, steer_input: f32, dt: f32, spec: &CarSpec) {
        let throttle = clampf(throttle, -1.0, 1.0);
        let steer_input = clampf(steer_input, -1.0, 1.0);

        // Steering actuator has finite slew. Available lock shrinks with speed
        // to respect a lateral-grip budget: v^2 * tan(delta) / L <= a_lat.
        let grip_lock = if self.speed > 1.0 {
            (spec.max_lat_accel * spec.wheelbase / (self.speed * self.speed)).atan()
        } else {
            spec.max_steer
        };
        let lock = spec.max_steer.min(grip_lock);
        // Steering input is a fraction of the *mechanical* lock, then clamped
        // to the grip-limited lock at the current speed. Controllers (and the
        // network) can therefore request a precise angle and saturate at the
        // limit, which is the cue to slow down.
        let target = clampf(steer_input * spec.max_steer, -lock, lock);
        let max_step = spec.steer_rate * dt;
        self.steer += clampf(target - self.steer, -max_step, max_step);

        // Longitudinal forces: engine/brake, drag, rolling resistance.
        let accel = if throttle >= 0.0 {
            throttle * spec.engine_accel
        } else {
            throttle * spec.brake_accel
        };
        let resist = spec.drag * self.speed.abs() + spec.rolling;
        self.speed += (accel - resist.copysign(self.speed.max(1e-3))) * dt;
        self.speed = clampf(self.speed, -spec.max_reverse, spec.max_speed);

        // Bicycle model integration.
        let yaw = self.yaw_rate(spec);
        self.heading += yaw * dt;
        self.pos += Vec2::from_angle(self.heading) * (self.speed * dt);
    }

    /// Four body corners in world space (rear-left, front-left, front-right,
    /// rear-right), computed from body-local extents.
    pub fn corners(&self, spec: &CarSpec) -> [Vec2; 4] {
        let f = self.forward();
        let r = f.perp();
        let body_center = self.pos + f * (spec.wheelbase * 0.5);
        let l = spec.wheelbase * 0.5 + spec.rear_overhang + spec.front_overhang;
        let half_l = l * 0.5;
        let hw = spec.half_width;
        [
            body_center - f * half_l - r * hw,
            body_center + f * half_l - r * hw,
            body_center + f * half_l + r * hw,
            body_center - f * half_l + r * hw,
        ]
    }

    #[inline]
    pub fn body_center(&self, spec: &CarSpec) -> Vec2 {
        self.pos + self.forward() * (spec.wheelbase * 0.5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::vec2;

    #[test]
    fn accelerates_toward_terminal_speed() {
        let spec = CarSpec::default();
        let mut c = Car::new(Vec2::ZERO, 0.0);
        for _ in 0..300 {
            c.step(1.0, 0.0, 1.0 / 60.0, &spec);
        }
        assert!(c.speed > 20.0, "speed {}", c.speed);
        assert!(c.pos.x > 0.0);
        assert!(c.pos.y.abs() < 1e-3);
        // Terminal speed respects drag: a = 12.5 - 0.3v - 0.7.
        for _ in 0..3000 {
            c.step(1.0, 0.0, 1.0 / 60.0, &spec);
        }
        let expected = ((spec.engine_accel - spec.rolling) / spec.drag).min(spec.max_speed);
        assert!(
            (c.speed - expected).abs() < 0.5,
            "speed {} expected {}",
            c.speed,
            expected
        );
        assert!(c.speed <= spec.max_speed);
    }

    #[test]
    fn turning_radius_matches_bicycle_model() {
        let mut spec = CarSpec::default();
        spec.max_lat_accel = 10_000.0; // isolate the bicycle geometry
        let mut c = Car::new(Vec2::ZERO, 0.0);
        // Reach a steady speed first, then hold full lock.
        for _ in 0..120 {
            c.step(0.8, 0.0, 1.0 / 60.0, &spec);
        }
        let v = c.speed;
        for _ in 0..900 {
            c.step(0.4, 1.0, 1.0 / 60.0, &spec);
        }
        // Instantaneous curvature is tan(steer) / wheelbase at full lock.
        // Instantaneous curvature is tan(steer) / wheelbase, capped by grip.
        let k = c.yaw_rate(&spec) / c.speed.abs().max(1e-3);
        assert!(
            (k - spec.max_steer.tan() / spec.wheelbase).abs() < 0.01,
            "k {k}"
        );
        assert!(v > 5.0, "test should be at speed, v {v}");
    }

    #[test]
    fn braking_and_reverse_are_bounded() {
        let spec = CarSpec::default();
        let mut c = Car::new(Vec2::ZERO, 0.0);
        for _ in 0..600 {
            c.step(-1.0, 0.0, 1.0 / 60.0, &spec);
        }
        assert!(c.speed >= -spec.max_reverse - 1e-3);
        assert!(c.pos.x < 0.0);
    }

    #[test]
    fn corners_surround_body_center() {
        let spec = CarSpec::default();
        let c = Car::new(vec2(10.0, 20.0), 0.7);
        let corners = c.corners(&spec);
        let center = c.body_center(&spec);
        for k in 0..4 {
            let a = corners[k] - center;
            let b = corners[(k + 1) % 4] - center;
            assert!((a.length() - b.length()).abs() < 0.2);
            assert!(a.cross(b) > 0.0, "corners not in order");
        }
    }
}
