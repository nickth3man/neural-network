//! Tiny vector math shared by simulation, brains and rendering.

use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

pub const fn vec2(x: f32, y: f32) -> Vec2 {
    Vec2 { x, y }
}

impl Vec2 {
    pub const ZERO: Vec2 = vec2(0.0, 0.0);

    #[inline]
    pub fn dot(self, o: Vec2) -> f32 {
        self.x * o.x + self.y * o.y
    }

    #[inline]
    pub fn cross(self, o: Vec2) -> f32 {
        self.x * o.y - self.y * o.x
    }

    #[inline]
    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    #[inline]
    pub fn length_sq(self) -> f32 {
        self.dot(self)
    }

    #[inline]
    pub fn normalized(self) -> Vec2 {
        let l = self.length();
        if l > 1e-9 { self / l } else { Vec2::ZERO }
    }

    #[inline]
    pub fn perp(self) -> Vec2 {
        vec2(-self.y, self.x)
    }

    #[inline]
    pub fn rotated(self, a: f32) -> Vec2 {
        let (s, c) = a.sin_cos();
        vec2(self.x * c - self.y * s, self.x * s + self.y * c)
    }

    #[inline]
    pub fn dist(self, o: Vec2) -> f32 {
        (self - o).length()
    }

    #[inline]
    pub fn lerp(self, o: Vec2, t: f32) -> Vec2 {
        self + (o - self) * t
    }

    #[inline]
    pub fn from_angle(a: f32) -> Vec2 {
        vec2(a.cos(), a.sin())
    }

    pub fn angle(self) -> f32 {
        self.y.atan2(self.x)
    }

    pub fn clamp_len(self, max: f32) -> Vec2 {
        let l = self.length();
        if l > max && l > 1e-9 {
            self * (max / l)
        } else {
            self
        }
    }
}

impl Add for Vec2 {
    type Output = Vec2;
    #[inline]
    fn add(self, o: Vec2) -> Vec2 {
        vec2(self.x + o.x, self.y + o.y)
    }
}

impl Sub for Vec2 {
    type Output = Vec2;
    #[inline]
    fn sub(self, o: Vec2) -> Vec2 {
        vec2(self.x - o.x, self.y - o.y)
    }
}

impl Mul<f32> for Vec2 {
    type Output = Vec2;
    #[inline]
    fn mul(self, s: f32) -> Vec2 {
        vec2(self.x * s, self.y * s)
    }
}

impl Div<f32> for Vec2 {
    type Output = Vec2;
    #[inline]
    fn div(self, s: f32) -> Vec2 {
        vec2(self.x / s, self.y / s)
    }
}

impl Neg for Vec2 {
    type Output = Vec2;
    #[inline]
    fn neg(self) -> Vec2 {
        vec2(-self.x, -self.y)
    }
}

impl AddAssign for Vec2 {
    #[inline]
    fn add_assign(&mut self, o: Vec2) {
        self.x += o.x;
        self.y += o.y;
    }
}

impl SubAssign for Vec2 {
    #[inline]
    fn sub_assign(&mut self, o: Vec2) {
        self.x -= o.x;
        self.y -= o.y;
    }
}

/// Clamp helper that works for both ints and floats.
#[inline]
pub fn clampf(v: f32, lo: f32, hi: f32) -> f32 {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}

#[inline]
pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Shortest signed angular difference `to - from`, wrapped to [-PI, PI].
#[inline]
pub fn angle_delta(from: f32, to: f32) -> f32 {
    let mut d = to - from;
    while d > std::f32::consts::PI {
        d -= std::f32::consts::TAU;
    }
    while d < -std::f32::consts::PI {
        d += std::f32::consts::TAU;
    }
    d
}

/// Intersection of a ray (`origin` + t * `dir`, t >= 0) with segment `a`-`b`.
/// Returns the ray parameter t (distance when dir is unit length), if the hit
/// lies within the segment.
#[inline]
pub fn ray_segment(origin: Vec2, dir: Vec2, a: Vec2, b: Vec2) -> Option<f32> {
    let e = b - a;
    let denom = dir.cross(e);
    if denom.abs() < 1e-7 {
        return None;
    }
    let diff = a - origin;
    let t = diff.cross(e) / denom;
    let u = diff.cross(dir) / denom;
    if t >= 0.0 && (0.0..=1.0).contains(&u) {
        Some(t)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ray_segment_hits_and_misses() {
        let a = vec2(0.0, 0.0);
        let b = vec2(10.0, 0.0);
        // Straight down onto the segment.
        let t = ray_segment(vec2(5.0, -5.0), vec2(0.0, 1.0), a, b).unwrap();
        assert!((t - 5.0).abs() < 1e-4);
        // Aiming away.
        assert!(ray_segment(vec2(5.0, -5.0), vec2(0.0, -1.0), a, b).is_none());
        // Missing beyond the end.
        assert!(ray_segment(vec2(30.0, -5.0), vec2(0.0, 1.0), a, b).is_none());
        // Parallel.
        assert!(ray_segment(vec2(0.0, 1.0), vec2(1.0, 0.0), a, b).is_none());
    }

    #[test]
    fn angle_delta_wraps() {
        let d = angle_delta(3.0, -3.0);
        assert!(d > 0.0 && d < 0.5, "got {d}");
    }
}
