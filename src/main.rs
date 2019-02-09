use png::HasParameters;

use std::cmp::Ordering;
use std::env;
use std::error::Error;
use std::f32;
use std::fs::File;
use std::io::BufWriter;
use std::ops::{Add, Mul, Neg, Sub};
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, Default)]
struct Vec4f(f32, f32, f32, f32);

#[derive(Debug, Clone, Copy, Default)]
struct Vec3f(f32, f32, f32);

impl Vec3f {
    const ZERO: Vec3f = Vec3f(0., 0., 0.);

    fn norm(&self) -> f32 {
        let norm_square = self.dot(self);
        norm_square.sqrt()
    }

    fn normalize(&self) -> Self {
        let norm = self.norm();
        Self(self.0 / norm, self.1 / norm, self.2 / norm)
    }

    fn dot(&self, other: &Self) -> f32 {
        self.0 * other.0 + self.1 * other.1 + self.2 * other.2
    }
}

impl Neg for &Vec3f {
    type Output = Vec3f;
    fn neg(self) -> Self::Output {
        Vec3f(-self.0, -self.1, -self.2)
    }
}

impl Neg for Vec3f {
    type Output = Vec3f;
    fn neg(self) -> Self::Output {
        -&self
    }
}

impl Add<Vec3f> for &Vec3f {
    type Output = Vec3f;
    fn add(self, other: Vec3f) -> Self::Output {
        Vec3f(self.0 + other.0, self.1 + other.1, self.2 + other.2)
    }
}

impl Add<Vec3f> for Vec3f {
    type Output = Vec3f;
    fn add(self, other: Vec3f) -> Self::Output {
        &self + other
    }
}

impl Sub<&Vec3f> for Vec3f {
    type Output = Self;
    fn sub(self, other: &Vec3f) -> Self::Output {
        Vec3f(self.0 - other.0, self.1 - other.1, self.2 - other.2)
    }
}

impl Sub<&Vec3f> for &Vec3f {
    type Output = Vec3f;
    fn sub(self, other: &Vec3f) -> Self::Output {
        Vec3f(self.0 - other.0, self.1 - other.1, self.2 - other.2)
    }
}

impl Sub<Vec3f> for Vec3f {
    type Output = Vec3f;
    fn sub(self, other: Vec3f) -> Self::Output {
        Vec3f(self.0 - other.0, self.1 - other.1, self.2 - other.2)
    }
}

impl Mul<f32> for &Vec3f {
    type Output = Vec3f;
    fn mul(self, scalar: f32) -> Self::Output {
        Vec3f(scalar * self.0, scalar * self.1, scalar * self.2)
    }
}

impl Mul<f32> for Vec3f {
    type Output = Vec3f;
    fn mul(self, scalar: f32) -> Self::Output {
        &self * scalar
    }
}

fn clap(val: f32) -> u8 {
    (255f32 * val.max(0f32).min(1f32)) as u8
}

#[derive(Debug)]
struct Sphere {
    center: Vec3f,
    radius: f32,
    material: Material,
}

impl Sphere {
    fn new(center: Vec3f, radius: f32, material: Material) -> Self {
        Self {
            center,
            radius,
            material,
        }
    }

    fn ray_intersect(&self, orig: &Vec3f, dir: &Vec3f) -> Option<f32> {
        let l = self.center - orig;
        let tca = l.dot(dir);
        let d2 = l.dot(&l) - tca * tca;
        if self.radius * self.radius < d2 {
            return None;
        }
        let thc = (self.radius * self.radius - d2).sqrt();
        let mut t0 = tca - thc;
        let t1 = tca + thc;
        if t0 < 0. {
            t0 = t1;
        }
        if t0 < 0. {
            return None;
        }
        Some(t0)
    }
}

#[derive(Debug, Clone, Copy)]
struct Material {
    refractive_index: f32,
    albedo: Vec4f,
    diffuse_color: Vec3f,
    specular_exponent: f32,
}

impl Material {
    fn new(
        refractive_index: f32,
        albedo: Vec4f,
        diffuse_color: Vec3f,
        specular_exponent: f32,
    ) -> Self {
        Self {
            refractive_index,
            albedo,
            diffuse_color,
            specular_exponent,
        }
    }
}

impl Default for Material {
    fn default() -> Self {
        Self {
            refractive_index: 1.,
            albedo: Vec4f(1., 0., 0., 0.),
            diffuse_color: Vec3f::ZERO,
            specular_exponent: 0.,
        }
    }
}

struct Intersection {
    dist: f32,
    point: Vec3f,
    normal: Vec3f,
    material: Material,
}

#[derive(Debug)]
struct Light {
    position: Vec3f,
    intensity: f32,
}

impl Light {
    fn new(position: Vec3f, intensity: f32) -> Self {
        Self {
            position,
            intensity,
        }
    }
}

fn reflect(i: &Vec3f, n: &Vec3f) -> Vec3f {
    i - &(n * 2. * i.dot(n))
}

// Snell's law
fn refract(i: &Vec3f, n: &Vec3f, eta_t: f32, eta_i: f32) -> Vec3f {
    let cosi = -i.dot(n).min(1.).max(-1.);
    if cosi < 0. {
        // if the ray comes from the inside the object, swap the air and the media
        refract(i, &-n, eta_i, eta_t)
    } else {
        let eta = eta_i / eta_t;
        let k = 1. - eta * eta * (1. - cosi * cosi);
        // k < 0 = total reflection, no ray to refract. I refract it anyways, this has no physical meaning
        if k < 0. {
            Vec3f(1., 0., 0.)
        } else {
            i * eta + n * (eta * cosi - k.sqrt())
        }
    }
}

fn scene_intersect(orig: &Vec3f, dir: &Vec3f, spheres: &[Sphere]) -> Option<Intersection> {
    let spheres_intersection = spheres
        .iter()
        .filter_map(|s| s.ray_intersect(orig, dir).map(|t| (s, t)))
        .min_by(|(_, t0), (_, t1)| t0.partial_cmp(t1).unwrap_or(Ordering::Equal))
        .map(|(sphere, dist)| {
            let point = orig + dir * dist;
            let normal = (point - &sphere.center).normalize();
            Intersection {
                dist,
                point,
                normal,
                material: sphere.material,
            }
        });

    // checkerboard intersection
    if dir.1.abs() > 1e-3 {
        let dist = -(orig.1 + 4.) / dir.1; // the checkerboard plane has equation y = -4
        let point = orig + dir * dist;
        if dist > 0.
            && point.0.abs() < 10.0
            && point.2 < -10.0
            && point.2 > -30.0
            && spheres_intersection
                .as_ref()
                .map(|i| dist < i.dist)
                .unwrap_or(true)
        {
            let diffuse_color =
                if ((0.5 * point.0 + 1000.) as i32 + (0.5 * point.2) as i32) & 1 == 1 {
                    Vec3f(1., 1., 1.)
                } else {
                    Vec3f(1., 0.7, 0.3)
                };
            let material = Material {
                diffuse_color: diffuse_color * 0.3,
                ..Material::default()
            };
            return Some(Intersection {
                dist,
                point,
                normal: Vec3f(0., 1., 0.),
                material,
            });
        }
    }

    spheres_intersection
}

fn cast_ray(
    orig: &Vec3f,
    dir: &Vec3f,
    spheres: &[Sphere],
    lights: &[Light],
    depth: usize,
) -> Vec3f {
    if depth > 4 {
        return Vec3f(0.2, 0.7, 0.8); // background color
    }

    if let Some(Intersection {
        point,
        normal,
        material,
        ..
    }) = scene_intersect(orig, dir, spheres)
    {
        let reflect_dir = reflect(dir, &normal).normalize();
        let refract_dir = refract(dir, &normal, material.refractive_index, 1.).normalize();
        // offset the original point to avoid occlusion by the object itself
        let reflect_orig = if reflect_dir.dot(&normal) < 0.0 {
            point - normal * 1e-3
        } else {
            point + normal * 1e-3
        };
        let refract_orig = if refract_dir.dot(&normal) < 0.0 {
            point - normal * 1e-3
        } else {
            point + normal * 1e-3
        };

        let reflect_color = cast_ray(&reflect_orig, &reflect_dir, spheres, lights, depth + 1);
        let refract_color = cast_ray(&refract_orig, &refract_dir, spheres, lights, depth + 1);

        let (diffuse_light_intensity, specular_light_intensity) =
            lights.iter().fold((0., 0.), |(diff, spec), light| {
                let light_dir = (light.position - &point).normalize();
                let light_distance = (light.position - &point).norm();

                // checking if the point lies in the shadow of the light
                let shadow_orig = if light_dir.dot(&normal) < 0.0 {
                    point - normal * 1e-3
                } else {
                    point + normal * 1e-3
                };
                if let Some(Intersection {
                    point: shadow_pt, ..
                }) = scene_intersect(&shadow_orig, &light_dir, spheres)
                {
                    if (shadow_pt - shadow_orig).norm() < light_distance {
                        return (diff, spec);
                    }
                }

                (
                    diff + light.intensity * light_dir.dot(&normal).max(0.),
                    spec + (-reflect(&-light_dir, &normal))
                        .dot(&dir)
                        .max(0.)
                        .powf(material.specular_exponent)
                        * light.intensity,
                )
            });

        material.diffuse_color * diffuse_light_intensity * material.albedo.0
            + Vec3f(1., 1., 1.) * specular_light_intensity * material.albedo.1
            + reflect_color * material.albedo.2
            + refract_color * material.albedo.3
    } else {
        Vec3f(0.2, 0.7, 0.8) // background color
    }
}

fn render(spheres: &[Sphere], lights: &[Light]) -> Result<(), Box<Error>> {
    const WIDTH: usize = 1024;
    const HEIGHT: usize = 768;
    const FOV: f32 = f32::consts::PI / 2.;

    let mut framebuffer: Vec<Vec3f> = vec![Vec3f::ZERO; WIDTH * HEIGHT];
    for j in 0..HEIGHT {
        for i in 0..WIDTH {
            let x = (2. * (i as f32 + 0.5) / WIDTH as f32 - 1.) * (FOV / 2.).tan() * WIDTH as f32
                / HEIGHT as f32;
            let y = -(2. * (j as f32 + 0.5) / HEIGHT as f32 - 1.) * (FOV / 2.).tan();
            let dir = Vec3f(x, y, -1.).normalize();
            framebuffer[i + j * WIDTH] = cast_ray(&Vec3f::ZERO, &dir, spheres, lights, 0);
        }
    }

    let path = PathBuf::from(env::args().nth(1).unwrap());
    let file = File::create(path)?;
    let w = BufWriter::new(file);

    let mut encoder = png::Encoder::new(w, WIDTH as u32, HEIGHT as u32);
    encoder.set(png::ColorType::RGBA).set(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;

    let mut data = Vec::new();
    for v in framebuffer {
        data.push(clap(v.0));
        data.push(clap(v.1));
        data.push(clap(v.2));
        data.push(255);
    }
    writer.write_image_data(&data)?;

    Ok(())
}

fn main() -> Result<(), Box<Error>> {
    let ivory = Material::new(1.0, Vec4f(0.6, 0.3, 0.1, 0.0), Vec3f(0.4, 0.4, 0.3), 50.);
    let glass = Material::new(1.5, Vec4f(0.0, 0.5, 0.1, 0.8), Vec3f(0.6, 0.7, 0.8), 125.);
    let red_rubber = Material::new(1.0, Vec4f(0.9, 0.1, 0.0, 0.0), Vec3f(0.3, 0.1, 0.1), 10.);
    let mirror = Material::new(1.0, Vec4f(0.0, 10.0, 0.8, 0.0), Vec3f(1.0, 1.0, 1.0), 1425.);

    let spheres = vec![
        Sphere::new(Vec3f(-3., 0., -16.), 2., ivory),
        Sphere::new(Vec3f(-1.0, -1.5, -12.), 2., glass),
        Sphere::new(Vec3f(1.5, -0.5, -18.), 3., red_rubber),
        Sphere::new(Vec3f(7., 5., -18.), 4., mirror),
    ];

    let lights = vec![
        Light::new(Vec3f(-20., 20., 20.), 1.5),
        Light::new(Vec3f(30., 50., -25.), 1.8),
        Light::new(Vec3f(30., 20., 30.), 1.7),
    ];

    render(&spheres, &lights)
}
