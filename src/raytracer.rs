use jpeg_decoder as jpeg;
use nalgebra as na;
use png::HasParameters;

type Vec3f = na::Vector3<f32>;
type Vec4f = na::Vector4<f32>;

use std::cmp::Ordering;
use std::env;
use std::error::Error;
use std::f32;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};

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
            albedo: Vec4f::new(1., 0., 0., 0.),
            diffuse_color: Vec3f::zeros(),
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

fn offset(point: &Vec3f, normal: &Vec3f, dir: &Vec3f) -> Vec3f {
    if dir.dot(&normal) < 0. {
        point - normal * 1e-3
    } else {
        point + normal * 1e-3
    }
}

fn reflect(i: &Vec3f, n: &Vec3f) -> Vec3f {
    i - n * 2. * i.dot(n)
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
            Vec3f::new(1., 0., 0.)
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
            let normal = (point - sphere.center).normalize();
            Intersection {
                dist,
                point,
                normal,
                material: sphere.material,
            }
        });

    // checkerboard intersection
    if dir.y.abs() > 1e-3 {
        let dist = -(orig.y + 4.) / dir.y; // the checkerboard plane has equation y = -4
        let point = orig + dir * dist;
        if dist > 0.
            && point.x.abs() < 10.
            && point.z < -10.
            && point.z > -30.
            && spheres_intersection
                .as_ref()
                .map(|i| dist < i.dist)
                .unwrap_or(true)
        {
            let diffuse_color =
                if ((0.5 * point.x + 1000.) as i32 + (0.5 * point.z) as i32) & 1 == 1 {
                    Vec3f::new(1., 1., 1.)
                } else {
                    Vec3f::new(1., 0.7, 0.3)
                };
            let material = Material {
                diffuse_color: diffuse_color * 0.3,
                ..Material::default()
            };
            return Some(Intersection {
                dist,
                point,
                normal: Vec3f::new(0., 1., 0.),
                material,
            });
        }
    }

    spheres_intersection
}

fn spherical_coordinates(normal: &Vec3f) -> (f32, f32) {
    let s = normal.z.atan2(normal.x) / (2. * f32::consts::PI) + 0.5;
    let t = normal.y.acos() / f32::consts::PI;
    (s, t)
}

fn cast_ray(
    orig: &Vec3f,
    dir: &Vec3f,
    spheres: &[Sphere],
    lights: &[Light],
    envmap: &Envmap,
    depth: usize,
) -> Vec3f {
    if depth > 4 {
        let (s, t) = spherical_coordinates(&dir.normalize());
        return envmap.texture(s, t);
    }

    if let Some(Intersection {
        ref point,
        ref normal,
        material,
        ..
    }) = scene_intersect(orig, dir, spheres)
    {
        let reflect_dir = reflect(dir, normal).normalize();
        let refract_dir = refract(dir, normal, material.refractive_index, 1.).normalize();
        let reflect_orig = offset(point, normal, &reflect_dir);
        let refract_orig = offset(point, normal, &refract_dir);
        let reflect_color = cast_ray(
            &reflect_orig,
            &reflect_dir,
            spheres,
            lights,
            envmap,
            depth + 1,
        );
        let refract_color = cast_ray(
            &refract_orig,
            &refract_dir,
            spheres,
            lights,
            envmap,
            depth + 1,
        );

        let (diffuse_light_intensity, specular_light_intensity) =
            lights.iter().fold((0., 0.), |(diff, spec), light| {
                let light_dir = (light.position - point).normalize();
                let light_distance = (light.position - point).norm();

                // checking if the point lies in the shadow of the light
                let shadow_orig = offset(point, normal, &light_dir);
                if let Some(Intersection {
                    point: shadow_pt, ..
                }) = scene_intersect(&shadow_orig, &light_dir, spheres)
                {
                    if (shadow_pt - shadow_orig).norm() < light_distance {
                        return (diff, spec);
                    }
                }

                (
                    diff + light.intensity * light_dir.dot(normal).max(0.),
                    spec + (-reflect(&-light_dir, &normal))
                        .dot(&dir)
                        .max(0.)
                        .powf(material.specular_exponent)
                        * light.intensity,
                )
            });

        material.diffuse_color * diffuse_light_intensity * material.albedo[0]
            + Vec3f::new(1., 1., 1.) * specular_light_intensity * material.albedo[1]
            + reflect_color * material.albedo[2]
            + refract_color * material.albedo[3]
    } else {
        let (s, t) = spherical_coordinates(&dir.normalize());
        envmap.texture(s, t)
    }
}

fn clamp(val: f32) -> u8 {
    (255. * val.max(0.).min(1.)) as u8
}

fn render(spheres: &[Sphere], lights: &[Light], envmap: &Envmap) -> Result<(), Box<Error>> {
    const WIDTH: usize = 1024;
    const HEIGHT: usize = 768;
    const FOV: f32 = f32::consts::PI / 2.;

    let mut framebuffer: Vec<Vec3f> = vec![Vec3f::zeros(); WIDTH * HEIGHT];
    for j in 0..HEIGHT {
        for i in 0..WIDTH {
            let dir_x = (i as f32 + 0.5) - WIDTH as f32 / 2.;
            let dir_y = -(j as f32 + 0.5) + HEIGHT as f32 / 2.;
            let dir_z = -HEIGHT as f32 / (2. * (FOV / 2.).tan());

            if let Some(_) = sphere_trace(
                Vec3f::new(0., 0., 3.),
                Vec3f(dir_x, dir_y, dir_z).normalize(),
            ) {
                framebuffer[i * WIDTH * j] = Vec3f::new(1., 1., 1.);
            } else {
                framebuffer[i * WIDTH * j] = Vec3f::new(0.2, 0.7, 0.8);
            }
        }
    }

    let path = PathBuf::from(env::args().nth(1).unwrap());
    let file = File::create(path)?;
    let w = BufWriter::new(file);

    let mut encoder = png::Encoder::new(w, WIDTH as u32, HEIGHT as u32);
    encoder.set(png::ColorType::RGBA).set(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;

    let mut data = Vec::with_capacity(WIDTH * HEIGHT);
    for v in framebuffer {
        data.push(clamp(v[0]));
        data.push(clamp(v[1]));
        data.push(clamp(v[2]));
        data.push(255);
    }
    writer.write_image_data(&data)?;

    Ok(())
}

struct Envmap {
    width: usize,
    height: usize,
    pixels: Vec<u8>,
}

impl Envmap {
    fn load<P: AsRef<Path>>(path: P) -> Result<Self, Box<Error>> {
        let file = File::open(path)?;
        let mut decoder = jpeg::Decoder::new(BufReader::new(file));
        let pixels = decoder.decode()?;
        let metadata = decoder.info().unwrap();
        assert_eq!(
            metadata.width as usize * metadata.height as usize * 3,
            pixels.len()
        );
        Ok(Self {
            width: metadata.width as usize,
            height: metadata.height as usize,
            pixels,
        })
    }

    fn texture(&self, s: f32, t: f32) -> Vec3f {
        let x = (s * self.width as f32) as usize;
        let y = (t * self.height as f32) as usize;
        let idx = (y * self.width + x) * 3;
        Vec3f::new(
            f32::from(self.pixels[idx]) / 255.,
            f32::from(self.pixels[idx + 1]) / 255.,
            f32::from(self.pixels[idx + 2]) / 255.,
        )
    }
}

fn main() -> Result<(), Box<Error>> {
    let ivory = Material::new(
        1.,
        Vec4f::new(0.6, 0.3, 0.1, 0.),
        Vec3f::new(0.4, 0.4, 0.3),
        50.,
    );
    let glass = Material::new(
        1.5,
        Vec4f::new(0., 0.5, 0.1, 0.8),
        Vec3f::new(0.6, 0.7, 0.8),
        125.,
    );
    let red_rubber = Material::new(
        1.,
        Vec4f::new(0.9, 0.1, 0., 0.),
        Vec3f::new(0.3, 0.1, 0.1),
        10.,
    );
    let mirror = Material::new(
        1.,
        Vec4f::new(0., 10., 0.8, 0.),
        Vec3f::new(1., 1., 1.),
        1425.,
    );

    let spheres = vec![
        Sphere::new(Vec3f::new(-3., 0., -16.), 2., ivory),
        Sphere::new(Vec3f::new(-1., -1.5, -12.), 2., glass),
        Sphere::new(Vec3f::new(1.5, -0.5, -18.), 3., red_rubber),
        Sphere::new(Vec3f::new(7., 5., -18.), 4., mirror),
    ];

    let lights = vec![
        Light::new(Vec3f::new(-20., 20., 20.), 1.5),
        Light::new(Vec3f::new(30., 50., -25.), 1.8),
        Light::new(Vec3f::new(30., 20., 30.), 1.7),
    ];

    let envmap = Envmap::load("envmap.jpg")?;

    render(&spheres, &lights, &envmap)
}
