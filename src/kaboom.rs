use nalgebra as na;
use png::HasParameters;

use std::env;
use std::error::Error;
use std::f32;
use std::fs::File;
use std::io::BufWriter;
use std::ops::{Add, Mul, Sub};
use std::path::PathBuf;

type Vec3f = na::Vector3<f32>;

fn clamp(val: f32) -> u8 {
    (255. * val.max(0.).min(1.)) as u8
}

const SPHERE_RADIUS: f32 = 1.5;
const NOISE_AMPLITUDE: f32 = 1.0;

fn lerp<T>(v0: T, v1: T, t: f32) -> T
where
    T: Copy + Add<Output = T> + Sub<Output = T> + Mul<f32, Output = T>,
{
    v0 + (v1 - v0) * t.min(1.).max(0.)
}

fn hash(n: f32) -> f32 {
    let x = n.sin() * 43_758.547;
    x - x.floor()
}

fn noise(x: &Vec3f) -> f32 {
    let p = Vec3f::new(x.x.floor(), x.y.floor(), x.z.floor());
    let mut f = Vec3f::new(x.x - p.x, x.y - p.y, x.z - p.z);
    f = f * f.dot(&(Vec3f::new(3., 3., 3.) - f * 2.));
    let n = p.dot(&Vec3f::new(1., 57., 113.));

    // lerp(hash(n + 0.), hash(n + 1.), f.x)

    lerp(
        lerp(
            lerp(hash(n + 0.), hash(n + 1.), f.x),
            lerp(hash(n + 57.), hash(n + 58.), f.x),
            f.y,
        ),
        lerp(
            lerp(hash(n + 113.), hash(n + 114.), f.x),
            lerp(hash(n + 170.), hash(n + 171.), f.x),
            f.y,
        ),
        f.z,
    )
}

fn rotate(v: &Vec3f) -> Vec3f {
    Vec3f::new(
        Vec3f::new(0.00, 0.80, 0.60).dot(v),
        Vec3f::new(-0.80, 0.36, -0.48).dot(v),
        Vec3f::new(-0.60, -0.48, 0.64).dot(v),
    )
}

fn fractal_brownian_motion(x: &Vec3f) -> f32 {
    let mut p = rotate(x);
    let mut f = 0.;
    f += 0.5000 * noise(&p);
    p *= 2.32;
    f += 0.2500 * noise(&p);
    p *= 3.03;
    f += 0.1250 * noise(&p);
    p *= 2.61;
    f += 0.0625 * noise(&p);
    f / 0.9375
}

fn palette_fire(d: f32) -> Vec3f {
    let yellow = Vec3f::new(1.7, 1.3, 1.0); // note that the color is "hot", i.e. has components >1
    let orange = Vec3f::new(1.0, 0.6, 0.0);
    let red = Vec3f::new(1.0, 0.0, 0.0);
    let darkgray = Vec3f::new(0.2, 0.2, 0.2);
    let gray = Vec3f::new(0.4, 0.4, 0.4);

    let x = d.max(0.).min(1.);
    if x < 0.25 {
        lerp(gray, darkgray, x * 4.)
    } else if x < 0.5 {
        lerp(darkgray, red, x * 4. - 1.)
    } else if x < 0.75 {
        lerp(red, orange, x * 4. - 2.)
    } else {
        lerp(orange, yellow, x * 4. - 3.)
    }
}

fn signed_distance(p: &Vec3f) -> f32 {
    let displacement = -fractal_brownian_motion(&(p * 3.4)) * NOISE_AMPLITUDE;
    p.norm() - (SPHERE_RADIUS + displacement)
}

fn sphere_trace(orig: &Vec3f, dir: &Vec3f) -> Option<Vec3f> {
    // early discard
    if orig.dot(orig) - orig.dot(dir).powi(2) > SPHERE_RADIUS.powi(2) {
        return None;
    }

    let mut pos = *orig;
    for _ in 0..128 {
        let d = signed_distance(&pos);
        if d < 0. {
            return Some(pos);
        }
        pos += dir * (d * 0.1).max(0.01);
    }
    None
}

fn distance_field_normal(pos: &Vec3f) -> Vec3f {
    const EPS: f32 = 0.1;
    let d = signed_distance(pos);
    let nx = signed_distance(&(pos + Vec3f::new(EPS, 0., 0.))) - d;
    let ny = signed_distance(&(pos + Vec3f::new(0., EPS, 0.))) - d;
    let nz = signed_distance(&(pos + Vec3f::new(0., 0., EPS))) - d;
    Vec3f::new(nx, ny, nz).normalize()
}

fn main() -> Result<(), Box<Error>> {
    const WIDTH: usize = 640;
    const HEIGHT: usize = 480;
    const FOV: f32 = f32::consts::PI / 3.;

    let mut framebuffer: Vec<Vec3f> = vec![Vec3f::zeros(); WIDTH * HEIGHT];
    for j in 0..HEIGHT {
        for i in 0..WIDTH {
            let dir_x = (i as f32 + 0.5) - WIDTH as f32 / 2.;
            let dir_y = -(j as f32 + 0.5) + HEIGHT as f32 / 2.;
            let dir_z = -(HEIGHT as f32) / (2. * (FOV / 2.).tan());

            if let Some(ref hit) = sphere_trace(
                &Vec3f::new(0., 0., 3.),
                &Vec3f::new(dir_x, dir_y, dir_z).normalize(),
            ) {
                let noise_level = (SPHERE_RADIUS - hit.norm()) / NOISE_AMPLITUDE;
                // one light is placed to (10,10,10)
                let light_dir = (Vec3f::new(10., 10., 10.) - hit).normalize();
                let light_intensity = (light_dir.dot(&distance_field_normal(hit))).max(0.4);
                framebuffer[i + WIDTH * j] =
                    palette_fire((-0.2 + noise_level) * 2.) * light_intensity;
            } else {
                framebuffer[i + WIDTH * j] = Vec3f::new(0.2, 0.7, 0.8);
            }
        }
    }

    save(framebuffer, WIDTH, HEIGHT)
}

fn save(framebuffer: Vec<Vec3f>, width: usize, height: usize) -> Result<(), Box<Error>> {
    let path = PathBuf::from(env::args().nth(1).unwrap());
    let file = File::create(path)?;
    let w = BufWriter::new(file);

    let mut encoder = png::Encoder::new(w, width as u32, height as u32);
    encoder.set(png::ColorType::RGBA).set(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;

    let mut data = Vec::with_capacity(width * height);
    for v in framebuffer {
        data.push(clamp(v[0]));
        data.push(clamp(v[1]));
        data.push(clamp(v[2]));
        data.push(255);
    }
    writer.write_image_data(&data)?;
    Ok(())
}
