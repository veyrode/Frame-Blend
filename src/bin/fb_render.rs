//! Offline renderer built on the shared core.
//!
//! Decodes a source video with ffmpeg to full-range RGB24 (BT.709 matrix, the
//! file's own input range, no resize or transfer conversion), blends every
//! output frame with the core's kernel and writes lossless FFV1 RGB. Optionally
//! writes a near-lossless H.264 viewing copy from that FFV1 file.
//!
//! Output frame n has time `start + n / fps` measured from source frame 0, so
//! its centre in the source is `c = (start + n / fps) * source_fps`.

use std::collections::VecDeque;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Instant;

use frame_blend::blend::blend_u8;
use frame_blend::kernel::{self, Availability, Tap};
use frame_blend::rational::Rational;

const USAGE: &str = "usage: fb_render --input VIDEO --fps RATE --start SECONDS --duration SECONDS --output OUT.mkv
                 [--preview OUT.mp4] [--preview-range pc|tv] [--preview-qp N] [--ffmpeg PATH] [--ffprobe PATH]

RATE, SECONDS: exact decimals or fractions (30, 30000/1001, 11.2, 56/5).
--start is the time of output frame 0 relative to source frame 0.";

struct Args {
    input: PathBuf,
    fps: Rational,
    start: Rational,
    duration: Rational,
    output: PathBuf,
    preview: Option<PathBuf>,
    preview_range: String,
    preview_qp: String,
    ffmpeg: String,
    ffprobe: String,
}

fn parse_args() -> Result<Args, String> {
    let mut raw = std::env::args().skip(1);
    let (mut input, mut fps, mut start, mut duration, mut output) = (None, None, None, None, None);
    let mut preview = None;
    let mut preview_range = "tv".to_string();
    let mut preview_qp = "1".to_string();
    let mut ffmpeg = "ffmpeg".to_string();
    let mut ffprobe = "ffprobe".to_string();
    while let Some(flag) = raw.next() {
        let mut value = || raw.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--input" => input = Some(PathBuf::from(value()?)),
            "--fps" => fps = Some(value()?.parse::<Rational>()?),
            "--start" => start = Some(value()?.parse::<Rational>()?),
            "--duration" => duration = Some(value()?.parse::<Rational>()?),
            "--output" => output = Some(PathBuf::from(value()?)),
            "--preview" => preview = Some(PathBuf::from(value()?)),
            "--preview-range" => preview_range = value()?,
            "--preview-qp" => preview_qp = value()?,
            "--ffmpeg" => ffmpeg = value()?,
            "--ffprobe" => ffprobe = value()?,
            "-h" | "--help" => return Err(USAGE.to_string()),
            other => return Err(format!("unknown argument {other}\n{USAGE}")),
        }
    }
    if preview_range != "pc" && preview_range != "tv" {
        return Err("--preview-range must be pc or tv".into());
    }
    let missing = |name: &str| format!("missing {name}\n{USAGE}");
    Ok(Args {
        input: input.ok_or_else(|| missing("--input"))?,
        fps: fps.ok_or_else(|| missing("--fps"))?,
        start: start.ok_or_else(|| missing("--start"))?,
        duration: duration.ok_or_else(|| missing("--duration"))?,
        output: output.ok_or_else(|| missing("--output"))?,
        preview,
        preview_range,
        preview_qp,
        ffmpeg,
        ffprobe,
    })
}

struct Probe {
    width: usize,
    height: usize,
    rate: Rational,
    frames: i64,
    color_range: String,
    color_transfer: String,
}

fn probe(args: &Args) -> Result<Probe, String> {
    let out = Command::new(&args.ffprobe)
        .args(["-v", "error", "-select_streams", "v:0", "-show_entries"])
        .arg("stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,color_range,color_space,color_transfer")
        .args(["-of", "default=noprint_wrappers=1"])
        .arg(&args.input)
        .output()
        .map_err(|e| format!("cannot run ffprobe: {e}"))?;
    if !out.status.success() {
        return Err(format!("ffprobe failed: {}", String::from_utf8_lossy(&out.stderr)));
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let field = |key: &str| {
        text.lines()
            .find_map(|l| l.strip_prefix(key).and_then(|r| r.strip_prefix('=')))
            .map(|v| v.trim().to_string())
            .ok_or_else(|| format!("ffprobe did not report {key}"))
    };
    let rate: Rational = field("avg_frame_rate")?.parse()?;
    let nominal: Rational = field("r_frame_rate")?.parse()?;
    if rate != nominal || !rate.is_positive() {
        return Err(format!("source is not constant frame rate: avg {rate}, r {nominal}"));
    }
    if field("color_space")? != "bt709" {
        return Err("this renderer decodes BT.709-matrix sources only".into());
    }
    Ok(Probe {
        width: field("width")?.parse().map_err(|_| "bad width")?,
        height: field("height")?.parse().map_err(|_| "bad height")?,
        rate,
        frames: field("nb_frames")?.parse().map_err(|_| "source frame count unavailable")?,
        color_range: field("color_range")?,
        color_transfer: field("color_transfer")?,
    })
}

/// The decode used by the measurement tools: explicit BT.709 matrix, the
/// file's own input range, full-range RGB24 output, no resize.
fn decode_filter(color_range: &str) -> Result<String, String> {
    let in_range = match color_range {
        "pc" => "full",
        "tv" => "limited",
        other => return Err(format!("unsupported color range {other:?}")),
    };
    Ok(format!(
        "scale=iw:ih:flags=accurate_rnd+full_chroma_int+bitexact:in_color_matrix=bt709:out_color_matrix=bt709:\
         in_range={in_range}:out_range=full,format=rgb24"
    ))
}

fn text_weights(taps: &[Tap]) -> String {
    taps.iter().map(|t| format!("{}:{}", t.index, t.weight)).collect::<Vec<_>>().join(" ")
}

fn run(args: Args) -> Result<(), String> {
    let source = probe(&args)?;
    let frame_bytes = source.width * source.height * 3;
    let output_frames = (args.duration * args.fps).floor();
    if output_frames <= 0 {
        return Err("duration covers no output frame".into());
    }
    let available = Availability { first: 0, last: source.frames - 1 };
    let kernels: Vec<Vec<Tap>> = (0..output_frames)
        .map(|n| {
            let time = args.start + Rational::integer(n) / args.fps;
            let center = kernel::center_at_time(time, source.rate);
            kernel::kernel(source.rate, args.fps, center, available).map_err(|e| format!("kernel: {e:?}"))
        })
        .collect::<Result<_, _>>()?;
    let first_needed = kernels.iter().map(|k| k[0].index).min().unwrap();
    let last_needed = kernels.iter().map(|k| k.last().unwrap().index).max().unwrap();
    let ratio = kernel::rate_ratio(source.rate, args.fps).map_err(|e| format!("{e:?}"))?;
    eprintln!(
        "source {}x{} @ {} fps, {} frames; output {} frames @ {} fps; R = {}; source frames {}..={}",
        source.width, source.height, source.rate, source.frames, output_frames, args.fps, ratio, first_needed, last_needed
    );
    for (n, k) in kernels.iter().enumerate().take(ratio.denom().min(8) as usize) {
        eprintln!("frame {n} taps {}", text_weights(k));
    }

    let filter = format!("trim=start_frame={first_needed},{}", decode_filter(&source.color_range)?);
    let frame_count = (last_needed - first_needed + 1).to_string();
    let size = format!("{}x{}", source.width, source.height);
    let rate = args.fps.to_string();
    let mut decoder = Command::new(&args.ffmpeg)
        .args(["-v", "error", "-nostdin", "-i"])
        .arg(&args.input)
        .args(["-map", "0:v:0", "-an", "-sn", "-dn", "-vf", filter.as_str(), "-fps_mode", "passthrough"])
        .args(["-frames:v", frame_count.as_str(), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
        .stdout(Stdio::piped())
        .spawn()
        .map_err(|e| format!("cannot run ffmpeg decoder: {e}"))?;
    let mut encoder = Command::new(&args.ffmpeg)
        .args(["-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24"])
        .args(["-s", size.as_str(), "-r", rate.as_str(), "-i", "pipe:0"])
        .args(["-an", "-c:v", "ffv1", "-level", "3", "-pix_fmt", "bgr0", "-color_range", "pc", "-colorspace", "bt709"])
        .args(["-color_trc", source.color_transfer.as_str(), "-color_primaries", "bt709"])
        .arg(&args.output)
        .stdin(Stdio::piped())
        .spawn()
        .map_err(|e| format!("cannot run ffmpeg encoder: {e}"))?;

    let mut reader = BufReader::with_capacity(frame_bytes * 2, decoder.stdout.take().unwrap());
    let mut writer = BufWriter::with_capacity(frame_bytes * 2, encoder.stdin.take().unwrap());
    let mut window: VecDeque<(i64, Vec<u8>)> = VecDeque::new();
    let mut next_index = first_needed;
    let mut out = vec![0u8; frame_bytes];
    let started = Instant::now();
    let mut blend_seconds = 0.0f64;
    for (n, taps) in kernels.iter().enumerate() {
        let (lo, hi) = (taps[0].index, taps.last().unwrap().index);
        while window.front().is_some_and(|(i, _)| *i < lo) {
            window.pop_front();
        }
        while next_index <= hi {
            let mut frame = vec![0u8; frame_bytes];
            reader.read_exact(&mut frame).map_err(|e| format!("decoder ended at source frame {next_index}: {e}"))?;
            if next_index >= lo {
                window.push_back((next_index, frame));
            }
            next_index += 1;
        }
        let sources: Vec<&[u8]> = taps
            .iter()
            .map(|t| window.iter().find(|(i, _)| *i == t.index).map(|(_, f)| f.as_slice()).ok_or("frame missing from window"))
            .collect::<Result<_, _>>()?;
        let weights: Vec<f32> = taps.iter().map(|t| t.weight.to_f64() as f32).collect();
        let t0 = Instant::now();
        blend_u8(&mut out, &sources, &weights, source.height);
        blend_seconds += t0.elapsed().as_secs_f64();
        writer.write_all(&out).map_err(|e| format!("encoder write failed: {e}"))?;
        if n % 50 == 0 || n as i64 == output_frames - 1 {
            eprintln!("frame {}/{}", n + 1, output_frames);
        }
    }
    drop(writer);
    drop(reader);
    let _ = decoder.kill();
    let _ = decoder.wait();
    let status = encoder.wait().map_err(|e| e.to_string())?;
    if !status.success() {
        return Err("FFV1 encoder failed".into());
    }
    eprintln!(
        "wrote {} in {:.1} s ({:.2} ms/frame CPU blend)",
        args.output.display(),
        started.elapsed().as_secs_f64(),
        1000.0 * blend_seconds / output_frames as f64
    );

    if let Some(preview) = &args.preview {
        // Near-lossless viewing copy: RGB -> BT.709 YUV 4:2:0 in the requested
        // range, x264 High profile at constant QP 1 (QP 0 would need High 4:4:4).
        let yuv = format!(
            "scale=flags=accurate_rnd+full_chroma_int:out_color_matrix=bt709:out_range={},format=yuv420p",
            if args.preview_range == "pc" { "full" } else { "limited" }
        );
        let status = Command::new(&args.ffmpeg)
            .args(["-v", "error", "-y", "-i"])
            .arg(&args.output)
            .args(["-vf", yuv.as_str(), "-c:v", "libx264", "-preset", "slow", "-profile:v", "high", "-qp", args.preview_qp.as_str()])
            .args(["-color_range", args.preview_range.as_str(), "-colorspace", "bt709", "-color_primaries", "bt709"])
            .args(["-color_trc", source.color_transfer.as_str(), "-movflags", "+faststart"])
            .arg(preview)
            .status()
            .map_err(|e| format!("cannot run ffmpeg for the viewing copy: {e}"))?;
        if !status.success() {
            return Err("viewing copy encode failed".into());
        }
        eprintln!("wrote {}", preview.display());
    }
    Ok(())
}

fn main() {
    let result = parse_args().and_then(run);
    if let Err(message) = result {
        eprintln!("{message}");
        std::process::exit(1);
    }
}
