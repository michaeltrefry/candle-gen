//! Candle (Windows/CUDA) FLUX.2-klein **single-file → diffusers** converter (sc-7459, epic 6564
//! story 3) — the candle twin of `mlx_gen_flux2::convert`'s `convert_and_assemble` (sc-3136).
//!
//! Community FLUX.2-klein fine-tunes such as `wikeeyang/Flux2-Klein-9B-True-V2` (the `flux2_klein_9b_true_v2`
//! variant) ship the transformer ONLY, as a single flat `.safetensors` in the original (ComfyUI/BFL)
//! key convention — no diffusers subfolders, no text-encoder / VAE. The candle `flux2_klein_9b`
//! loader ([`crate::transformer::Flux2Transformer::new`]) consumes the **diffusers** key tree, so the
//! on-disk tensors must already be in diffusers convention before a true_v2 snapshot can load. This
//! module reproduces, in candle, the exact key remap the MLX converter (and diffusers'
//! `convert_flux2_transformer_checkpoint_to_diffusers`) applies:
//!
//!   * key renames (`img_in` → `x_embedder`, `*.lin` → `*.linear`, …),
//!   * double-block fused `qkv` `[3·d, d]` row-split into `to_q`/`to_k`/`to_v` (img stream) and
//!     `add_q_proj`/`add_k_proj`/`add_v_proj` (txt stream),
//!   * single-block `linear1`/`linear2` → `to_qkv_mlp_proj`/`to_out` (1:1 — diffusers keeps the
//!     single block fused),
//!   * `final_layer.adaLN_modulation.1` → `norm_out.linear` WITH a **scale/shift swap**: BFL packs
//!     `(shift, scale)`; diffusers/this crate expect `(scale, shift)`. This one swap is load-bearing —
//!     that tensor modulates every output patch, so getting it wrong corrupts the whole image with a
//!     periodic weave (mlx sc-2220).
//!
//! then assembles a complete local diffusers model dir by borrowing the untouched VAE / text encoder /
//! tokenizer / scheduler from an already-installed base FLUX.2-klein-9B snapshot.
//!
//! **Pure structural transform.** The renames + `qkv` row-split + the adaLN half-swap are all
//! contiguous-slice memory ops (no arithmetic), so the tensors' dtype (bf16) is preserved bit-exactly
//! — candle's CPU `narrow`/`cat`/`contiguous` just reshape the buffer. No GPU, no quantization (klein
//! loads dense; the dev pre-quant path is a separate concern).
//!
//! **Borrowing on Windows.** Unlike the MLX converter (macOS, absolute symlinks), the candle box is
//! Windows: directory symlinks need privilege AND later fail to read with `ERROR_UNTRUSTED_MOUNT_POINT`
//! (the same defect the worker's `downloads.rs` documents for HF-cache symlinks). So the borrowed
//! components are **hardlinked** file-by-file (recreating the dir tree) — no privilege, no reparse-point
//! read defect, and no multi-GB duplication on the same volume — with a copy fallback for cross-volume.
//! On unix the borrow is an absolute symlink, matching MLX.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_gen::candle_core::safetensors::MmapedSafetensors;
// Use candle_core's `Error`/`Result` (not candle-gen's `CandleError`) throughout: candle_core::Error
// carries `From<std::io::Error>` (the `Io` variant) + a `Msg` constructor, so the converter's many
// filesystem `?` calls just work and a `candle_core::Error` flows up to the worker (which maps it to a
// String). candle-gen's `CandleError` has no `From<io::Error>`, so it would force a `.map_err` per fs call.
use candle_gen::candle_core::{Device, Error, Result, Tensor};

/// Borrowed-from-base subdirs the candle [`crate`] loader consumes (`text_encoder` / `vae` /
/// `tokenizer`). A transformer-only fine-tune does not touch these, so taking them from the installed
/// base klein-9B is correct.
const BORROWED_SUBDIRS_REQUIRED: &[&str] = &["vae", "text_encoder", "tokenizer"];
/// Borrowed-from-base subdirs that complete the diffusers snapshot but are NOT read by the candle
/// loader (the scheduler config is applied in-code); borrowed when present, skipped when absent so a
/// base snapshot laid out slightly differently still converts.
const BORROWED_SUBDIRS_OPTIONAL: &[&str] = &["scheduler"];
/// Borrowed-from-base top-level files (copied as real files — small, and must survive the worker's
/// temp→final atomic rename).
const BORROWED_FILES: &[&str] = &["model_index.json"];

/// Top-level (non-block) direct renames: original → diffusers.
const TOP_RENAMES: &[(&str, &str)] = &[
    ("img_in.weight", "x_embedder.weight"),
    ("txt_in.weight", "context_embedder.weight"),
    (
        "time_in.in_layer.weight",
        "time_guidance_embed.timestep_embedder.linear_1.weight",
    ),
    (
        "time_in.out_layer.weight",
        "time_guidance_embed.timestep_embedder.linear_2.weight",
    ),
    (
        "double_stream_modulation_img.lin.weight",
        "double_stream_modulation_img.linear.weight",
    ),
    (
        "double_stream_modulation_txt.lin.weight",
        "double_stream_modulation_txt.linear.weight",
    ),
    (
        "single_stream_modulation.lin.weight",
        "single_stream_modulation.linear.weight",
    ),
    ("final_layer.linear.weight", "proj_out.weight"),
];

/// Handled separately (scale/shift swap): `final_layer.adaLN_modulation.1` → `norm_out.linear`.
const ADALN_SOURCE: &str = "final_layer.adaLN_modulation.1.weight";
const ADALN_TARGET: &str = "norm_out.linear.weight";

/// Per-double-block renames (original suffix → diffusers suffix), excluding the fused qkv tensors
/// which are row-split below.
const DOUBLE_RENAMES: &[(&str, &str)] = &[
    ("img_attn.norm.query_norm.weight", "attn.norm_q.weight"),
    ("img_attn.norm.key_norm.weight", "attn.norm_k.weight"),
    ("img_attn.proj.weight", "attn.to_out.0.weight"),
    ("img_mlp.0.weight", "ff.linear_in.weight"),
    ("img_mlp.2.weight", "ff.linear_out.weight"),
    (
        "txt_attn.norm.query_norm.weight",
        "attn.norm_added_q.weight",
    ),
    ("txt_attn.norm.key_norm.weight", "attn.norm_added_k.weight"),
    ("txt_attn.proj.weight", "attn.to_add_out.weight"),
    ("txt_mlp.0.weight", "ff_context.linear_in.weight"),
    ("txt_mlp.2.weight", "ff_context.linear_out.weight"),
];

/// Fused qkv suffix → `(q, k, v)` target suffixes, per stream.
const DOUBLE_QKV: &[(&str, [&str; 3])] = &[
    (
        "img_attn.qkv.weight",
        ["attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight"],
    ),
    (
        "txt_attn.qkv.weight",
        [
            "attn.add_q_proj.weight",
            "attn.add_k_proj.weight",
            "attn.add_v_proj.weight",
        ],
    ),
];

/// Per-single-block renames (1:1; diffusers keeps the fused single block).
const SINGLE_RENAMES: &[(&str, &str)] = &[
    ("linear1.weight", "attn.to_qkv_mlp_proj.weight"),
    ("linear2.weight", "attn.to_out.weight"),
    ("norm.query_norm.weight", "attn.norm_q.weight"),
    ("norm.key_norm.weight", "attn.norm_k.weight"),
];

/// The transformer weights filename written into `out/transformer/` (the diffusers convention; the
/// loader reads every `.safetensors` in the dir, so the exact name is cosmetic).
const TRANSFORMER_WEIGHTS: &str = "diffusion_pytorch_model.safetensors";

/// Count the blocks under `prefix` (`max(i)+1` over keys matching `^{prefix}.{i}.…`), the fork's
/// `_count_blocks` — derives the layer count from the checkpoint itself rather than the config.
fn count_blocks<'a>(keys: impl Iterator<Item = &'a str>, prefix: &str) -> usize {
    let pat = format!("{prefix}.");
    let mut max_idx: Option<usize> = None;
    for k in keys {
        let Some(rest) = k.strip_prefix(&pat) else {
            continue;
        };
        let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        // Require a trailing '.' after the index so `double_blocks.10.` parses as 10, not a prefix
        // collision; a key like `double_blocksX` (no match) is already filtered by `strip_prefix`.
        if digits.is_empty() || !rest[digits.len()..].starts_with('.') {
            continue;
        }
        if let Ok(i) = digits.parse::<usize>() {
            max_idx = Some(max_idx.map_or(i, |m| m.max(i)));
        }
    }
    max_idx.map_or(0, |m| m + 1)
}

/// Row-split a `[3·d, …]` tensor into three equal `[d, …]` chunks along axis 0 (the candle twin of
/// `mx.split(t, 3, 0)`). Contiguous slices — no arithmetic, dtype-preserving.
fn chunk3(t: &Tensor) -> Result<[Tensor; 3]> {
    let rows = t.dim(0)?;
    if !rows.is_multiple_of(3) {
        return Err(Error::Msg(format!(
            "fused qkv split expects a row count divisible by 3, got shape {:?}",
            t.dims()
        )));
    }
    let each = rows / 3;
    let q = t.narrow(0, 0, each)?.contiguous()?;
    let k = t.narrow(0, each, each)?.contiguous()?;
    let v = t.narrow(0, 2 * each, each)?.contiguous()?;
    Ok([q, k, v])
}

/// Split a `[2·d, …]` tensor at the midpoint and swap the halves: BFL `(shift, scale)` → diffusers
/// `(scale, shift)`. Load-bearing (mlx sc-2220). Contiguous slices + cat — dtype-preserving.
fn swap_halves(t: &Tensor) -> Result<Tensor> {
    let rows = t.dim(0)?;
    if !rows.is_multiple_of(2) {
        return Err(Error::Msg(format!(
            "adaLN half-swap expects an even row count, got shape {:?}",
            t.dims()
        )));
    }
    let half = rows / 2;
    let first = t.narrow(0, 0, half)?;
    let second = t.narrow(0, half, half)?;
    Tensor::cat(&[&second, &first], 0)?.contiguous()
}

/// Map an original-format FLUX.2-klein transformer tensor set onto the diffusers key set (the candle
/// twin of the fork's `build_target_state_dict`). Pure remapping — renames + qkv row-split + the adaLN
/// half-swap. The produced keys are exactly the base diffusers transformer's keys. Source tensors are
/// loaded lazily from the mmap (and dropped after each op) so only the produced map is held resident.
fn build_target_state_dict(src: &MmapedSafetensors) -> Result<HashMap<String, Tensor>> {
    let cpu = Device::Cpu;
    let names: Vec<String> = src.tensors().into_iter().map(|(name, _)| name).collect();
    let load = |name: &str| -> Result<Tensor> {
        src.load(name, &cpu)
            .map_err(|e| Error::Msg(format!("flux2 convert: source is missing `{name}`: {e}")))
    };

    let mut out: HashMap<String, Tensor> = HashMap::new();
    for (s, d) in TOP_RENAMES {
        out.insert((*d).to_string(), load(s)?);
    }
    out.insert(ADALN_TARGET.to_string(), swap_halves(&load(ADALN_SOURCE)?)?);

    let n_double = count_blocks(names.iter().map(String::as_str), "double_blocks");
    for i in 0..n_double {
        let (s, d) = (
            format!("double_blocks.{i}"),
            format!("transformer_blocks.{i}"),
        );
        for (src_suffix, [q, k, v]) in DOUBLE_QKV {
            let [tq, tk, tv] = chunk3(&load(&format!("{s}.{src_suffix}"))?)?;
            out.insert(format!("{d}.{q}"), tq);
            out.insert(format!("{d}.{k}"), tk);
            out.insert(format!("{d}.{v}"), tv);
        }
        for (src_suffix, dst_suffix) in DOUBLE_RENAMES {
            out.insert(
                format!("{d}.{dst_suffix}"),
                load(&format!("{s}.{src_suffix}"))?,
            );
        }
    }

    let n_single = count_blocks(names.iter().map(String::as_str), "single_blocks");
    for i in 0..n_single {
        let (s, d) = (
            format!("single_blocks.{i}"),
            format!("single_transformer_blocks.{i}"),
        );
        for (src_suffix, dst_suffix) in SINGLE_RENAMES {
            out.insert(
                format!("{d}.{dst_suffix}"),
                load(&format!("{s}.{src_suffix}"))?,
            );
        }
    }

    Ok(out)
}

/// The `.safetensors` shards in a transformer dir (sorted).
fn safetensors_shards(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut shards: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors"))
        .collect();
    shards.sort();
    Ok(shards)
}

/// Hard guard: the produced key set + shapes must exactly match the base klein diffusers transformer
/// (the ground-truth layout the loader consumes). Catches a botched remap (missing / extra / wrong-shape
/// keys) at convert time rather than as garbage at generate time. Header-only read of the base shards.
fn validate_against_base(
    produced: &HashMap<String, Tensor>,
    base_transformer_dir: &Path,
) -> Result<()> {
    let shards = safetensors_shards(base_transformer_dir)?;
    if shards.is_empty() {
        return Err(Error::Msg(format!(
            "no base transformer safetensors in {}",
            base_transformer_dir.display()
        )));
    }
    // SAFETY: mmap of read-only weight files (header parse only; we never `.load` the bodies here).
    let base_st = unsafe { MmapedSafetensors::multi(&shards)? };
    let base: HashMap<String, Vec<usize>> = base_st
        .tensors()
        .into_iter()
        .map(|(name, view)| (name, view.shape().to_vec()))
        .collect();

    let mut missing: Vec<&String> = base.keys().filter(|k| !produced.contains_key(*k)).collect();
    let mut extra: Vec<&String> = produced.keys().filter(|k| !base.contains_key(*k)).collect();
    let mut bad_shape: Vec<&String> = produced
        .iter()
        .filter(|(k, v)| base.get(*k).is_some_and(|b| b.as_slice() != v.dims()))
        .map(|(k, _)| k)
        .collect();
    if missing.is_empty() && extra.is_empty() && bad_shape.is_empty() {
        return Ok(());
    }
    missing.sort();
    extra.sort();
    bad_shape.sort();
    Err(Error::Msg(format!(
        "flux2 convert validation FAILED vs base transformer: {} missing, {} extra, {} shape mismatch. \
         missing={:?} extra={:?} shape={:?}",
        missing.len(),
        extra.len(),
        bad_shape.len(),
        &missing[..missing.len().min(5)],
        &extra[..extra.len().min(5)],
        &bad_shape[..bad_shape.len().min(5)],
    )))
}

/// Remove an existing path (file, symlink, or directory) if present, so a re-convert is idempotent.
fn remove_if_exists(path: &Path) -> Result<()> {
    // `symlink_metadata` does not follow the link, so a dangling symlink is still detected.
    match std::fs::symlink_metadata(path) {
        Ok(meta) if meta.is_dir() => std::fs::remove_dir_all(path)?,
        Ok(_) => std::fs::remove_file(path)?,
        Err(ref e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(e.into()),
    }
    Ok(())
}

/// Borrow a base component directory into `dst`. unix: an absolute symlink (matches MLX). windows: a
/// hardlinked file tree (no privilege, no reparse-point read defect, no duplication on the same volume),
/// copy fallback for cross-volume.
fn borrow_dir(src: &Path, dst: &Path) -> Result<()> {
    remove_if_exists(dst)?;
    #[cfg(unix)]
    {
        let canonical = std::fs::canonicalize(src)?;
        std::os::unix::fs::symlink(&canonical, dst)?;
        Ok(())
    }
    #[cfg(windows)]
    {
        link_tree(src, dst)
    }
    #[cfg(not(any(unix, windows)))]
    {
        link_tree(src, dst)
    }
}

/// Recreate `src`'s directory tree under `dst`, hardlinking each (canonicalized — HF-cache files are
/// themselves symlinks/hardlinks to `blobs/`) file; copy fallback when a hardlink can't be made
/// (e.g. cross-volume). Used on windows (and any non-unix target).
#[cfg(not(unix))]
fn link_tree(src: &Path, dst: &Path) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            link_tree(&from, &to)?;
        } else {
            let canonical = std::fs::canonicalize(&from)?;
            if std::fs::hard_link(&canonical, &to).is_err() {
                std::fs::copy(&canonical, &to)?;
            }
        }
    }
    Ok(())
}

/// Convert `source_file` (an original single-file FLUX.2-klein transformer in BFL convention) into
/// `out_dir` as a complete diffusers model dir, borrowing the VAE / text encoder / tokenizer /
/// scheduler from `base_dir` (an installed base FLUX.2-klein-9B diffusers snapshot). Returns `out_dir`.
/// The result loads directly through the candle [`crate::config::FLUX2_KLEIN_9B_ID`] loader via the
/// worker's `modelPath` seam.
///
/// Candle twin of `mlx_gen_flux2::convert::convert_and_assemble` (sc-3136 / sc-7459). The transformer
/// weights + its `config.json` and `model_index.json` are written as real files (so they survive the
/// worker's temp→final atomic rename); the borrowed component dirs are absolute symlinks (unix) or
/// hardlink trees (windows).
pub fn convert_and_assemble(
    source_file: impl AsRef<Path>,
    base_dir: impl AsRef<Path>,
    out_dir: impl AsRef<Path>,
) -> Result<PathBuf> {
    let source = source_file.as_ref();
    let base = base_dir.as_ref();
    let out = out_dir.as_ref();
    let base_transformer = base.join("transformer");
    if !source.is_file() {
        return Err(Error::Msg(format!(
            "flux2 convert: source transformer file not found: {}",
            source.display()
        )));
    }
    if !base_transformer.is_dir() {
        return Err(Error::Msg(format!(
            "flux2 convert: base transformer dir not found: {}",
            base_transformer.display()
        )));
    }

    // SAFETY: mmap of a read-only weight file; standard candle loading path.
    let src = unsafe { MmapedSafetensors::new(source)? };
    let produced = build_target_state_dict(&src)?;
    validate_against_base(&produced, &base_transformer)?;

    let out_transformer = out.join("transformer");
    std::fs::create_dir_all(&out_transformer)?;
    candle_gen::candle_core::safetensors::save(
        &produced,
        out_transformer.join(TRANSFORMER_WEIGHTS),
    )?;
    std::fs::copy(
        base_transformer.join("config.json"),
        out_transformer.join("config.json"),
    )?;

    // Borrow the untouched components from the base klein snapshot.
    for name in BORROWED_FILES {
        let src_path = std::fs::canonicalize(base.join(name))?;
        let dst = out.join(name);
        remove_if_exists(&dst)?;
        std::fs::copy(&src_path, &dst)?;
    }
    for name in BORROWED_SUBDIRS_REQUIRED {
        let src_path = base.join(name);
        if !src_path.is_dir() {
            return Err(Error::Msg(format!(
                "flux2 convert: base component missing: {}",
                src_path.display()
            )));
        }
        borrow_dir(&src_path, &out.join(name))?;
    }
    for name in BORROWED_SUBDIRS_OPTIONAL {
        let src_path = base.join(name);
        if src_path.is_dir() {
            borrow_dir(&src_path, &out.join(name))?;
        }
    }

    Ok(out.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::DType;

    fn ramp(rows: usize, cols: usize, start: f32) -> Tensor {
        let v: Vec<f32> = (0..rows * cols).map(|i| start + i as f32).collect();
        Tensor::from_vec(v, (rows, cols), &Device::Cpu).unwrap()
    }

    #[test]
    fn chunk3_splits_in_qkv_order() {
        // [6, 2] = three [2,2] row chunks: q=rows0-1, k=rows2-3, v=rows4-5.
        let t = ramp(6, 2, 0.0);
        let [q, k, v] = chunk3(&t).unwrap();
        assert_eq!(q.dims(), &[2, 2]);
        assert_eq!(
            q.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![0., 1., 2., 3.]
        );
        assert_eq!(
            k.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![4., 5., 6., 7.]
        );
        assert_eq!(
            v.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![8., 9., 10., 11.]
        );
        // A non-divisible row count is rejected.
        assert!(chunk3(&ramp(5, 2, 0.0)).is_err());
    }

    #[test]
    fn swap_halves_swaps_shift_and_scale() {
        // [4, 2]: BFL (shift=rows0-1, scale=rows2-3) → diffusers (scale, shift).
        let t = ramp(4, 2, 0.0);
        let s = swap_halves(&t).unwrap();
        assert_eq!(s.dims(), &[4, 2]);
        // Now scale (was rows2-3) comes first, then shift (was rows0-1).
        assert_eq!(
            s.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![4., 5., 6., 7., 0., 1., 2., 3.]
        );
        assert!(swap_halves(&ramp(3, 2, 0.0)).is_err());
    }

    /// A minimal but complete fixture: one double + one single block + every top-level key, in BFL
    /// convention, plus a base diffusers snapshot whose transformer has exactly the keys/shapes the
    /// remap should produce. Proves the full key remap + assemble + borrow, and that
    /// `validate_against_base` passes only when the produced layout matches.
    #[test]
    fn convert_and_assemble_remaps_keys_and_borrows() {
        let d = 4usize; // inner width; all 2-D weights are [out, in] over this.
        let tmp = std::env::temp_dir().join(format!("cg_flux2_convert_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let src_dir = tmp.join("src");
        let base = tmp.join("base");
        let out = tmp.join("out");
        std::fs::create_dir_all(&src_dir).unwrap();
        let base_transformer = base.join("transformer");
        std::fs::create_dir_all(&base_transformer).unwrap();

        // --- source single-file (BFL keys) ---
        let mut src: HashMap<String, Tensor> = HashMap::new();
        // Top-level: every TOP_RENAME source (square [d, d] here for simplicity).
        for (s, _) in TOP_RENAMES {
            src.insert((*s).to_string(), ramp(d, d, 1.0));
        }
        // adaLN packs (shift, scale): [2d, d].
        src.insert(ADALN_SOURCE.to_string(), ramp(2 * d, d, 100.0));
        // Double block 0: fused qkv [3d, d] per stream + the renamed leaves.
        src.insert(
            "double_blocks.0.img_attn.qkv.weight".into(),
            ramp(3 * d, d, 10.0),
        );
        src.insert(
            "double_blocks.0.txt_attn.qkv.weight".into(),
            ramp(3 * d, d, 20.0),
        );
        src.insert(
            "double_blocks.0.img_attn.norm.query_norm.weight".into(),
            ramp(1, d, 0.0),
        );
        src.insert(
            "double_blocks.0.img_attn.norm.key_norm.weight".into(),
            ramp(1, d, 0.0),
        );
        src.insert(
            "double_blocks.0.img_attn.proj.weight".into(),
            ramp(d, d, 0.0),
        );
        src.insert(
            "double_blocks.0.img_mlp.0.weight".into(),
            ramp(2 * d, d, 0.0),
        );
        src.insert(
            "double_blocks.0.img_mlp.2.weight".into(),
            ramp(d, 2 * d, 0.0),
        );
        src.insert(
            "double_blocks.0.txt_attn.norm.query_norm.weight".into(),
            ramp(1, d, 0.0),
        );
        src.insert(
            "double_blocks.0.txt_attn.norm.key_norm.weight".into(),
            ramp(1, d, 0.0),
        );
        src.insert(
            "double_blocks.0.txt_attn.proj.weight".into(),
            ramp(d, d, 0.0),
        );
        src.insert(
            "double_blocks.0.txt_mlp.0.weight".into(),
            ramp(2 * d, d, 0.0),
        );
        src.insert(
            "double_blocks.0.txt_mlp.2.weight".into(),
            ramp(d, 2 * d, 0.0),
        );
        // Single block 0.
        src.insert("single_blocks.0.linear1.weight".into(), ramp(3 * d, d, 0.0));
        src.insert("single_blocks.0.linear2.weight".into(), ramp(d, 3 * d, 0.0));
        src.insert(
            "single_blocks.0.norm.query_norm.weight".into(),
            ramp(1, d, 0.0),
        );
        src.insert(
            "single_blocks.0.norm.key_norm.weight".into(),
            ramp(1, d, 0.0),
        );
        let source_file = src_dir.join("model.safetensors");
        candle_gen::candle_core::safetensors::save(&src, &source_file).unwrap();

        // --- base diffusers transformer (the expected produced layout) ---
        let mut base_tf: HashMap<String, Tensor> = HashMap::new();
        for (_, dkey) in TOP_RENAMES {
            base_tf.insert((*dkey).to_string(), ramp(d, d, 0.0));
        }
        base_tf.insert(ADALN_TARGET.to_string(), ramp(2 * d, d, 0.0));
        for q in ["attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight"] {
            base_tf.insert(format!("transformer_blocks.0.{q}"), ramp(d, d, 0.0));
        }
        for q in [
            "attn.add_q_proj.weight",
            "attn.add_k_proj.weight",
            "attn.add_v_proj.weight",
        ] {
            base_tf.insert(format!("transformer_blocks.0.{q}"), ramp(d, d, 0.0));
        }
        base_tf.insert(
            "transformer_blocks.0.attn.norm_q.weight".into(),
            ramp(1, d, 0.0),
        );
        base_tf.insert(
            "transformer_blocks.0.attn.norm_k.weight".into(),
            ramp(1, d, 0.0),
        );
        base_tf.insert(
            "transformer_blocks.0.attn.to_out.0.weight".into(),
            ramp(d, d, 0.0),
        );
        base_tf.insert(
            "transformer_blocks.0.ff.linear_in.weight".into(),
            ramp(2 * d, d, 0.0),
        );
        base_tf.insert(
            "transformer_blocks.0.ff.linear_out.weight".into(),
            ramp(d, 2 * d, 0.0),
        );
        base_tf.insert(
            "transformer_blocks.0.attn.norm_added_q.weight".into(),
            ramp(1, d, 0.0),
        );
        base_tf.insert(
            "transformer_blocks.0.attn.norm_added_k.weight".into(),
            ramp(1, d, 0.0),
        );
        base_tf.insert(
            "transformer_blocks.0.attn.to_add_out.weight".into(),
            ramp(d, d, 0.0),
        );
        base_tf.insert(
            "transformer_blocks.0.ff_context.linear_in.weight".into(),
            ramp(2 * d, d, 0.0),
        );
        base_tf.insert(
            "transformer_blocks.0.ff_context.linear_out.weight".into(),
            ramp(d, 2 * d, 0.0),
        );
        base_tf.insert(
            "single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight".into(),
            ramp(3 * d, d, 0.0),
        );
        base_tf.insert(
            "single_transformer_blocks.0.attn.to_out.weight".into(),
            ramp(d, 3 * d, 0.0),
        );
        base_tf.insert(
            "single_transformer_blocks.0.attn.norm_q.weight".into(),
            ramp(1, d, 0.0),
        );
        base_tf.insert(
            "single_transformer_blocks.0.attn.norm_k.weight".into(),
            ramp(1, d, 0.0),
        );
        candle_gen::candle_core::safetensors::save(
            &base_tf,
            base_transformer.join("diffusion_pytorch_model.safetensors"),
        )
        .unwrap();
        std::fs::write(base_transformer.join("config.json"), b"{}").unwrap();
        // Borrowed components.
        for sub in ["vae", "text_encoder", "tokenizer", "scheduler"] {
            std::fs::create_dir_all(base.join(sub)).unwrap();
            std::fs::write(base.join(sub).join("config.json"), b"{}").unwrap();
        }
        std::fs::write(base.join("model_index.json"), b"{}").unwrap();

        // --- convert ---
        let result = convert_and_assemble(&source_file, &base, &out).unwrap();
        assert_eq!(result, out);

        // Produced transformer loads + has EXACTLY the base key set.
        let produced = candle_gen::candle_core::safetensors::load(
            out.join("transformer").join(TRANSFORMER_WEIGHTS),
            &Device::Cpu,
        )
        .unwrap();
        let mut got: Vec<&String> = produced.keys().collect();
        let mut want: Vec<&String> = base_tf.keys().collect();
        got.sort();
        want.sort();
        assert_eq!(
            got, want,
            "produced key set must equal the base diffusers transformer"
        );

        // The qkv split is in q/k/v order: to_q == first third of the source fused img qkv.
        let to_q = produced["transformer_blocks.0.attn.to_q.weight"]
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let expect_q: Vec<f32> = (0..d * d).map(|i| 10.0 + i as f32).collect();
        assert_eq!(to_q, expect_q);

        // The adaLN half-swap landed: norm_out.linear first half == source second half (scale).
        let norm_out = produced[ADALN_TARGET]
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        // source adaLN ramps from 100.0 over [2d, d]; second half starts at 100.0 + d*d.
        assert_eq!(norm_out[0], 100.0 + (d * d) as f32);

        // Borrowed components are present + readable in the converted dir.
        for sub in ["vae", "text_encoder", "tokenizer", "scheduler"] {
            assert!(
                out.join(sub).join("config.json").is_file(),
                "{sub} borrowed"
            );
        }
        assert!(out.join("model_index.json").is_file());
        assert!(out.join("transformer").join("config.json").is_file());

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
