//! Gradient (activation) checkpointing (sc-5165) — the candle realization of MLX's `mx.checkpoint`
//! / torch's `torch.utils.checkpoint`, which candle ships **no** primitive for.
//!
//! candle is eager: `loss.backward()` walks an op graph whose nodes keep `Arc`s to their inputs, so
//! the graph holds **every** intermediate activation alive until the backward completes. A full UNet
//! backward must therefore retain the activations of every resnet / attention block at once — the
//! dominant training-memory cost, and what makes LoRA training on a 512² SDXL latent OOM on smaller
//! cards. Gradient checkpointing trades compute for memory: during the forward we keep only the
//! tensors that cross **segment boundaries** (letting each segment's internal activations free), then
//! during the backward we recompute one segment at a time, backprop the incoming cotangent through
//! its freshly-rebuilt local graph, and drop it before moving to the next. Peak activation memory
//! becomes `O(boundary tensors + one segment)` instead of `O(all activations)`, at the cost of one
//! extra forward pass.
//!
//! ## Why this can't be a thin wrapper over `loss.backward()`
//!
//! There is no extension point in candle's autograd to say "recompute this subgraph during backward":
//! `backward()` walks a graph that already exists, and the only custom-op hook ([`CustomOp1::bwd`] et
//! al.) returns a gradient for its tensor *input* only — it can't surface gradients for the adapter
//! `Var`s used *inside* a recomputed block, which are exactly what we train. So checkpointing here is
//! a **manual segmented vector-Jacobian product**: we drive the chain rule segment-by-segment
//! ourselves, calling candle's `backward()` once per segment on a local surrogate.
//!
//! ## The segmented VJP
//!
//! A model is expressed as a chain of [`Segment`]s, each mapping an input **state** — a list of
//! tensors, the live activations *and* skip residuals crossing that boundary — to an output state,
//! using the model's persistent adapter `Var`s internally. The final segment returns a one-element
//! state holding the scalar loss. [`checkpointed_backward`] then:
//!
//! 1. **Stash forward** — run the chain start to end, [`detach`](candle_core::Tensor::detach)ing each
//!    segment's outputs so its internal graph drops; keep only the (detached, storage-shared)
//!    boundary states.
//! 2. **Recompute backward** — from the loss back to the input, for each segment: wrap its stored
//!    input state as fresh leaf `Var`s, recompute the segment (rebuilding the local graph with the
//!    persistent adapter `Var`s inside it), form the surrogate scalar `s = Σₖ ⟨outₖ, cotₖ⟩`, and call
//!    `s.backward()`. Because candle seeds `ds/ds = 1`, the multiply rule sends each `cotₖ` back
//!    through `outₖ` — i.e. `s.backward()` *is* the VJP with cotangent `cot`. The grads w.r.t. the
//!    input `Var`s become the cotangent for the previous segment; the grads w.r.t. the adapter `Var`s
//!    accumulate into the returned [`GradStore`].
//!
//! ### Why the boundary inputs must be `Var`s
//!
//! Wrapping each boundary input as a `Var` is **required for correctness**, not cosmetic. candle's
//! `sorted_nodes` only walks branches that lead to a variable; a frozen-base branch (`base(x)`, which
//! holds no `Var`) is pruned, so its cotangent would never be delivered to `x`. Making `x` itself a
//! leaf `Var` keeps that branch live, so the boundary cotangent is the *full* upstream gradient (base
//! **+** adapter). With that, the segmented result is the chain rule exactly — identical, modulo
//! floating-point reassociation, to a monolithic `loss.backward()` (the `tests` assert this).
//!
//! The returned [`GradStore`] is keyed by the trainable `Var`s, so it drops straight into
//! [`clip_grad_norm`](super::optim::clip_grad_norm) + [`TrainOptimizer::step`](super::optim::TrainOptimizer::step).

use candle_core::backprop::GradStore;
use candle_core::{DType, Result as CandleResult, Tensor, Var};

use crate::{CandleError, Result};

/// A checkpointed segment: maps an input **state** (the activation + skip-residual tensors crossing
/// that boundary) to an output state, using the model's persistent adapter `Var`s internally. Boxed
/// so a trainer can capture the model and conditioning (timestep embedding, text encoder states, …)
/// by reference. The closure is invoked **twice per step** — once in the stash forward, once in the
/// recompute backward — so it must be pure w.r.t. its inputs (no interior mutation of model state).
pub type Segment<'a> = Box<dyn Fn(&[Tensor]) -> CandleResult<Vec<Tensor>> + 'a>;

/// Detach every tensor in a boundary state (share storage, drop the op) so the producing segment's
/// internal graph is free to drop.
fn detach_all(state: &[Tensor]) -> Vec<Tensor> {
    state.iter().map(Tensor::detach).collect()
}

/// Accumulate `grad` into `store` under `var`'s id (`+=`), detaching so no backward-of-backward graph
/// is retained. A given adapter `Var` lives in exactly one segment for the SDXL UNet, but accumulating
/// keeps the primitive correct for any layout where a factor is shared across segments.
fn accumulate(store: &mut GradStore, var: &Var, grad: &Tensor) -> CandleResult<()> {
    let t = var.as_tensor();
    let grad = grad.detach();
    let summed = match store.get(t) {
        Some(prev) => (prev + &grad)?.detach(),
        None => grad,
    };
    store.insert(t, summed);
    Ok(())
}

/// Run a checkpointed forward + backward over `segments`, returning the scalar loss and a
/// [`GradStore`] holding the accumulated gradient of every `Var` in `trainable`.
///
/// `inputs` is the initial differentiable state fed to `segments[0]` (e.g. `[conv_in_output]`);
/// constants the segments need (timestep embedding, encoder states, target noise) are captured by the
/// closures, not threaded here. The **final** segment must return a one-element `[loss]` state holding
/// a scalar. Errors if `segments` is empty, a segment returns an empty state, the final state isn't a
/// single tensor, or a recomputed segment's output arity drifts from the stashed forward.
///
/// The returned `GradStore` is ready for [`clip_grad_norm`](super::optim::clip_grad_norm) and
/// [`TrainOptimizer::step`](super::optim::TrainOptimizer::step) — exactly the store a monolithic
/// `loss.backward()` would hand them, just built one segment at a time.
///
/// This is the cotangent-discarding thin wrapper over [`checkpointed_backward_with_input_grad`]; reach
/// for that variant when the segment chain sits on top of a **retained upstream forward** whose
/// trainable `Var`s need the input-boundary gradient (the Z-Image trainer's pre-main refiner stack).
pub fn checkpointed_backward(
    segments: &[Segment],
    inputs: &[Tensor],
    trainable: &[Var],
) -> Result<(f32, GradStore)> {
    let (loss, grads, _input_cot) =
        checkpointed_backward_with_input_grad(segments, inputs, trainable)?;
    Ok((loss, grads))
}

/// Like [`checkpointed_backward`], but **also** returns the cotangent at the input boundary —
/// `dL/d inputsₖ`, one tensor per element of `inputs` — so a caller can continue the chain rule
/// through a *retained* upstream forward whose trainable `Var`s live **outside** the segment chain.
///
/// This is what the Z-Image DiT trainer needs: it checkpoints only the main `layers` stack (the bulk
/// of the activation working set) yet keeps the pre-main refiner/embedder forward retained so those
/// adapters train via ordinary autograd. Given the returned `dL/d unified`, it forms the surrogate
/// `s = Σₖ ⟨inputₖ_retained, cotₖ⟩` and `s.backward()` to fold the refiner/embedder grads in. The
/// boundary cotangent is the *full* upstream gradient (frozen base **+** adapter — see the
/// "Why the boundary inputs must be `Var`s" note above), so that surrogate is the exact chain rule.
pub fn checkpointed_backward_with_input_grad(
    segments: &[Segment],
    inputs: &[Tensor],
    trainable: &[Var],
) -> Result<(f32, GradStore, Vec<Tensor>)> {
    if segments.is_empty() {
        return Err(CandleError::Msg(
            "checkpointed_backward: at least one segment is required".into(),
        ));
    }

    // 1. Stash forward: keep only detached boundary states; each segment's internal graph drops when
    //    its (non-detached) `out` goes out of scope at the end of the iteration.
    let mut states: Vec<Vec<Tensor>> = Vec::with_capacity(segments.len() + 1);
    states.push(detach_all(inputs));
    for (i, seg) in segments.iter().enumerate() {
        let out = seg(states.last().unwrap())?;
        if out.is_empty() {
            return Err(CandleError::Msg(format!(
                "checkpointed_backward: segment {i} returned an empty state"
            )));
        }
        states.push(detach_all(&out));
    }

    // The final state is the scalar loss. `sum_all` coerces a 1-element tensor to 0-D defensively.
    let final_state = states.last().expect("states has segments.len()+1 entries");
    if final_state.len() != 1 {
        return Err(CandleError::Msg(format!(
            "checkpointed_backward: the final segment must return a 1-tensor [loss] state, got {}",
            final_state.len()
        )));
    }
    let loss = final_state[0]
        .to_dtype(DType::F32)?
        .sum_all()?
        .to_scalar::<f32>()?;

    // 2. Recompute backward. The cotangent of the loss w.r.t. itself is 1.
    let mut master = GradStore::default();
    let mut cot: Vec<Tensor> = final_state
        .iter()
        .map(|t| t.ones_like())
        .collect::<CandleResult<_>>()?;

    for i in (0..segments.len()).rev() {
        // Wrap this segment's stored input state as fresh leaf `Var`s so the full backward path
        // (frozen base + adapter) reaches them — see the module docs.
        let in_vars: Vec<Var> = states[i]
            .iter()
            .map(Var::from_tensor)
            .collect::<CandleResult<_>>()?;
        let in_tensors: Vec<Tensor> = in_vars.iter().map(|v| v.as_tensor().clone()).collect();
        let out = segments[i](&in_tensors)?;
        if out.len() != cot.len() {
            return Err(CandleError::Msg(format!(
                "checkpointed_backward: segment {i} produced {} output(s) on recompute but its \
                 cotangent has {} (the forward was non-deterministic in arity)",
                out.len(),
                cot.len()
            )));
        }

        // Surrogate s = Σₖ ⟨outₖ, cotₖ⟩, in f32 (cotangents are constants → detach). `s.backward()`
        // seeds ds/ds = 1, so the multiply rule sends cotₖ back through outₖ: this is the VJP.
        let mut surrogate: Option<Tensor> = None;
        for (o, g) in out.iter().zip(cot.iter()) {
            let term = (o.to_dtype(DType::F32)? * g.detach().to_dtype(DType::F32)?)?.sum_all()?;
            surrogate = Some(match surrogate {
                None => term,
                Some(s) => (s + term)?,
            });
        }
        let grads = surrogate
            .expect("out is non-empty (checked in the stash pass)")
            .backward()?;

        // The previous boundary's cotangent = grad w.r.t. each input `Var` (absent ⇒ zero). Detach so
        // the recompute graph is free to drop at the end of this iteration.
        cot = in_vars
            .iter()
            .map(|v| match grads.get(v.as_tensor()) {
                Some(g) => Ok(g.detach()),
                None => v.as_tensor().zeros_like(),
            })
            .collect::<CandleResult<_>>()?;

        // Accumulate this segment's adapter-factor grads into the master store.
        for t in trainable {
            if let Some(g) = grads.get(t.as_tensor()) {
                accumulate(&mut master, t, g)?;
            }
        }
        // `out`, `grads`, `in_vars` drop here → this segment's recompute graph frees.
    }

    // After the i=0 iteration, `cot` holds the cotangent at the input boundary (`dL/d inputsₖ`) — the
    // hook a caller uses to continue the chain rule through a retained upstream forward.
    Ok((loss, master, cot))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn t(data: &[f32], shape: (usize, usize)) -> Tensor {
        Tensor::from_vec(data.to_vec(), shape, &Device::Cpu).unwrap()
    }

    fn var(data: &[f32], shape: (usize, usize)) -> Var {
        Var::from_tensor(&t(data, shape)).unwrap()
    }

    fn grad_vec(grads: &GradStore, v: &Var) -> Vec<f32> {
        grads
            .get(v.as_tensor())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }

    /// The segmented VJP must reproduce a monolithic `loss.backward()` exactly (mod float reassoc),
    /// including a **skip residual** (`s`) produced in segment 0 that crosses an untouched boundary
    /// (segment 1) before being merged in the loss segment — the SDXL down→up residual pattern. If the
    /// pass-through cotangent threading were wrong, `Ws`'s gradient (reachable *only* via the skip)
    /// would be wrong.
    #[test]
    fn matches_monolithic_backward_with_skip() {
        let x = t(&[1.0, -2.0, 0.5, 3.0], (1, 4));
        // Shared trainable factors, built once and used by both the monolithic and segmented passes.
        let w0 = var(&[0.1, 0.2, -0.1, 0.0, 0.3, -0.2, 0.05, 0.4], (4, 2)); // main: [4->2]
        let ws = var(&[0.2, -0.3, 0.1, 0.25, -0.15, 0.05, 0.0, 0.3], (4, 2)); // skip: [4->2]
        let w1 = var(&[0.5, -0.1, 0.2, 0.4], (2, 2)); // transform main: [2->2]
        let w2 = var(&[0.3, 0.1, -0.2, 0.6], (2, 2)); // loss head: [2->2]
        let all = [w0.clone(), ws.clone(), w1.clone(), w2.clone()];

        // --- Monolithic reference ---
        let a = x.matmul(w0.as_tensor()).unwrap().tanh().unwrap(); // [1,2]
        let s = x.matmul(ws.as_tensor()).unwrap(); // [1,2] skip
        let b = a.matmul(w1.as_tensor()).unwrap(); // [1,2]
        let merged = (&b + &s).unwrap(); // residual merge
        let loss_mono = merged
            .matmul(w2.as_tensor())
            .unwrap()
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap();
        let loss_mono_val = loss_mono.to_scalar::<f32>().unwrap();
        let g_mono = loss_mono.backward().unwrap();

        // --- Segmented / checkpointed ---
        let segments: Vec<Segment> = vec![
            // seg 0: [x] -> [a, s]; both depend on Vars (a on w0, s on ws).
            Box::new(|st: &[Tensor]| {
                let a = st[0].matmul(w0.as_tensor())?.tanh()?;
                let s = st[0].matmul(ws.as_tensor())?;
                Ok(vec![a, s])
            }),
            // seg 1: [a, s] -> [b, s]; s passes through untouched (the residual crossing a boundary).
            Box::new(|st: &[Tensor]| {
                let b = st[0].matmul(w1.as_tensor())?;
                Ok(vec![b, st[1].clone()])
            }),
            // seg 2 (loss): [b, s] -> [loss]; merge the residual, then the head.
            Box::new(|st: &[Tensor]| {
                let merged = (&st[0] + &st[1])?;
                let loss = merged.matmul(w2.as_tensor())?.sqr()?.sum_all()?;
                Ok(vec![loss])
            }),
        ];
        let (loss_ckpt, g_ckpt) =
            checkpointed_backward(&segments, std::slice::from_ref(&x), &all).unwrap();

        assert!(
            (loss_ckpt - loss_mono_val).abs() < 1e-5,
            "loss mismatch: checkpointed {loss_ckpt} vs monolithic {loss_mono_val}"
        );
        for (name, v) in [("w0", &w0), ("ws", &ws), ("w1", &w1), ("w2", &w2)] {
            let gm = grad_vec(&g_mono, v);
            let gc = grad_vec(&g_ckpt, v);
            for (a, b) in gm.iter().zip(gc.iter()) {
                assert!(
                    (a - b).abs() < 1e-5,
                    "grad mismatch for {name}: monolithic {gm:?} vs checkpointed {gc:?}"
                );
            }
        }
    }

    /// [`checkpointed_backward_with_input_grad`]'s returned input cotangent must equal the monolithic
    /// `dL/d input` — the hook the Z-Image trainer uses to continue the chain rule into its retained
    /// pre-main forward. The input here carries **no** `Var` of its own (it's a plain activation), so
    /// the boundary cotangent must still be the full gradient (the leaf-`Var` wrapping inside the
    /// harness keeps the frozen-base branch live).
    #[test]
    fn input_cotangent_matches_monolithic_input_grad() {
        // x is a Var ONLY in the monolithic reference so we can read dL/dx; the checkpointed pass
        // treats it as a plain input tensor and recovers the same gradient via the returned cotangent.
        let x = var(&[1.0, -2.0, 0.5, 3.0], (1, 4));
        let w0 = var(&[0.1, 0.2, -0.1, 0.0, 0.3, -0.2, 0.05, 0.4], (4, 2));
        let w1 = var(&[0.5, -0.1, 0.2, 0.4], (2, 2));

        // Monolithic: loss = sum((tanh(x·w0)·w1)²); read dL/dx.
        let loss_mono = x
            .as_tensor()
            .matmul(w0.as_tensor())
            .unwrap()
            .tanh()
            .unwrap()
            .matmul(w1.as_tensor())
            .unwrap()
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap();
        let g_mono = loss_mono.backward().unwrap();
        let dldx_mono = grad_vec(&g_mono, &x);

        // Checkpointed: two segments, recover dL/dx from the returned input cotangent.
        let segments: Vec<Segment> = vec![
            Box::new(|st: &[Tensor]| Ok(vec![st[0].matmul(w0.as_tensor())?.tanh()?])),
            Box::new(|st: &[Tensor]| Ok(vec![st[0].matmul(w1.as_tensor())?.sqr()?.sum_all()?])),
        ];
        let (_loss, _grads, input_cot) = checkpointed_backward_with_input_grad(
            &segments,
            std::slice::from_ref(x.as_tensor()),
            &[w0.clone(), w1.clone()],
        )
        .unwrap();
        let dldx_ckpt = input_cot[0]
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        assert_eq!(dldx_mono.len(), dldx_ckpt.len());
        for (a, b) in dldx_mono.iter().zip(dldx_ckpt.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "input cotangent mismatch: monolithic {dldx_mono:?} vs checkpointed {dldx_ckpt:?}"
            );
        }
    }

    /// A degenerate single-segment chain must equal a direct `loss.backward()`.
    #[test]
    fn single_segment_matches_direct_backward() {
        let x = t(&[2.0, -1.0], (1, 2));
        let w = var(&[0.5, 0.1, -0.3, 0.2], (2, 2));
        let direct = x
            .matmul(w.as_tensor())
            .unwrap()
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap();
        let direct_val = direct.to_scalar::<f32>().unwrap();
        let g_direct = direct.backward().unwrap();

        let segments: Vec<Segment> = vec![Box::new(|st: &[Tensor]| {
            Ok(vec![st[0].matmul(w.as_tensor())?.sqr()?.sum_all()?])
        })];
        let (loss, g_ckpt) = checkpointed_backward(
            &segments,
            std::slice::from_ref(&x),
            std::slice::from_ref(&w),
        )
        .unwrap();

        assert!((loss - direct_val).abs() < 1e-6);
        let a = grad_vec(&g_direct, &w);
        let b = grad_vec(&g_ckpt, &w);
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-6, "grad mismatch: {a:?} vs {b:?}");
        }
    }

    /// The final segment must produce a single scalar-loss tensor.
    #[test]
    fn rejects_non_scalar_final_state() {
        let w = var(&[1.0], (1, 1));
        let segments: Vec<Segment> = vec![Box::new(|st: &[Tensor]| {
            Ok(vec![st[0].clone(), st[0].clone()])
        })];
        let err = checkpointed_backward(&segments, &[t(&[1.0], (1, 1))], &[w]).unwrap_err();
        assert!(matches!(err, CandleError::Msg(m) if m.contains("1-tensor [loss]")));
    }
}
