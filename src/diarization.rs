//! Online speaker diarization using WeSpeaker ONNX embeddings.
//!
//! # Pipeline (per VAD segment)
//!
//! ```text
//! i16 samples (16 kHz)
//!   → knf-rs Kaldi fbank (80-bin mel, utterance CMVN)
//!   → WeSpeaker ONNX  [1, T, 80] → [1, 256]
//!   → L2 normalize
//!   → online cosine clustering  → "SPEAKER_00" / "SPEAKER_01" / …
//! ```
//!
//! # Clustering
//!
//! Online variant of the agglomerative approach validated in `exp_g_diarize_agglomerative.rs`:
//! - Each new embedding is L2-normalised before comparison.
//! - Closest speaker centroid is found by cosine distance.
//! - If `dist < threshold` (default 0.9): assign to that speaker, update centroid
//!   via weighted mean + re-normalise.
//! - Otherwise: register as a new speaker.
//!
//! This achieves DER≈10.5% on VoxConverse sample 11 (60 s, 2 speakers)
//! vs. DER=100% with the unfixed `pyannote-rs` iterator.
//!
//! # Bug notes
//!
//! `pyannote_rs::get_segments()` has an iterator bug that returns 0 segments
//! when a speaker's speech crosses a 10-second window boundary.  We do **not**
//! call that function; we use `EmbeddingExtractor` only.

use std::path::Path;

/// Model download URL (GitHub releases, pyannote-rs v0.1.0).
pub const WESPEAKER_URL: &str =
    "https://github.com/thewh1teagle/pyannote-rs/releases/download/v0.1.0/wespeaker-voxceleb-resnet34-LM.onnx";

/// Online speaker diarization engine.
pub struct DiarizationEngine {
    emb_extractor: pyannote_rs::EmbeddingExtractor,
    /// L2-normalised centroid per identified speaker.
    speaker_centroids: Vec<Vec<f32>>,
    /// Number of segments assigned to each speaker (for weighted centroid update).
    speaker_counts: Vec<usize>,
    /// Cosine distance threshold for speaker identity.
    /// Segments within this distance are considered the same speaker.
    threshold: f32,
}

// EmbeddingExtractor wraps an ORT session which is Send.
unsafe impl Send for DiarizationEngine {}

impl DiarizationEngine {
    /// Load the WeSpeaker ONNX embedding model from `model_path`.
    pub fn new(model_path: &Path) -> Result<Self, String> {
        let path_str = model_path
            .to_str()
            .ok_or("Diarization model path is not valid UTF-8")?;
        let emb_extractor = pyannote_rs::EmbeddingExtractor::new(path_str)
            .map_err(|e| format!("Failed to load WeSpeaker model: {e}"))?;
        Ok(Self {
            emb_extractor,
            speaker_centroids: Vec::new(),
            speaker_counts: Vec::new(),
            threshold: 0.9,
        })
    }

    /// Assign a speaker label to a 16 kHz i16 audio segment.
    ///
    /// Returns `""` on failure (embedding error or segment too short) so the
    /// caller can store the segment without a speaker label rather than crashing.
    /// Returns `"SPEAKER_00"`, `"SPEAKER_01"`, … on success.
    pub fn process_segment(&mut self, samples_i16: &[i16]) -> String {
        // Segments shorter than 400 samples cause inf/NaN in the log-mel
        // filterbank (frame length = 400 samples at 16 kHz).
        if samples_i16.len() < 400 {
            tracing::debug!(
                "[diarization] segment too short ({} samples) — skipping",
                samples_i16.len()
            );
            return String::new();
        }

        let raw_emb: Vec<f32> = match self.emb_extractor.compute(samples_i16) {
            Ok(iter) => iter.collect(),
            Err(e) => {
                tracing::warn!("[diarization] embedding failed: {e}");
                return String::new();
            }
        };

        if raw_emb.is_empty() {
            return String::new();
        }

        // L2-normalise: WeSpeaker embeddings are NOT unit vectors (norms 18–434).
        // Without normalisation, large-norm embeddings dominate centroid updates,
        // causing same-speaker segments to be split into multiple clusters.
        let emb = l2_normalize(&raw_emb);

        // Find the closest existing speaker centroid.
        let (best_id, best_dist) = self
            .speaker_centroids
            .iter()
            .enumerate()
            .map(|(id, centroid)| (id, cosine_dist(&emb, centroid)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((usize::MAX, f32::MAX));

        let speaker_id = if best_dist < self.threshold {
            // Assign to existing speaker; update centroid with weighted mean.
            let n = self.speaker_counts[best_id] as f32;
            let new_centroid: Vec<f32> = self.speaker_centroids[best_id]
                .iter()
                .zip(emb.iter())
                .map(|(c, e)| (c * n + e) / (n + 1.0))
                .collect();
            // Re-normalise after merge to prevent centroid drift.
            self.speaker_centroids[best_id] = l2_normalize(&new_centroid);
            self.speaker_counts[best_id] += 1;
            tracing::debug!(
                "[diarization] segment → SPEAKER_{:02} (cosine_dist={:.3})",
                best_id,
                best_dist
            );
            best_id
        } else {
            // New speaker.
            let id = self.speaker_centroids.len();
            self.speaker_centroids.push(emb);
            self.speaker_counts.push(1);
            tracing::debug!(
                "[diarization] new speaker SPEAKER_{:02} (closest_dist={:.3})",
                id,
                best_dist
            );
            id
        };

        format!("SPEAKER_{:02}", speaker_id)
    }

    /// Reset speaker memory for a new meeting session.
    /// The model stays loaded (expensive to reload).
    pub fn reset(&mut self) {
        self.speaker_centroids.clear();
        self.speaker_counts.clear();
        tracing::debug!("[diarization] speaker state reset for new session");
    }

    /// Number of distinct speakers identified so far in this session.
    pub fn speaker_count(&self) -> usize {
        self.speaker_centroids.len()
    }
}

/// Convert f32 PCM samples in [-1.0, 1.0] to i16 PCM (required by WeSpeaker extractor).
pub fn f32_to_i16(samples: &[f32]) -> Vec<i16> {
    samples
        .iter()
        .map(|&s| (s.clamp(-1.0, 1.0) * 32767.0) as i16)
        .collect()
}

// ── Math helpers ───────────────────────────────────────────────────────────────

fn cosine_dist(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    // a and b are L2-normalised, so na ≈ nb ≈ 1.0.
    // Still compute norm to guard against floating-point drift.
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    1.0 - dot / (na * nb + 1e-9)
}

fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < 1e-9 {
        v.to_vec()
    } else {
        v.iter().map(|x| x / norm).collect()
    }
}
