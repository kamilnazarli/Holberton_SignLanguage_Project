/**
 * Dynamic Sign Language Recognition Module for Azerbaijani Sign Language (AzSLD).
 *
 * Implements client-side ONNX Runtime Web inference for the 7 dynamic letters:
 * C, D, Ö, Ş, Ü, Y, Z
 *
 * Features:
 * - 20-frame rolling landmark buffer (20 x 63 = 1260 features)
 * - Exact MediaPipe 21-landmark normalization (wrist-origin, middle-MCP scale, left-hand mirroring)
 * - Motion energy gating (differentiates stationary static letters from dynamic trajectories)
 * - Temporal stability smoothing (prevents per-frame prediction flicker)
 * - Preserves the exact 7-class label mapping: ["C", "D", "Ö", "Ş", "Ü", "Y", "Z"]
 */

export const DYNAMIC_CLASSES = ['C', 'D', 'Ö', 'Ş', 'Ü', 'Y', 'Z'];

// Landmark indices matching MediaPipe Hands
const LM = {
  WRIST: 0,
  MIDDLE_MCP: 9,
};

/**
 * Normalizes 21 MediaPipe hand landmarks to match Python dynamic_dataset.py exactly:
 * 1. Translates wrist to (0, 0, 0)
 * 2. Scales by distance to Middle MCP (landmark 9)
 * 3. Mirrors X if Left hand (canonical Right hand representation)
 * 4. Flattens into Float32Array(63) in order: [x0,y0,z0, x1,y1,z1, ... x20,y20,z20]
 *
 * @param {Array<{x: number, y: number, z: number}>} rawLandmarks 21 MediaPipe landmarks
 * @param {boolean} mirrorX Whether to mirror X coordinate (for left hand detections)
 * @returns {Float32Array} Normalized 63-dimensional feature vector
 */
export function normalizeLandmarks63(rawLandmarks, mirrorX) {
  if (!rawLandmarks || rawLandmarks.length < 21) {
    return new Float32Array(63);
  }

  const wrist = rawLandmarks[LM.WRIST];
  const shifted = new Array(21);
  for (let i = 0; i < 21; i++) {
    const p = rawLandmarks[i];
    shifted[i] = {
      x: p.x - wrist.x,
      y: p.y - wrist.y,
      z: (p.z || 0) - (wrist.z || 0),
    };
  }

  const mMcp = shifted[LM.MIDDLE_MCP];
  let scale = Math.sqrt(mMcp.x * mMcp.x + mMcp.y * mMcp.y + mMcp.z * mMcp.z);
  if (scale < 1e-6) scale = 1e-6;

  const out = new Float32Array(63);
  let k = 0;
  for (let i = 0; i < 21; i++) {
    const p = shifted[i];
    const nx = (p.x / scale) * (mirrorX ? -1.0 : 1.0);
    const ny = p.y / scale;
    const nz = p.z / scale;
    out[k++] = nx;
    out[k++] = ny;
    out[k++] = nz;
  }

  return out;
}

/**
 * Computes mean absolute landmark displacement between consecutive normalized frames.
 *
 * @param {Float32Array} curr 63D vector of current frame
 * @param {Float32Array} prev 63D vector of previous frame
 * @returns {number} Motion energy metric
 */
export function calculateMotionEnergy(curr, prev) {
  if (!curr || !prev || curr.length !== 63 || prev.length !== 63) return 0;
  let sum = 0;
  for (let i = 0; i < 63; i++) {
    sum += Math.abs(curr[i] - prev[i]);
  }
  return sum / 63.0;
}

/**
 * Streaming Dynamic Sign Predictor for live webcam inference.
 */
export class DynamicSignPredictor {
  constructor(options = {}) {
    this.modelUrl = options.modelUrl || '/models/dynamic_model.onnx';
    this.sequenceLength = options.sequenceLength || 20;
    this.motionThreshold = options.motionThreshold || 0.038;
    this.minConfidence = options.minConfidence || 0.60;
    this.stabilityWindow = options.stabilityWindow || 3;

    this.session = null;
    this.isModelLoading = false;
    this.modelReady = false;
    this.modelError = null;

    // Rolling 20-frame landmark buffer (each element is Float32Array(63))
    this.landmarkBuffer = [];
    this.recentEnergies = [];
    this.recentPredictions = [];
  }

  /**
   * Initializes ONNX Runtime Web session.
   */
  async loadModel() {
    if (this.session) return this.session;
    if (this.isModelLoading) return null;

    this.isModelLoading = true;
    try {
      if (typeof window.ort === 'undefined') {
        throw new Error('ONNX Runtime Web library (window.ort) is not loaded.');
      }

      // Configure ONNX Web runtime options (CPU / WASM)
      if (window.ort.env && window.ort.env.wasm) {
        window.ort.env.wasm.numThreads = 1;
        window.ort.env.wasm.simd = true;
      }

      this.session = await window.ort.InferenceSession.create(this.modelUrl, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      });

      this.modelReady = true;
      this.modelError = null;
      console.log('AzSL Dynamic ONNX Model loaded successfully from', this.modelUrl);
      return this.session;
    } catch (err) {
      this.modelReady = false;
      this.modelError = err.message;
      console.warn('Dynamic ONNX model failed to load:', err.message);
      return null;
    } finally {
      this.isModelLoading = false;
    }
  }

  /**
   * Resets temporal state buffers.
   */
  reset() {
    this.landmarkBuffer = [];
    this.recentEnergies = [];
    this.recentPredictions = [];
  }

  /**
   * Pushes a new frame's 63D normalized landmarks and returns analysis.
   *
   * @param {Float32Array} landmarks63 63D normalized landmarks of current frame
   * @returns {Promise<{
   *   isMoving: boolean,
   *   motionEnergy: number,
   *   bufferFilled: boolean,
   *   bufferLength: number,
   *   dynamicPrediction: { label: string, confidence: number, candidates: Array<{label: string, confidence: number}> } | null
   * }>}
   */
  async processFrame(landmarks63) {
    if (!landmarks63 || landmarks63.length !== 63) {
      this.reset();
      return {
        isMoving: false,
        motionEnergy: 0,
        bufferFilled: false,
        bufferLength: 0,
        dynamicPrediction: null,
      };
    }

    // Compute motion energy relative to previous frame
    let energy = 0;
    if (this.landmarkBuffer.length > 0) {
      const prev = this.landmarkBuffer[this.landmarkBuffer.length - 1];
      energy = calculateMotionEnergy(landmarks63, prev);
    }

    this.landmarkBuffer.push(landmarks63);
    if (this.landmarkBuffer.length > this.sequenceLength) {
      this.landmarkBuffer.shift();
    }

    this.recentEnergies.push(energy);
    if (this.recentEnergies.length > 10) {
      this.recentEnergies.shift();
    }

    // Average motion energy over recent frames
    const avgEnergy = this.recentEnergies.reduce((a, b) => a + b, 0) / (this.recentEnergies.length || 1);
    const isMoving = avgEnergy >= this.motionThreshold;
    const bufferFilled = this.landmarkBuffer.length >= this.sequenceLength;

    let dynamicPrediction = null;

    // Run inference when buffer is full and model is ready
    if (bufferFilled && this.modelReady && this.session) {
      try {
        dynamicPrediction = await this._runInference();
      } catch (err) {
        console.warn('Dynamic inference error:', err);
      }
    }

    return {
      isMoving,
      motionEnergy: avgEnergy,
      bufferFilled,
      bufferLength: this.landmarkBuffer.length,
      dynamicPrediction,
    };
  }

  /**
   * Internal method to run the ONNX model on the current 20-frame buffer.
   */
  async _runInference() {
    // Construct flattened Float32Array of shape (1, 20, 63) = 1260 values
    const flatSeq = new Float32Array(this.sequenceLength * 63);
    for (let t = 0; t < this.sequenceLength; t++) {
      const frameVec = this.landmarkBuffer[t];
      flatSeq.set(frameVec, t * 63);
    }

    const tensor = new window.ort.Tensor('float32', flatSeq, [1, this.sequenceLength, 63]);
    const feeds = { landmark_sequence: tensor };
    const results = await this.session.run(feeds);

    const outputTensor = results.probabilities || results[Object.keys(results)[0]];
    const probs = outputTensor.data; // Float32Array(7)

    const candidates = DYNAMIC_CLASSES.map((label, i) => ({
      label: label,
      confidence: probs[i],
    })).sort((a, b) => b.confidence - a.confidence);

    const topCandidate = candidates[0];

    // Temporal stability tracking
    this.recentPredictions.push(topCandidate);
    if (this.recentPredictions.length > this.stabilityWindow) {
      this.recentPredictions.shift();
    }

    // Compute smoothed confidence for the top candidate across recent predictions
    const sameLabelMatches = this.recentPredictions.filter((p) => p.label === topCandidate.label);
    const stabilityRatio = sameLabelMatches.length / this.recentPredictions.length;
    const smoothedConfidence = topCandidate.confidence * (0.6 + 0.4 * stabilityRatio);

    return {
      label: topCandidate.label,
      confidence: smoothedConfidence,
      rawConfidence: topCandidate.confidence,
      candidates: candidates,
      probabilities: Object.fromEntries(DYNAMIC_CLASSES.map((c, i) => [c, probs[i]])),
    };
  }
}

if (typeof window !== 'undefined') {
  window.SignaDynamic = {
    DYNAMIC_CLASSES,
    normalizeLandmarks63,
    calculateMotionEnergy,
    DynamicSignPredictor,
  };
}

