// Landmark assets and the hand topology for drawing. The landmarkers themselves run in the
// inference worker (src/inference.worker.ts), so the page does not load MediaPipe.

/** MediaPipe's 21-point hand connections (HandLandmarker.HAND_CONNECTIONS). */
export const HAND_CONNECTIONS: readonly { start: number; end: number }[] = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11],
  [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
].map(([start, end]) => ({ start: start as number, end: end as number }));

/** Absolute URL of a file under the app's base path (workers need absolute URLs). */
export function assetUrl(path: string): string {
  return new URL(`${import.meta.env.BASE_URL}${path}`, document.baseURI).href;
}

export function trackerAssets(): { loader: string; binary: string; hand: string; pose: string } {
  return {
    loader: assetUrl("mediapipe/vision_wasm_internal.js"),
    binary: assetUrl("mediapipe/vision_wasm_internal.wasm"),
    hand: assetUrl("mediapipe/hand_landmarker.task"),
    pose: assetUrl("mediapipe/pose_landmarker_lite.task"),
  };
}
