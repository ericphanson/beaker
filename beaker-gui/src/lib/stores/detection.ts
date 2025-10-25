import { writable, derived } from "svelte/store";
import type { DetectResponse } from "$lib/types/beaker";

// Detection state
export const detectionResults = writable<DetectResponse | null>(null);
export const isDetecting = writable(false);
export const detectionError = writable<string | null>(null);

// Detection parameters
export const selectedImage = writable<string | null>(null);
export const confidence = writable(0.5);
export const iouThreshold = writable(0.45);
export const selectedClasses = writable<string[]>(["bird"]);
export const showBoundingBox = writable(true);

// Derived stores
export const processingTime = derived(detectionResults, ($results) => {
  if (!$results) return null;
  return `${Math.round($results.processingTimeMs)}ms`;
});

export const detectionCount = derived(detectionResults, ($results) => {
  if (!$results) return 0;
  return $results.detections.length;
});
