import { invoke } from "@tauri-apps/api/core";
import type { DetectRequest, DetectResponse } from "$lib/types/beaker";

export const detectionApi = {
  async detectObjects(request: DetectRequest): Promise<DetectResponse> {
    return await invoke<DetectResponse>("detect_objects", { request });
  },
};
