<script lang="ts">
  import { open } from "@tauri-apps/plugin-dialog";
  import { convertFileSrc } from "@tauri-apps/api/core";
  import { detectionApi } from "$lib/api/detection";
  import {
    selectedImage,
    confidence,
    iouThreshold,
    selectedClasses,
    showBoundingBox,
    detectionResults,
    isDetecting,
    detectionError,
    processingTime,
    detectionCount,
  } from "$lib/stores/detection";

  // Available detection classes
  const availableClasses = ["bird", "head", "eye", "beak"];

  async function selectImage() {
    const selected = await open({
      multiple: false,
      filters: [
        {
          name: "Image",
          extensions: ["png", "jpg", "jpeg"],
        },
      ],
    });

    if (selected && typeof selected === "string") {
      selectedImage.set(selected);
      // Clear previous results when new image is selected
      detectionResults.set(null);
      detectionError.set(null);
    }
  }

  async function runDetection() {
    if (!$selectedImage) return;

    isDetecting.set(true);
    detectionError.set(null);

    try {
      const result = await detectionApi.detectObjects({
        imagePath: $selectedImage,
        confidence: $confidence,
        iouThreshold: $iouThreshold,
        classes: $selectedClasses,
        boundingBox: $showBoundingBox,
      });

      detectionResults.set(result);
    } catch (error) {
      detectionError.set(error instanceof Error ? error.message : String(error));
    } finally {
      isDetecting.set(false);
    }
  }

  function toggleClass(className: string) {
    selectedClasses.update((classes) => {
      if (classes.includes(className)) {
        return classes.filter((c) => c !== className);
      } else {
        return [...classes, className];
      }
    });
  }

  // Get image URL for display
  $: imageUrl = $selectedImage ? convertFileSrc($selectedImage) : null;
  $: boundingBoxUrl =
    $detectionResults?.boundingBoxPath
      ? convertFileSrc($detectionResults.boundingBoxPath)
      : null;
</script>

<div class="detection-panel p-6 max-w-7xl mx-auto">
  <h1 class="text-3xl font-bold mb-6">Bird Detection</h1>

  <!-- Controls -->
  <div class="controls mb-6 p-4 bg-white dark:bg-gray-800 rounded-lg shadow">
    <div class="mb-4">
      <button
        onclick={selectImage}
        class="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition"
      >
        Select Image
      </button>
      {#if $selectedImage}
        <span class="ml-4 text-sm text-gray-600 dark:text-gray-300">
          {$selectedImage.split("/").pop()}
        </span>
      {/if}
    </div>

    {#if $selectedImage}
      <div class="grid grid-cols-2 gap-4 mb-4">
        <!-- Confidence slider -->
        <div>
          <label class="block text-sm font-medium mb-2">
            Confidence: {$confidence.toFixed(2)}
          </label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            bind:value={$confidence}
            class="w-full"
          />
        </div>

        <!-- IoU threshold slider -->
        <div>
          <label class="block text-sm font-medium mb-2">
            IoU Threshold: {$iouThreshold.toFixed(2)}
          </label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            bind:value={$iouThreshold}
            class="w-full"
          />
        </div>
      </div>

      <!-- Class checkboxes -->
      <div class="mb-4">
        <label class="block text-sm font-medium mb-2">Detection Classes:</label>
        <div class="flex gap-4">
          {#each availableClasses as className}
            <label class="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={$selectedClasses.includes(className)}
                onchange={() => toggleClass(className)}
                class="w-4 h-4"
              />
              <span class="capitalize">{className}</span>
            </label>
          {/each}
        </div>
      </div>

      <!-- Bounding box checkbox -->
      <div class="mb-4">
        <label class="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            bind:checked={$showBoundingBox}
            class="w-4 h-4"
          />
          <span>Show bounding boxes</span>
        </label>
      </div>

      <!-- Run button -->
      <button
        onclick={runDetection}
        disabled={$isDetecting || $selectedClasses.length === 0}
        class="px-6 py-2 bg-green-600 text-white rounded hover:bg-green-700 transition disabled:bg-gray-400 disabled:cursor-not-allowed"
      >
        {$isDetecting ? "Detecting..." : "Run Detection"}
      </button>
    {/if}
  </div>

  <!-- Error display -->
  {#if $detectionError}
    <div class="error mb-6 p-4 bg-red-100 dark:bg-red-900 text-red-800 dark:text-red-100 rounded-lg">
      <strong>Error:</strong>
      {$detectionError}
    </div>
  {/if}

  <!-- Results -->
  {#if $detectionResults}
    <div class="results mb-6 p-4 bg-white dark:bg-gray-800 rounded-lg shadow">
      <h2 class="text-xl font-semibold mb-4">Results</h2>
      <div class="stats mb-4 flex gap-6 text-sm">
        <div>
          <span class="font-medium">Detections:</span>
          {$detectionCount}
        </div>
        <div>
          <span class="font-medium">Processing Time:</span>
          {$processingTime}
        </div>
        <div>
          <span class="font-medium">Image Size:</span>
          {$detectionResults.inputImgWidth}x{$detectionResults.inputImgHeight}
        </div>
      </div>

      <!-- Detections list -->
      {#if $detectionResults.detections.length > 0}
        <div class="detections-list space-y-2">
          {#each $detectionResults.detections as detection, i}
            <div
              class="detection-item p-3 bg-gray-50 dark:bg-gray-700 rounded"
            >
              <div class="flex justify-between items-center">
                <div>
                  <span class="font-medium capitalize"
                    >{detection.className}</span
                  >
                  <span class="text-sm text-gray-600 dark:text-gray-300 ml-2">
                    Confidence: {(detection.confidence * 100).toFixed(1)}%
                  </span>
                </div>
                <div class="text-sm text-gray-600 dark:text-gray-300">
                  BBox: [{detection.bbox.x.toFixed(0)}, {detection.bbox.y.toFixed(
                    0,
                  )}, {detection.bbox.width.toFixed(0)}, {detection.bbox.height.toFixed(
                    0,
                  )}]
                </div>
              </div>
              {#if detection.cropPath}
                <div class="mt-2">
                  <img
                    src={convertFileSrc(detection.cropPath)}
                    alt="{detection.className} crop {i + 1}"
                    class="h-24 object-contain"
                  />
                </div>
              {/if}
            </div>
          {/each}
        </div>
      {/if}
    </div>
  {/if}

  <!-- Image display -->
  {#if imageUrl}
    <div class="images grid grid-cols-1 md:grid-cols-2 gap-4">
      <!-- Original image -->
      <div class="image-container">
        <h3 class="text-lg font-medium mb-2">Original Image</h3>
        <img
          src={imageUrl}
          alt="Selected"
          class="w-full h-auto rounded-lg shadow"
        />
      </div>

      <!-- Bounding box image -->
      {#if boundingBoxUrl}
        <div class="image-container">
          <h3 class="text-lg font-medium mb-2">With Detections</h3>
          <img
            src={boundingBoxUrl}
            alt="With bounding boxes"
            class="w-full h-auto rounded-lg shadow"
          />
        </div>
      {/if}
    </div>
  {/if}
</div>

<style>
  /* Custom styles if needed */
  input[type="range"] {
    cursor: pointer;
  }
</style>
