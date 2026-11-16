import { fileURLToPath } from "node:url";
import { defaultClientConditions } from "vite";
import { defineConfig } from "vitest/config";
import { CROSS_ORIGIN_ISOLATION_HEADERS, contentSecurityPolicy, serviceWorker } from "./config/plugins.ts";

// GitHub Pages serves the app under /edge-vision/; set BASE_PATH=/edge-vision/ for that build.
const base = process.env.BASE_PATH ?? "/";
const repoRoot = fileURLToPath(new URL("..", import.meta.url));

export default defineConfig({
  base,
  plugins: [
    contentSecurityPolicy(),
    serviceWorker(fileURLToPath(new URL("./config/service-worker.template.js", import.meta.url))),
  ],
  resolve: {
    // onnxruntime-web's non-bundled builds load their wasm glue (ort-wasm-simd-threaded*.mjs)
    // as a separate file, which wasm threads and the WebGPU backend need: each pthread worker
    // loads that file by URL. src/ortBackend.ts passes the emitted URLs as wasmPaths.
    conditions: ["onnxruntime-web-use-extern-wasm", ...defaultClientConditions],
  },
  build: {
    target: "es2022",
    chunkSizeWarningLimit: 1500,
    rollupOptions: {
      input: {
        main: fileURLToPath(new URL("./index.html", import.meta.url)),
        bench: fileURLToPath(new URL("./bench.html", import.meta.url)),
      },
    },
  },
  worker: { format: "es" },
  // Pre-bundling would move onnxruntime-web away from its import.meta.url-relative files.
  optimizeDeps: { exclude: ["onnxruntime-web"] },
  server: {
    // features_spec.json lives at the repository root, shared with Python.
    fs: { allow: [repoRoot] },
    headers: CROSS_ORIGIN_ISOLATION_HEADERS,
  },
  preview: { port: 4173, strictPort: true, headers: CROSS_ORIGIN_ISOLATION_HEADERS },
  test: {
    include: ["tests/unit/**/*.test.ts"],
    environment: "node",
  },
});
