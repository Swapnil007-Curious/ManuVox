import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { defaultClientConditions } from "vite";
import { defineConfig } from "vitest/config";
import { CROSS_ORIGIN_ISOLATION_HEADERS, contentSecurityPolicy, modelVariantAssets, serviceWorker } from "./config/plugins.ts";
import { MODEL_VARIANTS, describeModelVariant, privateModelFiles, resolveModelVariant } from "./model-variant.ts";
import { assertWorkerUrl, describeWorkerUrl, resolveWorkerUrl, workerOrigin, type SiteFile } from "./site-config.ts";

// GitHub Pages serves the app under /edge-vision/; set BASE_PATH=/edge-vision/ for that build.
const base = process.env.BASE_PATH ?? "/";
const repoRoot = fileURLToPath(new URL("..", import.meta.url));
const site = JSON.parse(readFileSync(new URL("./site.config.json", import.meta.url), "utf8")) as SiteFile;
// The translation Worker's URL: VITE_WORKER_URL when non-blank (the Pages workflow passes the
// repository variable WORKER_URL, "" when unset), else site.config.json. Empty: sentences are
// joined on the device ("basic"); with REQUIRE_WORKER_URL=true the build fails instead.
const worker = resolveWorkerUrl(process.env.VITE_WORKER_URL, site.workerUrl);
// The classifier: site.config.json "modelVariant" or VITE_MODEL_VARIANT (web/model-variant.ts). Only
// v2-kaggle may be deployed publicly; v3-citizen's files come from the gitignored web/models/.
const modelVariant = resolveModelVariant(process.env.VITE_MODEL_VARIANT, site.modelVariant);
const variantFiles = privateModelFiles(modelVariant, fileURLToPath(new URL(".", import.meta.url)));
if (process.argv.includes("build")) {
  console.log(describeModelVariant(modelVariant));
  console.log(describeWorkerUrl(worker));
  if (process.env.REQUIRE_WORKER_URL === "true") assertWorkerUrl(worker);
}

export default defineConfig({
  base,
  define: {
    __WORKER_URL__: JSON.stringify(worker.value),
    __MODEL_DIR__: JSON.stringify(MODEL_VARIANTS[modelVariant].dir),
    __MODEL_VARIANT__: JSON.stringify(modelVariant),
  },
  plugins: [
    contentSecurityPolicy(workerOrigin(worker.value)),
    modelVariantAssets(variantFiles),
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
