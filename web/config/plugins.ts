// Build-only Vite plugins: a Content-Security-Policy that confines every network request to this
// origin, and the generated service worker that precaches the whole build.

import { createHash } from "node:crypto";
import { readFile, readdir, writeFile } from "node:fs/promises";
import { join, relative, resolve, sep } from "node:path";
import type { Plugin, ResolvedConfig } from "vite";

/**
 * connect-src 'self' (plus the translation Worker's origin when one is configured) makes the
 * browser refuse any other request, whatever a library tries. 'wasm-unsafe-eval' allows
 * WebAssembly compilation only, not JavaScript eval.
 */
export function contentSecurityPolicyText(workerOrigin = ""): string {
  return [
    "default-src 'self'",
    "script-src 'self' 'wasm-unsafe-eval'",
    "style-src 'self'",
    "img-src 'self' data: blob:",
    "media-src 'self' blob: mediastream:",
    `connect-src 'self'${workerOrigin ? ` ${workerOrigin}` : ""}`,
    "font-src 'self'",
    "worker-src 'self' blob:",
    "manifest-src 'self'",
    "object-src 'none'",
    "base-uri 'self'",
    "form-action 'none'",
  ].join("; ");
}

export const CONTENT_SECURITY_POLICY = contentSecurityPolicyText();

/**
 * Cross-origin isolation: SharedArrayBuffer, and so onnxruntime-web's wasm threads, needs
 * COOP same-origin and COEP require-corp. The dev and preview servers send them; on GitHub
 * Pages, which cannot set headers, the service worker adds them to every response it serves
 * (config/service-worker.template.js). Every resource is same-origin, so require-corp blocks
 * nothing the app loads.
 */
export const CROSS_ORIGIN_ISOLATION_HEADERS: Record<string, string> = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
  "Cross-Origin-Resource-Policy": "same-origin",
};

export function contentSecurityPolicy(workerOrigin = ""): Plugin {
  return {
    name: "edge-vision-csp",
    apply: "build",
    transformIndexHtml() {
      return [
        {
          tag: "meta",
          attrs: { "http-equiv": "Content-Security-Policy", content: contentSecurityPolicyText(workerOrigin) },
          injectTo: "head-prepend",
        },
      ];
    },
  };
}

async function listFiles(directory: string): Promise<string[]> {
  const entries = await readdir(directory, { withFileTypes: true });
  const nested = await Promise.all(
    entries.map((entry) => {
      const path = join(directory, entry.name);
      return entry.isDirectory() ? listFiles(path) : Promise.resolve([path]);
    }),
  );
  return nested.flat();
}

const EXCLUDED = [/\.map$/, /^sw\.js$/, /(^|\/)\.DS_Store$/];

export function serviceWorker(templatePath: string): Plugin {
  let config: ResolvedConfig;
  return {
    name: "edge-vision-service-worker",
    apply: "build",
    configResolved(resolved) {
      config = resolved;
    },
    async closeBundle() {
      const outDir = resolve(config.root, config.build.outDir);
      const files = (await listFiles(outDir))
        .map((path) => relative(outDir, path).split(sep).join("/"))
        .filter((path) => !EXCLUDED.some((pattern) => pattern.test(path)))
        .sort();
      const template = await readFile(templatePath, "utf8");
      // The cache version changes with any precached file or with the worker's own code.
      const digest = createHash("sha256").update(template);
      for (const path of files) {
        digest.update(path);
        digest.update(await readFile(join(outDir, path)));
      }
      const version = digest.digest("hex").slice(0, 16);
      const source = template
        .replace("__CACHE_VERSION__", version)
        .replace("__PRECACHE_URLS__", JSON.stringify(files, null, 2))
        .replace("__ISOLATION_HEADERS__", JSON.stringify(CROSS_ORIGIN_ISOLATION_HEADERS, null, 2));
      await writeFile(join(outDir, "sw.js"), source);
      config.logger.info(`service worker: ${files.length} files precached, version ${version}`);
    },
  };
}

/**
 * Serves (dev and preview) and emits (build) the files of a model variant kept outside public/
 * (web/model-variant.ts). For a public variant it does nothing, so those files never reach a
 * default build, its service worker or the deployed site.
 */
export function modelVariantAssets(files: { url: string; path: string }[]): Plugin {
  const byUrl = new Map(files.map((file) => [file.url, file.path]));
  const middleware =
    (base: string) =>
    (req: { url?: string }, res: { setHeader(name: string, value: string): void; end(body: Buffer): void }, next: () => void): void => {
      const path = (req.url ?? "").split("?")[0] ?? "";
      const relativeUrl = path.startsWith(base) ? path.slice(base.length) : path.replace(/^\//, "");
      const file = byUrl.get(relativeUrl);
      if (!file) return next();
      res.setHeader("Content-Type", file.endsWith(".json") ? "application/json" : "application/octet-stream");
      for (const [name, value] of Object.entries(CROSS_ORIGIN_ISOLATION_HEADERS)) res.setHeader(name, value);
      void readFile(file).then((body) => res.end(body));
    };
  let base = "/";
  return {
    name: "edge-vision-model-variant",
    configResolved(resolved) {
      base = resolved.base;
    },
    configureServer(server) {
      if (files.length) server.middlewares.use(middleware(base));
    },
    configurePreviewServer(server) {
      if (files.length) server.middlewares.use(middleware(base));
    },
    async generateBundle() {
      for (const file of files) this.emitFile({ type: "asset", fileName: file.url, source: await readFile(file.path) });
    },
  };
}
