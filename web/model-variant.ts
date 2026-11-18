// Which classifier the build ships: web/site.config.json "modelVariant", overridable with
// VITE_MODEL_VARIANT. vite.config.ts compiles the model directory into the bundle (__MODEL_DIR__)
// and, for a variant that is not public, serves and emits its files from outside public/.
//
// - v2-kaggle (default, the public site): trained on the asl-signs (Kaggle ISLR) data only,
//   web/public/model, committed.
// - v3-citizen: also trained on ASL Citizen, whose licence forbids redistribution. Its files live
//   in web/models/v3-citizen/ (gitignored) and enter a build only when this variant is chosen, so
//   the public site never carries them.

import { existsSync } from "node:fs";
import { join } from "node:path";

export interface VariantInfo {
  /** The model directory under the app's base URL. */
  dir: string;
  /** Where its files are, relative to web/. */
  source: string;
  /** Whether the files may be deployed publicly (public/ is copied into every build). */
  public: boolean;
  description: string;
}

export const MODEL_VARIANTS = {
  "v2-kaggle": { dir: "model", source: "public/model", public: true, description: "asl-signs only (public)" },
  "v3-citizen": {
    dir: "models/v3-citizen",
    source: "models/v3-citizen",
    public: false,
    description: "asl-signs + ASL Citizen (private: research licence, never deploy publicly)",
  },
} as const satisfies Record<string, VariantInfo>;

export type ModelVariant = keyof typeof MODEL_VARIANTS;
export const DEFAULT_MODEL_VARIANT: ModelVariant = "v2-kaggle";
/** The files a variant needs besides the app: the classifier and its metadata. */
export const MODEL_FILES = ["model.onnx", "meta.json"] as const;

/** The override when non-blank, else the file's value, else the default; unknown names throw. */
export function resolveModelVariant(override: string | undefined, fileValue: unknown): ModelVariant {
  const env = override?.trim() ?? "";
  const file = typeof fileValue === "string" ? fileValue.trim() : "";
  const name = env || file || DEFAULT_MODEL_VARIANT;
  if (!(name in MODEL_VARIANTS)) {
    throw new Error(`unknown modelVariant ${JSON.stringify(name)}; expected one of ${Object.keys(MODEL_VARIANTS).join(", ")}`);
  }
  return name as ModelVariant;
}

/**
 * The files a build or dev server must add for a variant: none for a public one (public/ holds
 * them), the variant's own files otherwise. Throws when they are missing.
 */
export function privateModelFiles(variant: ModelVariant, webRoot: string): { url: string; path: string }[] {
  const info: VariantInfo = MODEL_VARIANTS[variant];
  if (info.public) return [];
  const files = MODEL_FILES.map((name) => ({ url: `${info.dir}/${name}`, path: join(webRoot, info.source, name) }));
  const missing = files.filter((file) => !existsSync(file.path)).map((file) => file.path);
  if (missing.length > 0) {
    throw new Error(`modelVariant ${variant} needs ${missing.join(", ")} (see kaggle/ASL_CITIZEN.md, step 4)`);
  }
  return files;
}

export function describeModelVariant(variant: ModelVariant): string {
  return `model variant: ${variant} (${MODEL_VARIANTS[variant].description}), served from ${MODEL_VARIANTS[variant].dir}/`;
}
