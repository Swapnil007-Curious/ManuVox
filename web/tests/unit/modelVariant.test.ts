// The modelVariant switch (web/model-variant.ts): v2-kaggle is the default and adds nothing to a
// build; v3-citizen's files, kept outside public/, enter a build only when it is chosen.

import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { modelVariantAssets } from "../../config/plugins";
import { DEFAULT_MODEL_VARIANT, MODEL_VARIANTS, privateModelFiles, resolveModelVariant } from "../../model-variant";

const site = JSON.parse(readFileSync(new URL("../../site.config.json", import.meta.url), "utf8")) as { modelVariant?: string };

describe("resolveModelVariant", () => {
  it("defaults to the public v2 model, and the committed config keeps it", () => {
    expect(DEFAULT_MODEL_VARIANT).toBe("v2-kaggle");
    expect(resolveModelVariant(undefined, undefined)).toBe("v2-kaggle");
    expect(resolveModelVariant("  ", site.modelVariant)).toBe("v2-kaggle");
    expect(MODEL_VARIANTS["v2-kaggle"]).toMatchObject({ dir: "model", public: true });
    expect(MODEL_VARIANTS["v3-citizen"].public).toBe(false);
  });

  it("takes the environment over the file and rejects unknown names", () => {
    expect(resolveModelVariant("v3-citizen", "v2-kaggle")).toBe("v3-citizen");
    expect(resolveModelVariant("", "v3-citizen")).toBe("v3-citizen");
    expect(() => resolveModelVariant("v4", undefined)).toThrow(/unknown modelVariant/);
  });
});

describe("private model files", () => {
  it("are none for the public variant", () => {
    expect(privateModelFiles("v2-kaggle", "/nowhere")).toEqual([]);
  });

  it("must exist for v3-citizen", () => {
    const root = mkdtempSync(join(tmpdir(), "variant-"));
    expect(() => privateModelFiles("v3-citizen", root)).toThrow(/needs/);
    mkdirSync(join(root, "models", "v3-citizen"), { recursive: true });
    for (const name of ["model.onnx", "meta.json"]) writeFileSync(join(root, "models", "v3-citizen", name), name);
    const files = privateModelFiles("v3-citizen", root);
    expect(files.map((file) => file.url)).toEqual(["models/v3-citizen/model.onnx", "models/v3-citizen/meta.json"]);
  });

  it("are emitted into a build only when given", async () => {
    const root = mkdtempSync(join(tmpdir(), "variant-"));
    mkdirSync(join(root, "models", "v3-citizen"), { recursive: true });
    for (const name of ["model.onnx", "meta.json"]) writeFileSync(join(root, "models", "v3-citizen", name), name);
    const emitted: string[] = [];
    const context = { emitFile: (file: { fileName: string }) => emitted.push(file.fileName) };
    const run = async (files: { url: string; path: string }[]) => {
      const plugin = modelVariantAssets(files) as unknown as { generateBundle(this: typeof context): Promise<void> };
      await plugin.generateBundle.call(context);
    };
    await run(privateModelFiles("v2-kaggle", root));
    expect(emitted).toEqual([]);
    await run(privateModelFiles("v3-citizen", root));
    expect(emitted).toEqual(["models/v3-citizen/model.onnx", "models/v3-citizen/meta.json"]);
  });
});
