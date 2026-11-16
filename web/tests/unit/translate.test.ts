// The translation client: the Worker when it answers, the basic joiner otherwise.

import { describe, expect, it, vi } from "vitest";
import { MAX_GLOSSES, translate } from "../../src/ui/translate";

const URL_ = "https://translate.example";

function reply(status: number, body: unknown): typeof fetch {
  return vi.fn(async () => new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } })) as unknown as typeof fetch;
}

describe("translate", () => {
  it("posts the glosses (not the model labels) and returns the Worker's sentence", async () => {
    const fetchFn = reply(200, { sentence: "Thank you, mom.", model: "m" });
    const result = await translate(["thankyou", "mom"], { workerUrl: `${URL_}/`, fetchFn });
    expect(result).toEqual({ text: "Thank you, mom.", source: "worker", reason: null });
    const [url, init] = (fetchFn as unknown as ReturnType<typeof vi.fn>).mock.calls[0] as [string, RequestInit];
    expect(url).toBe(`${URL_}/translate`);
    expect(JSON.parse(String(init.body))).toEqual({ glosses: ["THANK-YOU", "MOM"] });
  });

  it("uses the basic joiner without a Worker URL or offline, without a request", async () => {
    const fetchFn = reply(200, { sentence: "x" });
    expect(await translate(["hello", "mom"], { workerUrl: "", fetchFn })).toEqual({ text: "Hello, mom.", source: "basic", reason: "no translation service configured" });
    expect((await translate(["hello"], { workerUrl: URL_, online: false, fetchFn })).source).toBe("basic");
    expect(fetchFn).not.toHaveBeenCalled();
  });

  it("falls back on refusals, errors, empty replies and network failures, keeping the reason", async () => {
    const limited = await translate(["yes"], { workerUrl: URL_, fetchFn: reply(429, { error: "rate_limited", message: "Please try again in about 5 minutes." }) });
    expect(limited).toEqual({ text: "Yes.", source: "basic", reason: "Please try again in about 5 minutes." });
    expect((await translate(["yes"], { workerUrl: URL_, fetchFn: reply(502, {}) })).reason).toBe("HTTP 502");
    expect((await translate(["yes"], { workerUrl: URL_, fetchFn: reply(200, { sentence: "  " }) })).reason).toBe("empty reply");
    const down = vi.fn(async () => {
      throw new TypeError("Failed to fetch");
    }) as unknown as typeof fetch;
    expect(await translate(["no"], { workerUrl: URL_, fetchFn: down })).toEqual({ text: "No.", source: "basic", reason: "the translation service is unreachable" });
  });

  it("gives up after the timeout", async () => {
    const hang = vi.fn(
      (_url: string, init?: RequestInit) =>
        new Promise<Response>((_resolve, reject) => init?.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")))),
    ) as unknown as typeof fetch;
    const result = await translate(["water"], { workerUrl: URL_, fetchFn: hang, timeoutMs: 20 });
    expect(result).toEqual({ text: "Water.", source: "basic", reason: "the translation service did not answer in time" });
  });

  it("sends at most the last 24 glosses and nothing for an empty transcript", async () => {
    const fetchFn = reply(200, { sentence: "ok" });
    await translate(Array.from({ length: 30 }, (_, i) => (i < 6 ? "no" : "yes")), { workerUrl: URL_, fetchFn });
    const [, init] = (fetchFn as unknown as ReturnType<typeof vi.fn>).mock.calls[0] as [string, RequestInit];
    const sent = JSON.parse(String(init.body)).glosses as string[];
    expect(sent).toHaveLength(MAX_GLOSSES);
    expect(new Set(sent)).toEqual(new Set(["YES"]));
    expect(await translate([], { workerUrl: URL_, fetchFn })).toEqual({ text: "", source: "basic", reason: null });
  });

  it("rethrows when the caller aborts (a newer request replaced it)", async () => {
    const controller = new AbortController();
    const hang = vi.fn(
      (_url: string, init?: RequestInit) =>
        new Promise<Response>((_resolve, reject) => init?.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")))),
    ) as unknown as typeof fetch;
    const pending = translate(["water"], { workerUrl: URL_, fetchFn: hang, signal: controller.signal });
    controller.abort();
    await expect(pending).rejects.toThrow();
  });
});
