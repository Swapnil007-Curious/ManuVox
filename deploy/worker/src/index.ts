// Sentence translator for the ManuVox web app (web/), a Cloudflare Worker.
//
// POST /translate takes {glosses: [...]} (recognised signs, each from the app's vocabulary),
// asks the provider's OpenAI-compatible chat completions for one short English sentence with a
// fixed prompt and at most 120 output tokens, and returns {sentence}. The API key lives only in
// the LLM_API_KEY secret; it is read for the upstream Authorization header and nowhere else
// (never logged, never echoed). Glosses and sentences are not stored or logged. Per-client limits
// (CF-Connecting-IP): 30 per 10 minutes and 300 per day; a global daily cap shields the key.

import { clientLimiter, globalCounter, LimitError, type LimitSettings } from "./limits.ts";
import { buildMessages, cleanSentence, validate, ValidationError, type TranslateRequest } from "./request.ts";

export interface Env {
  LLM_API_KEY: string;
  LLM_BASE_URL?: string;
  LLM_MODEL?: string;
  ALLOWED_ORIGINS?: string; // comma separated; default: the Pages site (localhost is always allowed)
  MAX_OUTPUT_TOKENS?: string;
  LIMIT_PER_10MIN?: string;
  LIMIT_PER_DAY?: string;
  GLOBAL_PER_DAY?: string;
  LIMITS: DurableObjectNamespace;
}

export { RateLimiter } from "./limits.ts";

const DEFAULT_ORIGINS = ["https://swapnil007-curious.github.io"];
const DEFAULT_BASE_URL = "https://api.groq.com/openai/v1";
const DEFAULT_MODEL = "llama-3.1-8b-instant";
/** The ceiling on output tokens; MAX_OUTPUT_TOKENS can only lower it. */
export const OUTPUT_TOKEN_CAP = 120;
const BUSY = "The translator is busy right now. Please try again in a minute.";

function num(raw: string | undefined, fallback: number): number {
  const n = Number.parseInt(raw ?? "", 10);
  return Number.isFinite(n) && n > 0 ? n : fallback;
}

export function settings(env: Env): LimitSettings {
  return {
    perWindow: num(env.LIMIT_PER_10MIN, 30),
    windowSeconds: 600,
    perDay: num(env.LIMIT_PER_DAY, 300),
    globalPerDay: num(env.GLOBAL_PER_DAY, 3000),
  };
}

export function maxTokens(env: Env): number {
  return Math.min(OUTPUT_TOKEN_CAP, num(env.MAX_OUTPUT_TOKENS, OUTPUT_TOKEN_CAP));
}

export function allowedOrigins(env: Env): string[] {
  const listed = (env.ALLOWED_ORIGINS ?? "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
  return listed.length ? listed : DEFAULT_ORIGINS;
}

/** Development origins: any http://localhost or http://127.0.0.1 port. */
export function originAllowed(origin: string | null, env: Env): boolean {
  if (!origin) return false;
  if (/^http:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/.test(origin)) return true;
  return allowedOrigins(env).includes(origin);
}

function corsHeaders(origin: string): Record<string, string> {
  return {
    "access-control-allow-origin": origin,
    "access-control-allow-methods": "POST, GET, OPTIONS",
    "access-control-allow-headers": "content-type",
    "access-control-max-age": "86400",
    vary: "origin",
  };
}

function json(status: number, body: Record<string, unknown>, headers: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store", ...headers },
  });
}

async function upstream(env: Env, request: TranslateRequest): Promise<Response> {
  return fetch(`${(env.LLM_BASE_URL || DEFAULT_BASE_URL).replace(/\/+$/, "")}/chat/completions`, {
    method: "POST",
    headers: { "content-type": "application/json", authorization: `Bearer ${env.LLM_API_KEY}` },
    body: JSON.stringify({
      model: env.LLM_MODEL || DEFAULT_MODEL,
      messages: buildMessages(request),
      max_tokens: maxTokens(env),
      temperature: 0.2,
      stream: false,
    }),
  });
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    const origin = request.headers.get("origin");
    const allowed = originAllowed(origin, env);
    const cors = allowed && origin ? corsHeaders(origin) : {};
    if (request.method === "OPTIONS") {
      return allowed ? new Response(null, { status: 204, headers: cors }) : json(403, { error: "forbidden", message: "This origin is not allowed." });
    }
    if (url.pathname === "/health" && request.method === "GET") {
      return json(200, { status: "ok", model: env.LLM_MODEL || DEFAULT_MODEL, max_output_tokens: maxTokens(env), limits: settings(env) }, cors);
    }
    if (url.pathname !== "/translate" || request.method !== "POST") {
      return json(404, { error: "not_found", message: "POST /translate" }, cors);
    }
    if (!allowed) return json(403, { error: "forbidden", message: "This origin is not allowed." });
    if (!env.LLM_API_KEY) return json(503, { error: "not_configured", message: "The translator has no API key configured." }, cors);

    let body: TranslateRequest;
    try {
      body = validate(await request.json());
    } catch (err) {
      const message = err instanceof ValidationError ? err.message : "The body must be JSON.";
      return json(400, { error: "bad_request", message }, cors);
    }

    const limits = settings(env);
    const ip = request.headers.get("cf-connecting-ip") ?? "unknown";
    try {
      await clientLimiter(env.LIMITS, ip).take(limits);
      await globalCounter(env.LIMITS).take(limits);
    } catch (err) {
      if (err instanceof LimitError) {
        return json(429, { error: err.code, message: err.message, retry_after: err.retryAfter }, { ...cors, "retry-after": String(Math.ceil(err.retryAfter)) });
      }
      throw err;
    }

    let res: Response;
    try {
      res = await upstream(env, body);
    } catch {
      return json(503, { error: "unavailable", message: "The model could not be reached. Please try again in a moment." }, cors);
    }
    if (res.status === 429) {
      const retry = Number.parseFloat(res.headers.get("retry-after") ?? "") || 60;
      return json(429, { error: "busy", message: BUSY, retry_after: retry }, { ...cors, "retry-after": String(Math.ceil(retry)) });
    }
    if (!res.ok) {
      // The provider's error text could carry request details; only the status is passed on.
      return json(502, { error: "upstream", message: `The model answered with an error (HTTP ${res.status}).` }, cors);
    }
    let reply: { choices?: { message?: { content?: unknown } }[] };
    try {
      reply = await res.json();
    } catch {
      return json(502, { error: "upstream", message: "The model's reply could not be read." }, cors);
    }
    const content = reply.choices?.[0]?.message?.content;
    const sentence = typeof content === "string" ? cleanSentence(content) : "";
    if (!sentence) return json(502, { error: "empty", message: "The model returned no sentence." }, cors);
    return json(200, { sentence, model: env.LLM_MODEL || DEFAULT_MODEL }, cors);
  },
} satisfies ExportedHandler<Env>;
