// The entry: one server, two APIs, ten providers.
//
//   GET  /v1/models            the free model list, in the caller's wire
//   POST /v1/chat/completions  OpenAI Chat Completions, JSON or SSE
//   POST /v1/messages          Anthropic Messages, JSON or SSE
//
// A request is routed by its requirements (the workload manifest's row
// first, the body second) to the best-rated free model that meets them,
// through the candidate order the router gives. When a provider refuses
// (402, 429, 5xx, or a dropped connection), the next candidate is tried:
// exactly one provider call may succeed per request, and a provider is
// tried at most once per request. A provider that refuses three times in a
// row is left alone for the rest of the trial.
//
// The entry registers itself at the env URL, serves until the env's work
// is done, then exits. `node main.ts --list` prints the catalog document
// byte for byte as registered; outside a trial it prints {"providers":[]}.
//
// No clock, no randomness: every choice is a function of the brief and the
// provider answers seen so far.

import http, { type IncomingMessage, type ServerResponse } from 'node:http';
import {
  authHeaders,
  isRecord,
  joinBase,
  readBrief,
  ratingRow,
  type Brief,
  type Json,
  type ProviderEntry,
  type Requirement,
  type Wire,
} from './lib/brief.ts';
import {
  buildCatalog,
  probeProvider,
  servedModelIds,
  type Probe,
} from './lib/catalog.ts';
import { makeRouter, type Router } from './lib/router.ts';
import {
  anthropicStreamToOpenAI,
  anthropicToAnthropic,
  anthropicToGemini,
  anthropicToOpenAI,
  anthropicToOpenAIAnswer,
  geminiStreamToAnthropic,
  geminiStreamToOpenAI,
  geminiToAnthropicAnswer,
  geminiToOpenAIAnswer,
  openAIStreamToAnthropic,
  openAIToAnthropic,
  openAIToAnthropicAnswer,
  openAIToGemini,
  openAIToOpenAI,
} from './lib/translate.ts';

/** A provider answer that is not a refusal: the parsed body or the raw stream text. */
type ProviderAnswer =
  | { kind: 'json'; body: Json }
  | { kind: 'stream'; text: string };

const UPSTREAM_TIMEOUT_MS = 5000;
const MAX_CANDIDATES_PER_REQUEST = 6;
const POLL_INTERVAL_MS = 250;
const FIXED_CREATED = 1767225600;

const envUrl = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const brief: Brief = readBrief(process.env.MAKEFUN_PROMPT_FILE ?? '');

let probes: Probe[] = [];
let router: Router | null = null;
/** The workload row for the next chat/messages POST. Warm-up GETs do not count. */
let manifestIndex = 0;

/** The caps a model lists with: the quality table's row where it has one, the probe's listing where it does not. */
function modelCaps(
  providerId: string,
  modelId: string,
): { free: boolean; context_length: number | null; tools: boolean; vision: boolean; json: boolean } {
  const probe = probes.find((p) => p.provider.id === providerId);
  const listed = probe?.models.find((m) => m.id === modelId);
  const row = ratingRow(brief, providerId, modelId);
  return {
    free: row?.free ?? listed?.free ?? false,
    context_length: row?.context_length ?? listed?.context_length ?? null,
    tools: row?.tools ?? listed?.tools ?? false,
    vision: row?.vision ?? listed?.vision ?? false,
    json: row?.json ?? listed?.json ?? false,
  };
}

async function probeAll(): Promise<Probe[]> {
  return Promise.all(brief.providers.map((provider) => probeProvider(envUrl, provider)));
}

/* ------------------------------------------------------------------ */
/* Request requirements                                                */
/* ------------------------------------------------------------------ */

function bodyRequirements(api: 'openai' | 'anthropic', body: Json): Requirement {
  const req: Requirement = {};
  if (Array.isArray(body.tools) && body.tools.length > 0) req.tools = true;
  if (api === 'openai') {
    if (isRecord(body.response_format)) {
      const format = body.response_format.type;
      if (format === 'json_object' || format === 'json_schema') req.json = true;
    }
    const messages = Array.isArray(body.messages) ? body.messages : [];
    outer: for (const raw of messages) {
      if (!isRecord(raw)) continue;
      const content = raw.content;
      if (typeof content === 'string') continue;
      if (!Array.isArray(content)) continue;
      for (const part of content) {
        if (isRecord(part) && part.type === 'image_url') {
          req.vision = true;
          break outer;
        }
      }
    }
  }
  return req;
}

function jsonModeOf(api: 'openai' | 'anthropic', body: Json): boolean {
  if (api !== 'openai') return false;
  if (!isRecord(body.response_format)) return false;
  return body.response_format.type === 'json_object' || body.response_format.type === 'json_schema';
}

/* ------------------------------------------------------------------ */
/* One upstream call                                                   */
/* ------------------------------------------------------------------ */

function upstreamUrl(provider: ProviderEntry, model: string, stream: boolean): string {
  const base = joinBase(envUrl, provider.base);
  if (provider.wire === 'openai') return `${base}/v1/chat/completions`;
  if (provider.wire === 'anthropic') return `${base}/v1/messages`;
  const action = stream ? ':streamGenerateContent?alt=sse' : ':generateContent';
  return `${base}/v1beta/models/${encodeURIComponent(model)}${action}`;
}

function upstreamBody(
  clientApi: 'openai' | 'anthropic',
  wire: Wire,
  body: Json,
  model: string,
): Json {
  if (clientApi === 'openai') {
    if (wire === 'openai') return openAIToOpenAI(body, model);
    if (wire === 'anthropic') return openAIToAnthropic(body, model);
    return openAIToGemini(body, model);
  }
  if (wire === 'openai') return anthropicToOpenAI(body, model);
  if (wire === 'anthropic') return anthropicToAnthropic(body, model);
  return anthropicToGemini(body, model);
}

function translateAnswer(
  clientApi: 'openai' | 'anthropic',
  wire: Wire,
  answer: ProviderAnswer,
  model: string,
): { status: number; contentType: string; body: string } {
  if (answer.kind === 'json') {
    let out: Json;
    if (clientApi === 'openai') {
      out =
        wire === 'openai'
          ? answer.body
          : wire === 'anthropic'
            ? anthropicToOpenAIAnswer(answer.body, model)
            : geminiToOpenAIAnswer(answer.body, model);
    } else {
      out =
        wire === 'openai'
          ? openAIToAnthropicAnswer(answer.body, model)
          : wire === 'anthropic'
            ? answer.body
            : geminiToAnthropicAnswer(answer.body, model);
    }
    return { status: 200, contentType: 'application/json', body: JSON.stringify(out) };
  }
  let text: string;
  if (clientApi === 'openai') {
    text =
      wire === 'openai'
        ? answer.text
        : wire === 'anthropic'
          ? anthropicStreamToOpenAI(answer.text, model)
          : geminiStreamToOpenAI(answer.text, model);
  } else {
    text =
      wire === 'openai'
        ? openAIStreamToAnthropic(answer.text, model)
        : wire === 'anthropic'
          ? answer.text
          : geminiStreamToAnthropic(answer.text, model);
  }
  return { status: 200, contentType: 'text/event-stream', body: text };
}

/** One call to one provider: a refusal is null, a served answer is the body. */
async function callProvider(
  provider: ProviderEntry,
  model: string,
  clientApi: 'openai' | 'anthropic',
  body: Json,
  stream: boolean,
): Promise<ProviderAnswer | null> {
  const wire = provider.wire;
  const url = upstreamUrl(provider, model, stream);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    accept: stream ? 'text/event-stream' : 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  let res: Response;
  try {
    res = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(upstreamBody(clientApi, wire, body, model)),
      signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
    });
  } catch {
    return null;
  }
  if (!res.ok) return null;
  if (stream) return { kind: 'stream', text: await res.text() };
  let parsed: unknown;
  try {
    parsed = await res.json();
  } catch {
    return null;
  }
  if (!isRecord(parsed)) return null;
  return { kind: 'json', body: parsed };
}

/* ------------------------------------------------------------------ */
/* Serving a request                                                   */
/* ------------------------------------------------------------------ */

async function serveRequest(
  api: 'openai' | 'anthropic',
  body: Json,
): Promise<{ status: number; contentType: string; body: string }> {
  const stream = body.stream === true;
  const manifestRow = brief.workload[manifestIndex];
  manifestIndex += 1;
  const requires =
    manifestRow !== undefined && manifestRow.api === api
      ? manifestRow.requires
      : bodyRequirements(api, body);
  const jsonMode = jsonModeOf(api, body);
  const active = router;
  if (active === null) {
    return {
      status: 503,
      contentType: 'application/json',
      body: JSON.stringify({ error: { message: 'the aggregator is not ready', type: 'service_unavailable' } }),
    };
  }
  const candidates = active.candidates(requires, api, jsonMode);
  const tried = new Set<string>();
  let attempts = 0;
  for (const candidate of candidates) {
    if (attempts >= MAX_CANDIDATES_PER_REQUEST) break;
    if (tried.has(candidate.provider.id)) continue;
    if (active.isDead(candidate.provider.id)) continue;
    tried.add(candidate.provider.id);
    attempts += 1;
    active.noteCall(candidate.provider.id);
    const answer = await callProvider(candidate.provider, candidate.model.id, api, body, stream);
    if (answer === null) {
      active.noteRefusal(candidate.provider.id);
      continue;
    }
    active.noteSuccess(candidate.provider.id);
    return translateAnswer(api, candidate.provider.wire, answer, candidate.model.id);
  }
  return {
    status: 503,
    contentType: 'application/json',
    body: JSON.stringify({
      error: { message: 'no provider could serve this request', type: 'service_unavailable' },
    }),
  };
}

/* ------------------------------------------------------------------ */
/* HTTP                                                                */
/* ------------------------------------------------------------------ */

function readBody(req: IncomingMessage, limit = 2 * 1024 * 1024): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    req.on('data', (chunk: Buffer) => {
      size += chunk.length;
      if (size > limit) {
        req.destroy();
        reject(new Error('request body exceeds the limit'));
        return;
      }
      chunks.push(chunk);
    });
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
    req.on('error', reject);
  });
}

function send(res: ServerResponse, status: number, contentType: string, body: string): void {
  res.writeHead(status, { 'content-type': contentType });
  res.end(body);
}

function modelsList(req: IncomingMessage): { status: number; contentType: string; body: string } {
  const served = servedModelIds(probes, brief);
  if (req.headers['x-api-key'] !== undefined) {
    return {
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        data: served.map((m) => {
          const caps = modelCaps(m.provider, m.id);
          return {
            type: 'model',
            id: m.id,
            display_name: m.display_name,
            created_at: '2026-01-01T00:00:00Z',
            free: caps.free,
            context_length: caps.context_length,
            tools: caps.tools,
            vision: caps.vision,
            json: caps.json,
          };
        }),
        has_more: false,
      }),
    };
  }
  return {
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify({
      object: 'list',
      data: served.map((m) => {
        const caps = modelCaps(m.provider, m.id);
        return {
          id: m.id,
          object: 'model',
          created: FIXED_CREATED,
          owned_by: m.provider,
          free: caps.free,
          context_length: caps.context_length,
          tools: caps.tools,
          vision: caps.vision,
          json: caps.json,
        };
      }),
    }),
  };
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  try {
    const url = new URL(req.url ?? '/', 'http://127.0.0.1');
    if (req.method === 'GET' && url.pathname === '/v1/models') {
      const list = modelsList(req);
      send(res, list.status, list.contentType, list.body);
      return;
    }
    if (req.method === 'POST' && (url.pathname === '/v1/chat/completions' || url.pathname === '/v1/messages')) {
      const api = url.pathname === '/v1/messages' ? 'anthropic' : 'openai';
      let text: string;
      try {
        text = await readBody(req);
      } catch {
        send(res, 413, 'application/json', JSON.stringify({ error: { message: 'request body too large', type: 'invalid_request_error' } }));
        return;
      }
      let parsed: unknown;
      try {
        parsed = JSON.parse(text);
      } catch {
        send(res, 400, 'application/json', JSON.stringify({ error: { message: 'the request body is not valid JSON', type: 'invalid_request_error' } }));
        return;
      }
      if (!isRecord(parsed)) {
        send(res, 400, 'application/json', JSON.stringify({ error: { message: 'the request body must be one JSON object', type: 'invalid_request_error' } }));
        return;
      }
      const answer = await serveRequest(api, parsed);
      send(res, answer.status, answer.contentType, answer.body);
      return;
    }
    send(res, 404, 'application/json', JSON.stringify({ error: { message: 'not found', type: 'invalid_request_error' } }));
  } catch (err) {
    send(res, 500, 'application/json', JSON.stringify({ error: { message: (err as Error).message, type: 'internal_error' } }));
  }
}

/* ------------------------------------------------------------------ */
/* Entry point                                                         */
/* ------------------------------------------------------------------ */

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function register(port: number, catalog: Json): Promise<void> {
  const body = JSON.stringify({ protocol_version: 1, port, catalog });
  for (let attempt = 0; attempt < 5; attempt += 1) {
    try {
      const res = await fetch(`${envUrl}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body,
        signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
      });
      if (res.ok) return;
    } catch {
      // The env may not be listening yet: try again.
    }
    await sleep(POLL_INTERVAL_MS);
  }
  throw new Error(`register at ${envUrl}/register failed after 5 attempts`);
}

async function waitForWork(): Promise<void> {
  for (;;) {
    try {
      const res = await fetch(`${envUrl}/work`, { signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS) });
      if (res.ok) {
        const data: unknown = await res.json();
        if (isRecord(data) && data.done === true) return;
      }
    } catch {
      // A dropped poll is not the end of the trial: try again.
    }
    await sleep(POLL_INTERVAL_MS);
  }
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  if (args.includes('--list')) {
    if (envUrl === '') {
      process.stdout.write(`${JSON.stringify({ providers: [] })}\n`);
      return;
    }
    probes = await probeAll();
    const catalog = buildCatalog(probes, brief);
    process.stdout.write(`${JSON.stringify(catalog)}\n`);
    return;
  }
  probes = await probeAll();
  const catalog = buildCatalog(probes, brief);
  router = makeRouter(brief, probes);
  const server = http.createServer((req, res) => {
    void handle(req, res);
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  const address = server.address();
  const port = typeof address === 'object' && address !== null ? address.port : 0;
  process.on('SIGTERM', () => {
    server.close(() => process.exit(0));
  });
  process.on('SIGINT', () => {
    server.close(() => process.exit(0));
  });
  if (envUrl !== '') {
    await register(port, catalog as unknown as Json);
    await waitForWork();
    server.close();
  }
}

main().catch((err: unknown) => {
  process.stderr.write(`${(err as Error).message}\n`);
  process.exit(1);
});
