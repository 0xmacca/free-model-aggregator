// The trial brief: the three JSON documents inside MAKEFUN_PROMPT_FILE.
//
// The brief is prose with three JSON documents embedded in it. Each document
// carries a `schema` member, which is how this file finds it; nothing else in
// the prose matters.
//
//   providers  free-model-aggregator-01/providers/v1
//              one row per provider: id, name, wire, auth scheme, the key
//              environment variable, the key, and the base path (or absolute
//              URL) its routes hang off.
//   ratings    free-model-aggregator-01/ratings/v1
//              one row per (provider, model): the public rating in
//              milli-points, context length, whether it takes tools, images
//              and JSON, and whether it is free. This is the sealed truth
//              the scorer grades against.
//   workload   free-model-aggregator-01/workload/v1
//              the twenty requests the client will send, in order: id, api,
//              whether it streams, weight, and the requirements a model must
//              meet to serve it.
//
// Everything here is pure parsing: no clock, no randomness, so the same
// brief always reads the same way.

import { readFileSync } from 'node:fs';

export type Wire = 'openai' | 'anthropic' | 'gemini';
export type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';
export type Json = Record<string, unknown>;

export const isRecord = (v: unknown): v is Json =>
  v !== null && typeof v === 'object' && !Array.isArray(v);

export const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
export const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
export const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

export interface ProviderEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  base: string;
  key_env: string;
  key: string;
}

export interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

export interface WorkloadRequest {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

export interface Brief {
  providers: ProviderEntry[];
  ratings: RatingRow[];
  best: Map<string, number>;
  workload: WorkloadRequest[];
  error: string | null;
}

function textOf(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

function numOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

/** The index of the `}` closing the object that opens at `start`, or -1. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Every JSON object in `text` that carries a string `schema`, in order. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not a JSON object on its own: look on.
    }
  }
  return out;
}

function wireOf(v: unknown): Wire | null {
  return v === 'openai' || v === 'anthropic' || v === 'gemini' ? v : null;
}

function authOf(v: unknown): AuthScheme {
  if (v === 'x-api-key' || v === 'x-goog-api-key' || v === 'none') return v;
  return 'bearer';
}

function providerOf(v: unknown): ProviderEntry | null {
  if (!isRecord(v)) return null;
  const id = textOf(v.id);
  const wire = wireOf(v.wire);
  if (id === '' || wire === null) return null;
  // `path` first, then the spellings a directory may use instead.
  const base =
    textOf(v.path) !== ''
      ? textOf(v.path)
      : textOf(v.base_path) !== ''
        ? textOf(v.base_path)
        : textOf(v.base_url) !== ''
          ? textOf(v.base_url)
          : textOf(v.base);
  return {
    id,
    name: textOf(v.name) !== '' ? textOf(v.name) : id,
    wire,
    auth: authOf(v.auth),
    base,
    key_env: textOf(v.key_env),
    key: textOf(v.key),
  };
}

function ratingOf(v: unknown): RatingRow | null {
  if (!isRecord(v)) return null;
  const model = textOf(v.model);
  const provider = textOf(v.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    rating_milli:
      typeof v.rating_milli === 'number' && Number.isFinite(v.rating_milli) ? v.rating_milli : 0,
    context_length: numOrNull(v.context_length),
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free === true,
  };
}

function workloadOf(v: unknown): WorkloadRequest | null {
  if (!isRecord(v)) return null;
  const id = textOf(v.id);
  if (id === '') return null;
  const requires = isRecord(v.requires) ? v.requires : {};
  const req: Requirement = {};
  if (requires.tools === true) req.tools = true;
  if (requires.vision === true) req.vision = true;
  if (requires.json === true) req.json = true;
  if (typeof requires.min_context === 'number' && Number.isFinite(requires.min_context)) {
    req.min_context = requires.min_context;
  }
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: req,
  };
}

/** The brief at `path`, or an empty brief with `error` when it cannot be read. */
export function readBrief(path: string): Brief {
  const empty: Brief = { providers: [], ratings: [], best: new Map(), workload: [], error: null };
  if (path === '') return { ...empty, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...empty, error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const dirDoc = docs.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers));
  if (dirDoc === undefined || !Array.isArray(dirDoc.providers)) {
    return { ...empty, error: `the brief at ${path} holds no ${PROVIDERS_SCHEMA} document` };
  }
  const providers = dirDoc.providers.flatMap((v) => {
    const p = providerOf(v);
    return p === null ? [] : [p];
  });
  const ratingsDoc = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings));
  const ratings =
    ratingsDoc === undefined || !Array.isArray(ratingsDoc.ratings)
      ? []
      : ratingsDoc.ratings.flatMap((v) => {
          const r = ratingOf(v);
          return r === null ? [] : [r];
        });
  const best = new Map<string, number>();
  if (ratingsDoc !== undefined && Array.isArray(ratingsDoc.best)) {
    for (const v of ratingsDoc.best) {
      if (isRecord(v) && textOf(v.id) !== '' && typeof v.rating_milli === 'number') {
        best.set(textOf(v.id), v.rating_milli);
      }
    }
  }
  const workDoc = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest));
  const workload =
    workDoc === undefined || !Array.isArray(workDoc.manifest)
      ? []
      : workDoc.manifest.flatMap((v) => {
          const w = workloadOf(v);
          return w === null ? [] : [w];
        });
  return { providers, ratings, best, workload, error: null };
}

/** The quality table's row for one provider's model, when it has one. */
export function ratingRow(brief: Brief, providerId: string, modelId: string): RatingRow | null {
  return brief.ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;
}

/** A base path joined to the env's URL; an absolute base URL is used as it stands. */
export function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (b === '') return prefix;
  return `${prefix}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** The headers that carry a provider's key, by its own auth scheme. */
export function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}
