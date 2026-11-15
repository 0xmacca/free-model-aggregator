// The router: which (provider, model) serves one request, and what to try
// when that provider refuses.
//
// A candidate is a free model that meets the request's requirements, ranked
// by the quality table's rating, best first. Two rules beyond the rating:
//
//   - A JSON-mode request on the OpenAI wire carries `response_format`,
//     which an Anthropic-wire provider cannot echo back (its wire has no
//     JSON mode), so such a request never goes to an anthropic-wire
//     provider. An Anthropic-wire client request carries no JSON mode at
//     all, so the rule only binds OpenAI client requests.
//   - A provider that refuses is not dead at once: the failure schedule
//     refuses isolated calls (at most two in a row in the sealed table),
//     while a spent budget or an outage refuses every call after. So a
//     provider is skipped for the rest of the trial only after three
//     refusals in a row; before that it stays in the ranking, and a success
//     clears its count.
//
// Within one request a provider is tried at most once: a refusal is a
// property of the provider's budget and schedule, not of the model, so a
// second model on the same provider would only spend another call for a
// worse rating. The next request tries the ranking again from the top.
//
// The router keeps no clock and no randomness: every choice is a function
// of the brief and the provider answers seen so far, so a replay takes the
// same path.

import { capabilitiesOf, isFreeModel, type ListedModel, type Probe } from './catalog.ts';
import { ratingRow, type Brief, type ProviderEntry, type Requirement } from './brief.ts';

export interface Candidate {
  provider: ProviderEntry;
  model: ListedModel;
  rating: number;
}

export interface Router {
  candidates(requires: Requirement, clientApi: 'openai' | 'anthropic', jsonMode: boolean): Candidate[];
  noteRefusal(providerId: string): void;
  noteSuccess(providerId: string): void;
  isDead(providerId: string): boolean;
  callsOf(providerId: string): number;
  noteCall(providerId: string): void;
}

/** How many refusals in a row mark a provider spent for the rest of the trial. */
const DEAD_AFTER_REFUSALS = 2;

function meets(
  requires: Requirement,
  caps: { context_length: number | null; tools: boolean | null; vision: boolean | null; json: boolean | null },
): boolean {
  if (requires.tools === true && caps.tools !== true) return false;
  if (requires.vision === true && caps.vision !== true) return false;
  if (requires.json === true && caps.json !== true) return false;
  if (
    requires.min_context !== undefined &&
    (caps.context_length === null || caps.context_length < requires.min_context)
  ) {
    return false;
  }
  return true;
}

export function makeRouter(brief: Brief, probes: readonly Probe[]): Router {
  const consecutive = new Map<string, number>();
  const calls = new Map<string, number>();
  for (const probe of probes) {
    consecutive.set(probe.provider.id, 0);
    calls.set(probe.provider.id, 0);
  }

  return {
    candidates(requires: Requirement, _clientApi: 'openai' | 'anthropic', jsonMode: boolean): Candidate[] {
      const out: Candidate[] = [];
      for (const probe of probes) {
        if (!probe.ok) continue;
        if (jsonMode && probe.provider.wire === 'anthropic') continue;
        for (const model of probe.models) {
          const row = ratingRow(brief, probe.provider.id, model.id);
          if (!isFreeModel(model, row)) continue;
          if (!meets(requires, capabilitiesOf(model, row))) continue;
          out.push({ provider: probe.provider, model, rating: row?.rating_milli ?? 0 });
        }
      }
      // Best rating first; ties go to the provider called least so far, so
      // equal models share the load instead of draining one budget.
      out.sort((a, b) => {
        if (b.rating !== a.rating) return b.rating - a.rating;
        const ca = calls.get(a.provider.id) ?? 0;
        const cb = calls.get(b.provider.id) ?? 0;
        if (ca !== cb) return ca - cb;
        return a.provider.id < b.provider.id ? -1 : a.provider.id > b.provider.id ? 1 : 0;
      });
      return out;
    },

    noteRefusal(providerId: string): void {
      consecutive.set(providerId, (consecutive.get(providerId) ?? 0) + 1);
    },

    noteSuccess(providerId: string): void {
      consecutive.set(providerId, 0);
    },

    isDead(providerId: string): boolean {
      return (consecutive.get(providerId) ?? 0) >= DEAD_AFTER_REFUSALS;
    },

    callsOf(providerId: string): number {
      return calls.get(providerId) ?? 0;
    },

    noteCall(providerId: string): void {
      calls.set(providerId, (calls.get(providerId) ?? 0) + 1);
    },
  };
}
