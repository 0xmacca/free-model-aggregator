// The wires: translating requests and answers between the OpenAI Chat
// Completions wire, the Anthropic Messages wire and the Gemini generateContent
// wire, in JSON and in server-sent events.
//
// The rule that shapes every translation: the provider echoes what it
// parsed, and the client compares that echo with what it sent. So a
// translation must carry every echoed field across untouched:
//
//   system text, the last user message's text, the tool names in order,
//   max_tokens, the stop sequences in order, the image (its base64 bytes or
//   its URL), and JSON mode.
//
// A provider answers plain text only (its stop is always the plain "stop"
// family), so answers never carry tool calls; the translations below still
// handle a tool call if one appears, so a shape change on the provider's
// side degrades to text instead of breaking the response.
//
// Everything here is a pure function of its arguments: no clock and no
// randomness, so the same body and the same provider answer give the same
// bytes on a replay. Streaming translations read the whole upstream stream
// first and then emit the client's event sequence: the streams are a few
// hundred bytes, and buffering keeps the event order exact (the Anthropic
// message_start carries the input tokens, which only the stream's last
// frame knows).

import { isRecord, type Json } from './brief.ts';

export interface Usage {
  input: number;
  output: number;
}

/** A fixed epoch second for every OpenAI-shaped `created`: nothing reads the clock. */
const FIXED_CREATED = 1767225600;

function numOrZero(v: unknown): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : 0;
}

function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

/** Split text the way the providers split their streams: 13 characters a chunk. */
export function splitText(text: string): string[] {
  const out: string[] = [];
  for (let at = 0; at < text.length; at += 13) out.push(text.slice(at, at + 13));
  return out.length === 0 ? [''] : out;
}

/** One server-sent event frame. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** One SSE text as frames: {event, data} pairs, blank lines as separators. */
export function parseSse(text: string): { event: string | null; data: string }[] {
  const frames: { event: string | null; data: string }[] = [];
  for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function jsonOf(data: string): Json | null {
  try {
    const v: unknown = JSON.parse(data);
    return isRecord(v) ? v : null;
  } catch {
    return null;
  }
}

/* ------------------------------------------------------------------ */
/* Shared request pieces                                               */
/* ------------------------------------------------------------------ */

/** A tool call's arguments as an object: the provider's JSON string, or {}. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** An image_url part as its base64 data and media type, or its plain URL. */
function imageData(url: unknown): { mime: string; data: string } | { url: string } | null {
  if (typeof url !== 'string' || url === '') return null;
  const m = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  if (m !== null) return { mime: m[1] !== '' ? m[1] : 'image/png', data: m[2] };
  return { url };
}

/** An OpenAI content part as an Anthropic content block: text or image. */
function anthropicBlockOf(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url)) {
    const img = imageData(part.image_url.url);
    if (img === null) return null;
    if ('data' in img) {
      return { type: 'image', source: { type: 'base64', media_type: img.mime, data: img.data } };
    }
    return { type: 'image', source: { type: 'url', url: img.url } };
  }
  return null;
}

/** An Anthropic content block as an OpenAI content part: text or image. */
function openAIPartOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') {
      return { type: 'image_url', image_url: { url: source.url } };
    }
  }
  return null;
}

/** A tool_result block's content as text. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** OpenAI's finish_reason as Anthropic's stop_reason (the plain family). */
export function anthropicStopOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/** Anthropic's stop_reason as OpenAI's finish_reason (the plain family). */
export function openAIStopOf(stop: unknown): string {
  if (stop === 'max_tokens') return 'length';
  if (stop === 'tool_use') return 'tool_calls';
  if (stop === 'refusal') return 'content_filter';
  return 'stop';
}

function stopList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (Array.isArray(v)) return v.filter((x): x is string => typeof x === 'string');
  return [];
}

/* ------------------------------------------------------------------ */
/* Requests: OpenAI client                                             */
/* ------------------------------------------------------------------ */

/** An OpenAI request for an OpenAI provider: the same body, the routed model, usage on streams. */
export function openAIToOpenAI(body: Json, model: string): Json {
  const out: Json = { ...body, model };
  if (body.stream === true) out.stream_options = { include_usage: true };
  return out;
}

/** An OpenAI request as an Anthropic Messages request naming `model`. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const systemParts: string[] = [];
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  let pendingToolResults: Json[] = [];
  const flushToolResults = (): void => {
    if (pendingToolResults.length > 0) {
      messages.push({ role: 'user', content: pendingToolResults });
      pendingToolResults = [];
    }
  };
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = raw.role;
    if (role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (role === 'tool') {
      pendingToolResults.push({
        type: 'tool_result',
        tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
        content: toolResultText(raw.content),
      });
      continue;
    }
    flushToolResults();
    if (role === 'assistant') {
      const blocks: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      messages.push({ role: 'assistant', content: blocks.length > 0 ? blocks : '' });
      continue;
    }
    // A user message: text and image parts, in order.
    if (typeof raw.content === 'string') {
      messages.push({ role: 'user', content: raw.content });
      continue;
    }
    if (!Array.isArray(raw.content)) continue;
    const blocks: Json[] = [];
    for (const part of raw.content) {
      if (!isRecord(part)) continue;
      const block = anthropicBlockOf(part);
      if (block !== null) blocks.push(block);
    }
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }
  flushToolResults();

  const out: Json = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  const maxTokens =
    typeof body.max_tokens === 'number'
      ? body.max_tokens
      : typeof body.max_completion_tokens === 'number'
        ? body.max_completion_tokens
        : undefined;
  if (maxTokens !== undefined) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stopList(body.stop);
  if (stops.length > 0) out.stop_sequences = stops;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters)
            ? raw.function.parameters
            : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required' || choice === 'any') out.tool_choice = { type: 'any' };
  else if (choice === 'none') out.tool_choice = { type: 'none' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function)) {
    out.tool_choice = { type: 'tool', name: typeof choice.function.name === 'string' ? choice.function.name : '' };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** An OpenAI request as a Gemini generateContent request naming `model`. */
export function openAIToGemini(body: Json, model: string): Json {
  const { contents, systemText } = openAIContents(body);
  const out: Json = { model, contents };
  if (systemText !== '') out.systemInstruction = { parts: [{ text: systemText }] };
  const config: Json = {};
  const maxTokens =
    typeof body.max_tokens === 'number'
      ? body.max_tokens
      : typeof body.max_completion_tokens === 'number'
        ? body.max_completion_tokens
        : undefined;
  if (maxTokens !== undefined) config.maxOutputTokens = maxTokens;
  const stops = stopList(body.stop);
  if (stops.length > 0) config.stopSequences = stops;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (isRecord(body.response_format)) {
    const format = body.response_format.type;
    if (format === 'json_object' || format === 'json_schema') config.responseMimeType = 'application/json';
  }
  if (Object.keys(config).length > 0) out.generationConfig = config;
  if (Array.isArray(body.tools)) {
    const decls = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          parameters: isRecord(raw.function.parameters)
            ? raw.function.parameters
            : { type: 'object', properties: {} },
        },
      ];
    });
    if (decls.length > 0) out.tools = [{ functionDeclarations: decls }];
  }
  return out;
}

/** An OpenAI request's messages as Gemini contents, with the system text aside. */
function openAIContents(body: Json): { contents: Json[]; systemText: string } {
  const systemParts: string[] = [];
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'tool') {
      contents.push({
        role: 'function',
        parts: [
          {
            functionResponse: {
              name: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
              response: { content: toolResultText(raw.content) },
            },
          },
        ],
      });
      continue;
    }
    const parts: Json[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') parts.push({ text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const part of raw.content) {
        if (!isRecord(part)) continue;
        if (part.type === 'text' && typeof part.text === 'string') parts.push({ text: part.text });
        else if (part.type === 'image_url' && isRecord(part.image_url)) {
          const img = imageData(part.image_url.url);
          if (img === null) continue;
          if ('data' in img) parts.push({ inlineData: { mimeType: img.mime, data: img.data } });
          else parts.push({ fileData: { fileUri: img.url } });
        }
      }
    }
    if (raw.role === 'assistant') {
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        parts.push({
          functionCall: {
            name: typeof call.function.name === 'string' ? call.function.name : '',
            args: argumentsOf(call.function.arguments),
          },
        });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  return { contents, systemText: systemParts.join('\n') };
}

/* ------------------------------------------------------------------ */
/* Requests: Anthropic client                                          */
/* ------------------------------------------------------------------ */

/** An Anthropic request as an OpenAI Chat Completions request naming `model`. */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = openAIPartOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stopList(body.stop_sequences);
  if (stops.length > 0) out.stop = stops;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'none') out.tool_choice = 'none';
    else if (choice.type === 'tool' && typeof choice.name === 'string') {
      out.tool_choice = { type: 'function', function: { name: choice.name } };
    }
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** An Anthropic request for an Anthropic provider: the same body, the routed model. */
export function anthropicToAnthropic(body: Json, model: string): Json {
  return { ...body, model };
}

/** An Anthropic request as a Gemini generateContent request naming `model`. */
export function anthropicToGemini(body: Json, model: string): Json {
  const systemText = textOf(body.system);
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const parts: Json[] = [];
    const content = raw.content;
    if (typeof content === 'string') {
      if (content !== '') parts.push({ text: content });
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
        else if (block.type === 'image' && isRecord(block.source)) {
          const src = block.source;
          if (src.type === 'base64' && typeof src.data === 'string') {
            parts.push({
              inlineData: {
                mimeType: typeof src.media_type === 'string' ? src.media_type : 'image/png',
                data: src.data,
              },
            });
          } else if (src.type === 'url' && typeof src.url === 'string') {
            parts.push({ fileData: { fileUri: src.url } });
          }
        } else if (block.type === 'tool_use') {
          parts.push({
            functionCall: {
              name: typeof block.name === 'string' ? block.name : '',
              args: isRecord(block.input) ? block.input : {},
            },
          });
        } else if (block.type === 'tool_result') {
          parts.push({
            functionResponse: {
              name: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
              response: { content: toolResultText(block.content) },
            },
          });
        }
      }
    }
    if (parts.length === 0) continue;
    contents.push({ role: raw.role === 'assistant' ? 'model' : 'user', parts });
  }
  const out: Json = { model, contents };
  if (systemText !== '') out.systemInstruction = { parts: [{ text: systemText }] };
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  const stops = stopList(body.stop_sequences);
  if (stops.length > 0) config.stopSequences = stops;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (Object.keys(config).length > 0) out.generationConfig = config;
  if (Array.isArray(body.tools)) {
    const decls = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          name: raw.name,
          description: typeof raw.description === 'string' ? raw.description : '',
          parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
        },
      ];
    });
    if (decls.length > 0) out.tools = [{ functionDeclarations: decls }];
  }
  return out;
}

/* ------------------------------------------------------------------ */
/* Answers: JSON                                                       */
/* ------------------------------------------------------------------ */

/** The text of a Chat Completions answer's first choice. */
function openAIText(body: Json): string {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  return typeof message.content === 'string' ? message.content : textOf(message.content);
}

/** A Chat Completions answer as an Anthropic Messages object. */
export function openAIToAnthropicAnswer(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = openAIText(body);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: anthropicStopOf(choice.finish_reason),
    stop_sequence: null,
    usage: {
      input_tokens: numOrZero(usage.prompt_tokens),
      output_tokens: numOrZero(usage.completion_tokens),
    },
  };
}

/** An Anthropic Messages answer as a Chat Completions object. */
export function anthropicToOpenAIAnswer(body: Json, model: string): Json {
  const content = Array.isArray(body.content) ? body.content : [];
  const text = content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('');
  const usage = isRecord(body.usage) ? body.usage : {};
  const input = numOrZero(usage.input_tokens);
  const output = numOrZero(usage.output_tokens);
  return {
    id: `chatcmpl-${typeof body.id === 'string' ? body.id : 'sim'}`,
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: text },
        finish_reason: openAIStopOf(body.stop_reason),
        logprobs: null,
      },
    ],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  };
}

/** The text of a Gemini generateContent answer's first candidate. */
function geminiText(body: Json): string {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(first.content) ? first.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  return parts
    .flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : []))
    .join('');
}

/** A Gemini answer's usage: prompt and candidate token counts. */
function geminiUsage(body: Json): Usage {
  const meta = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return { input: numOrZero(meta.promptTokenCount), output: numOrZero(meta.candidatesTokenCount) };
}

/** A Gemini generateContent answer as a Chat Completions object. */
export function geminiToOpenAIAnswer(body: Json, model: string): Json {
  const usage = geminiUsage(body);
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: geminiText(body) },
        finish_reason: 'stop',
        logprobs: null,
      },
    ],
    usage: {
      prompt_tokens: usage.input,
      completion_tokens: usage.output,
      total_tokens: usage.input + usage.output,
    },
  };
}

/** A Gemini generateContent answer as an Anthropic Messages object. */
export function geminiToAnthropicAnswer(body: Json, model: string): Json {
  const usage = geminiUsage(body);
  return {
    id: 'msg_sim',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: geminiText(body) }],
    stop_reason: 'end_turn',
    stop_sequence: null,
    usage: { input_tokens: usage.input, output_tokens: usage.output },
  };
}

/* ------------------------------------------------------------------ */
/* Answers: streams                                                    */
/* ------------------------------------------------------------------ */

/** A Chat Completions stream as text, usage and finish reason. */
export function parseOpenAIStream(text: string): { text: string; usage: Usage | null; finish: string | null } {
  let out = '';
  let usage: Usage | null = null;
  let finish: string | null = null;
  for (const frame of parseSse(text)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') continue;
    const chunk = jsonOf(payload);
    if (chunk === null) continue;
    if (isRecord(chunk.usage)) {
      usage = {
        input: numOrZero(chunk.usage.prompt_tokens),
        output: numOrZero(chunk.usage.completion_tokens),
      };
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    if (choices.length === 0) continue;
    const choice = isRecord(choices[0]) ? choices[0] : {};
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') out += delta.content;
    if (typeof choice.finish_reason === 'string') finish = choice.finish_reason;
  }
  return { text: out, usage, finish };
}

/** An Anthropic stream as text, usage and stop reason. */
export function parseAnthropicStream(text: string): { text: string; usage: Usage | null; stop: string | null } {
  let out = '';
  let input: number | null = null;
  let output: number | null = null;
  let stop: string | null = null;
  for (const frame of parseSse(text)) {
    const data = jsonOf(frame.data);
    if (data === null) continue;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      input = numOrZero(data.message.usage.input_tokens);
    }
    if (frame.event === 'content_block_delta' && isRecord(data.delta) && data.delta.type === 'text_delta') {
      if (typeof data.delta.text === 'string') out += data.delta.text;
    }
    if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stop = data.delta.stop_reason;
      if (isRecord(data.usage)) output = numOrZero(data.usage.output_tokens);
    }
  }
  return { text: out, usage: input === null && output === null ? null : { input: input ?? 0, output: output ?? 0 }, stop };
}

/** A Gemini stream as text, usage and finish reason. */
export function parseGeminiStream(text: string): { text: string; usage: Usage | null; finish: string | null } {
  let out = '';
  let usage: Usage | null = null;
  let finish: string | null = null;
  for (const frame of parseSse(text)) {
    const data = jsonOf(frame.data);
    if (data === null) continue;
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const first = isRecord(candidates[0]) ? candidates[0] : {};
    const content = isRecord(first.content) ? first.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    for (const part of parts) {
      if (isRecord(part) && typeof part.text === 'string') out += part.text;
    }
    if (typeof first.finishReason === 'string') finish = first.finishReason;
    if (isRecord(data.usageMetadata)) {
      usage = {
        input: numOrZero(data.usageMetadata.promptTokenCount),
        output: numOrZero(data.usageMetadata.candidatesTokenCount),
      };
    }
  }
  return { text: out, usage, finish };
}

/** Text and usage as an OpenAI event stream: role, deltas, finish, usage, [DONE]. */
export function emitOpenAIStream(text: string, usage: Usage | null, model: string): string {
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-sim',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  for (const piece of splitText(text)) frames.push(sseEvent(null, chunk({ content: piece }, null)));
  frames.push(sseEvent(null, chunk({}, 'stop')));
  if (usage !== null) {
    frames.push(
      sseEvent(null, {
        id: 'chatcmpl-sim',
        object: 'chat.completion.chunk',
        created: FIXED_CREATED,
        model,
        choices: [],
        usage: {
          prompt_tokens: usage.input,
          completion_tokens: usage.output,
          total_tokens: usage.input + usage.output,
        },
      }),
    );
  }
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** Text and usage as an Anthropic event stream: start, one text block, delta, stop. */
export function emitAnthropicStream(text: string, usage: Usage | null, model: string): string {
  const input = usage?.input ?? 0;
  const output = usage?.output ?? 0;
  const frames = [
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_stream',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: input, output_tokens: 0 },
      },
    }),
    sseEvent('content_block_start', {
      type: 'content_block_start',
      index: 0,
      content_block: { type: 'text', text: '' },
    }),
  ];
  for (const piece of splitText(text)) {
    frames.push(
      sseEvent('content_block_delta', {
        type: 'content_block_delta',
        index: 0,
        delta: { type: 'text_delta', text: piece },
      }),
    );
  }
  frames.push(
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: 'end_turn', stop_sequence: null },
      usage: { output_tokens: output },
    }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}

/** A Chat Completions stream as the Anthropic event sequence. */
export function openAIStreamToAnthropic(text: string, model: string): string {
  const parsed = parseOpenAIStream(text);
  return emitAnthropicStream(parsed.text, parsed.usage, model);
}

/** An Anthropic stream as the Chat Completions event sequence. */
export function anthropicStreamToOpenAI(text: string, model: string): string {
  const parsed = parseAnthropicStream(text);
  return emitOpenAIStream(parsed.text, parsed.usage, model);
}

/** A Gemini stream as the Chat Completions event sequence. */
export function geminiStreamToOpenAI(text: string, model: string): string {
  const parsed = parseGeminiStream(text);
  return emitOpenAIStream(parsed.text, parsed.usage, model);
}

/** A Gemini stream as the Anthropic event sequence. */
export function geminiStreamToAnthropic(text: string, model: string): string {
  const parsed = parseGeminiStream(text);
  return emitAnthropicStream(parsed.text, parsed.usage, model);
}
