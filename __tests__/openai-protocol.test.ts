/** @jest-environment node */
import { z } from 'zod';

jest.mock('@ai-sdk/openai', () => ({
  createOpenAI: () => ({
    responses: (id: string) => ({ id, protocol: 'responses' }),
    chat: (id: string) => ({ id, protocol: 'chat' }),
  }),
}));
jest.mock('ai', () => ({
  gateway: (id: string) => ({ id, protocol: 'gateway' }),
  generateText: jest.fn(async () => ({
    text: '红色方块',
    output: { title: '红色方块' },
  })),
  streamText: jest.fn(() => ({
    textStream: (async function* () { yield '红色方块'; })(),
  })),
  Output: { object: jest.fn() },
}));
jest.mock('@ai-sdk/rsc', () => ({
  createStreamableValue: () => ({
    value: 'stream', update: jest.fn(), done: jest.fn(),
  }),
}));
jest.mock('@/platforms/rate-limit', () => ({
  checkRateLimitAndThrow: jest.fn(),
}));
jest.mock('@/utility/image', () => ({
  removeBase64Prefix: (image: string) => image.split(',')[1],
}));

const image = 'data:image/png;base64,' +
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a' +
  'N1sAAAAASUVORK5CYII=';
const schema = z.object({ title: z.string() });

function load(mode = 'responses', provider = 'openai', hasKey = true) {
  jest.resetModules();
  jest.doMock('@/app/config', () => ({
    OPENAI_SECRET_KEY: hasKey ? 'test-key' : undefined,
    OPENAI_API_MODE: mode,
    OPENAI_MODEL: 'gemini-3.8-flash',
    OPENAI_BASE_URL: 'https://api.test/v1',
    AI_GATEWAY_MODEL: 'google/gemini-2.5-flash',
    AI_CONTENT_GENERATION_PROVIDER: provider,
  }));
  return require('@/platforms/openai') as
    typeof import('@/platforms/openai');
}

it.each(['responses', 'chat'])(
  '%s applies to text, streaming, objects, tests and explicit models',
  async protocol => {
    const api = load(protocol);
    const { generateText, streamText } = require('ai');
    await api.generateOpenAiImageQuery(image, 'Describe');
    await api.generateOpenAiImageObjectQuery(image, 'Describe', schema);
    await api.testOpenAiConnection();
    await api.generateOpenAiImageQueryForModel(image, 'Describe', 'custom');
    await api.generateOpenAiImageObjectQueryForModel(
      image, 'Describe', schema, 'custom',
    );
    await api.streamOpenAiImageQuery(image, 'Describe');
    expect(generateText).toHaveBeenCalledTimes(5);
    for (const [args] of generateText.mock.calls) {
      expect(args.model.protocol).toBe(protocol);
    }
    expect(generateText.mock.calls[3][0].model.id).toBe('custom');
    expect(generateText.mock.calls[4][0].model.id).toBe('custom');
    expect(streamText.mock.calls[0][0].model.protocol).toBe(protocol);
    expect(generateText.mock.calls[0][0].messages[0].content[1])
      .toEqual({ type: 'file', mediaType: 'image', data: image.split(',')[1] });
  },
);

it('keeps Gateway independent of the direct API protocol', async () => {
  const api = load('chat', 'gateway', false);
  const { generateText } = require('ai');
  await api.testOpenAiConnection();
  expect(generateText.mock.calls[0][0].model.protocol).toBe('gateway');
  await expect(api.generateOpenAiImageQueryForModel(image, '', 'custom'))
    .rejects.toThrow('OPENAI_SECRET_KEY required');
});

it('rejects an invalid protocol instead of silently using another API', () => {
  expect(() => load('invalid')).toThrow('OPENAI_API_MODE');
});

it('sends a PNG and JSON schema through the real Chat Completions adapter',
  async () => {
    jest.dontMock('ai');
    jest.dontMock('@ai-sdk/openai');
    const fetch = jest.spyOn(global, 'fetch').mockResolvedValue(new Response(
      JSON.stringify({
        id: 'test', object: 'chat.completion', created: 1,
        model: 'gemini-3.8-flash',
        choices: [{
          index: 0, finish_reason: 'stop',
          message: { role: 'assistant', content: '{"title":"红色方块"}' },
        }],
        usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
      }),
      { headers: { 'content-type': 'application/json' } },
    ));
    try {
      const api = load('chat');
      const result = await api.generateOpenAiImageObjectQuery(
        image, 'Describe', schema,
      );
      expect(result).toEqual({ title: '红色方块' });
      expect(fetch).toHaveBeenCalledTimes(1);
      const [url, options] = fetch.mock.calls[0];
      expect(url).toBe('https://api.test/v1/chat/completions');
      const body = JSON.parse(options?.body as string);
      expect(body.model).toBe('gemini-3.8-flash');
      expect(body.messages[0].content[1].image_url.url).toBe(image);
      expect(body.response_format.type).toBe('json_schema');
    } finally {
      fetch.mockRestore();
    }
  },
);
