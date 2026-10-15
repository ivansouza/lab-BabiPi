// Web Worker do bev: carrega o modelo e roda a inferência fora da thread principal,
// para a interface (cronômetro, barra de progresso) continuar atualizando.
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.all.bundle.min.mjs';
import { Tokenizer } from 'https://cdn.jsdelivr.net/npm/@huggingface/tokenizers@0.2.0/dist/tokenizers.min.mjs';
import { BevWeb } from './bev-web.js';

const BASE = new URL('./', import.meta.url).href;
let bev = null;

self.onmessage = async ({ data }) => {
  const { id, type } = data;
  try {
    if (type === 'status') {
      postMessage({ id, ok: true, result: await BevWeb.cacheStatus({ base: BASE }) });
    } else if (type === 'load') {
      ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
      let last = -1;
      bev = await BevWeb.load({
        base: BASE, ort, Tokenizer, executionProviders: ['wasm'],
        onProgress: (got, total, fromCache) => {
          const pct = Math.floor(1000 * got / total);
          if (pct !== last) { last = pct; postMessage({ type: 'progress', got, total, fromCache }); }
        },
      });
      postMessage({ type: 'progress', phase: 'warmup' });
      await bev.decide('test', { q: { type: 'noul', instructions: 'ok?' } });
      postMessage({ id, ok: true, result: { ep: bev.eps[0] } });
    } else if (type === 'decide') {
      const t = performance.now();
      const answers = await bev.decide(data.state, data.questions);
      postMessage({ id, ok: true, result: { answers, latency_ms: Math.round(performance.now() - t) } });
    }
  } catch (e) {
    postMessage({ id, ok: false, error: String(e && e.message || e) });
  }
};
