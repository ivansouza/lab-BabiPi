// Web Worker do Campo Minado 3: LFM2-350M via Transformers.js (modelo baixado do Hugging Face)
import { AutoTokenizer, AutoModelForCausalLM, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1';
const MODEL_ID = 'onnx-community/LFM2-350M-ONNX';
const LETRAS = 'ABCDEFGH'.split('');
let tok = null, model = null, backend = null, letraIds = null;
env.allowLocalModels = false;
env.useBrowserCache = true;
env.backends.onnx.wasm.numThreads = 1;

async function cacheStatus() {
  if (typeof caches === 'undefined') return { cached: {} };
  const names = await caches.keys(); const cached = {};
  for (const n of names) {
    const c = await caches.open(n);
    for (const req of await c.keys()) {
      if (!req.url.includes('LFM2-350M-ONNX')) continue;
      const m = req.url.match(/onnx\/model_(q4f16|q4|quantized|fp16)\.onnx_data/);
      if (m) cached[m[1]] = true;
    }
  }
  return { cached };
}

self.onmessage = async ({ data }) => {
  const { id, type } = data;
  try {
    if (type === 'status') {
      postMessage({ id, ok: true, result: await cacheStatus() });
    } else if (type === 'load') {
      const { cached } = await cacheStatus();
      tok = await AutoTokenizer.from_pretrained(MODEL_ID);
      letraIds = LETRAS.map((l) => tok.encode(l, { add_special_tokens: false })[0]);
      const tentativas = [];
      if (self.navigator && navigator.gpu && !data.forceWasm) {
        try { if (await navigator.gpu.requestAdapter()) tentativas.push({ device: 'webgpu', dtype: 'q4f16' }); } catch (e) {}
      }
      // q4 e q8 usam GatherBlockQuantized, que o WASM do onnxruntime-web não implementa: no WASM vai direto para fp16
      tentativas.push({ device: 'wasm', dtype: 'fp16' });
      const erros = [];
      for (const t of tentativas) {
        const arquivo = t.dtype === 'q8' ? 'quantized' : t.dtype;
        const files = new Map();
        const doCache = !!cached[arquivo];
        postMessage({ type: 'progress', phase: 'start', backend: t.device, dtype: t.dtype, fromCache: doCache });
        try {
          model = await AutoModelForCausalLM.from_pretrained(MODEL_ID, {
            device: t.device, dtype: t.dtype,
            progress_callback: (p) => {
              if (p.status !== 'progress' || !p.total) return;
              files.set(p.file, { loaded: p.loaded, total: p.total });
              let got = 0, total = 0; for (const f of files.values()) { got += f.loaded; total += f.total; }
              postMessage({ type: 'progress', got, total, fromCache: doCache, backend: t.device, dtype: t.dtype });
            },
          });
          postMessage({ type: 'progress', phase: 'warmup', backend: t.device });
          const inp = tok('ok', { add_special_tokens: false });
          await model(inp);
          backend = t;
          let bytes = 0; for (const f of files.values()) bytes += f.total;
          postMessage({ id, ok: true, result: { device: t.device, dtype: t.dtype, bytes, fromCache: doCache, erros } });
          return;
        } catch (e) {
          erros.push(`${t.device}/${t.dtype}: ${e && e.message || e}`);
          postMessage({ type: 'progress', phase: 'fallback', erro: erros[erros.length - 1] });
          try { model && model.dispose && await model.dispose(); } catch (_) {}
          model = null;
        }
      }
      throw new Error('nenhum backend funcionou: ' + erros.join(' | '));
    } else if (type === 'decide') {
      const { board, opcoes } = data;
      const content = `Minesweeper. Board (rows and columns are 0-indexed; ? = hidden, F = flag, _ = empty, numbers = adjacent mines):\n${board}\n\n` +
        `No certain move exists. Which hidden cell is the safest to dig? Options:\n` +
        opcoes.map((o, i) => `${LETRAS[i]}) cell ${o.r},${o.c}, estimated mine risk ${o.risco}%`).join('\n') +
        `\n\nAnswer with only the letter of the safest option.`;
      const text = tok.apply_chat_template([{ role: 'user', content }], { tokenize: false, add_generation_prompt: true });
      const inp = tok(text, { add_special_tokens: false });
      const t0 = performance.now();
      const out = await model(inp);
      const lg = out.logits, [, T, V] = lg.dims;
      let row = lg.data.slice((T - 1) * V, T * V);
      if (row instanceof Uint16Array) { // float16 em Uint16Array
        const f = new Float32Array(V);
        for (let i = 0; i < V; i++) { const h = row[i], s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
          f[i] = e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15); }
        row = f;
      }
      let mx = -Infinity; for (let i = 0; i < V; i++) { const x = Number(row[i]); if (x > mx) mx = x; }
      let z = 0; for (let i = 0; i < V; i++) z += Math.exp(Number(row[i]) - mx);
      const logZ = mx + Math.log(z);
      const logprobs = opcoes.map((_, i) => Number(row[letraIds[i]]) - logZ);
      let topId = 0; for (let i = 1; i < V; i++) if (Number(row[i]) > Number(row[topId])) topId = i;
      postMessage({ id, ok: true, result: { logprobs, top: tok.decode([topId]), tokens: T, ms: Math.round(performance.now() - t0) } });
    }
  } catch (e) {
    postMessage({ id, ok: false, error: String(e && e.message || e) });
  }
};
