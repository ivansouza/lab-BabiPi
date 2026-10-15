// Web Worker do Campo Minado 3: LFM2-350M via Transformers.js (modelo baixado do Hugging Face)
import { AutoTokenizer, AutoModelForCausalLM, Tensor, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1';
const MODEL_ID = 'onnx-community/LFM2-350M-ONNX';
const LETRAS = 'ABCDEFGH'.split('');
// Versão WASM própria (no repo): o q4 original usa GatherBlockQuantized, que o WASM não tem;
// aqui o embedding q4 foi reescrito com Gather/BitwiseAnd/BitShift/Mul (resultado idêntico).
const WASM_MODEL = { id: 'lfm2-350m-wasm', host: new URL('./', self.location.href).href, chunks: 4 };
const HF_HOST = env.remoteHost, HF_TEMPLATE = env.remotePathTemplate;
let tok = null, model = null, backend = null, letraIds = null, gpuPerdida = false;
env.allowLocalModels = false;
env.useBrowserCache = true;
env.backends.onnx.wasm.numThreads = 1;

// Forward único pedindo só os logits da última posição (num_logits_to_keep = 1):
// evita ler ~T x 65k logits da GPU, que derrubava o WebGPU no celular.
async function rodar(inp) {
  if (gpuPerdida) throw new Error('GPU perdida (device lost)');
  return model({ input_ids: inp.input_ids, attention_mask: inp.attention_mask, num_logits_to_keep: new Tensor('int64', [1n], []) });
}

async function vigiarGPU() {
  try {
    const dev = await env.backends.onnx.webgpu.device;
    if (dev && dev.lost) dev.lost.then((info) => { gpuPerdida = true; postMessage({ type: 'gpu', estado: 'perdida', erro: info && info.message || '' }); });
  } catch (_) {}
}

async function cacheStatus() {
  if (typeof caches === 'undefined') return { cached: {} };
  const names = await caches.keys(); const cached = {};
  for (const n of names) {
    const c = await caches.open(n);
    for (const req of await c.keys()) {
      if (req.url.includes('lfm2-350m-wasm/onnx/model_q4.onnx_data_3')) { cached.q4wasm = true; continue; }
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
      letraIds = LETRAS.map((l) => tok.encode(' ' + l, { add_special_tokens: false })[0]); // ' A'..' H' depois do prefixo 'Option'
      const tentativas = [];
      if (self.navigator && navigator.gpu && !data.forceWasm) {
        try { if (await navigator.gpu.requestAdapter()) tentativas.push({ device: 'webgpu', dtype: 'q4' }); } catch (e) {}
      }
      // Se havia WebGPU e ela falhou, NÃO troca sozinho para WASM: a página oferece o botão "usar CPU (WASM)".
      if (!tentativas.length) tentativas.push({ device: 'wasm', dtype: 'q4', proprio: true });
      if (model) { try { await model.dispose(); } catch (_) {} model = null; }
      gpuPerdida = false;
      const erros = [];
      for (const t of tentativas) {
        const arquivo = t.proprio ? 'q4wasm' : t.dtype === 'q8' ? 'quantized' : t.dtype;
        const files = new Map();
        const doCache = !!cached[arquivo];
        postMessage({ type: 'progress', phase: 'start', backend: t.device, dtype: t.dtype, fromCache: doCache, proprio: !!t.proprio });
        try {
          if (t.proprio) { env.remoteHost = WASM_MODEL.host; env.remotePathTemplate = '{model}/'; }
          else { env.remoteHost = HF_HOST; env.remotePathTemplate = HF_TEMPLATE; }
          model = await AutoModelForCausalLM.from_pretrained(t.proprio ? WASM_MODEL.id : MODEL_ID, {
            device: t.device, dtype: t.dtype,
            ...(t.proprio ? { use_external_data_format: WASM_MODEL.chunks } : {}),
            progress_callback: (p) => {
              if (p.status !== 'progress' || !p.total) return;
              files.set(p.file, { loaded: p.loaded, total: p.total });
              let got = 0, total = 0; for (const f of files.values()) { got += f.loaded; total += f.total; }
              postMessage({ type: 'progress', got, total, fromCache: doCache, backend: t.device, dtype: t.dtype });
            },
          });
          postMessage({ type: 'progress', phase: 'warmup', backend: t.device });
          backend = t;
          if (t.device === 'webgpu') vigiarGPU();
          await rodar(tok('ok', { add_special_tokens: false }));
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
      const err = new Error('nenhum backend funcionou: ' + erros.join(' | '));
      throw Object.assign(err, { webgpu: tentativas[0].device === 'webgpu' });
    } else if (type === 'decide') {
      const { board, opcoes } = data;
      // Prompt curto (~140 tokens); a resposta do assistente começa com 'Option' e lemos a próxima letra
      const content = `Minesweeper: choose the safest hidden cell to dig (lowest mine risk).\n` +
        opcoes.map((o, i) => `${LETRAS[i]}) cell ${o.r},${o.c}: ${o.risco}% risk`).join('\n') +
        `\nReply with the letter only.`;
      const text = tok.apply_chat_template([{ role: 'user', content }], { tokenize: false, add_generation_prompt: true }) + 'Option';
      const inp = tok(text, { add_special_tokens: false });
      const T = inp.input_ids.dims[1];
      const t0 = performance.now();
      let out;
      try {
        out = await rodar(inp);
      } catch (e) {
        // WebGPU: tenta recriar a sessão uma vez (o modelo já está no cache) e repetir
        if (backend && backend.device === 'webgpu') {
          postMessage({ type: 'gpu', estado: 'recriando', erro: String(e && e.message || e) });
          try { await model.dispose(); } catch (_) {}
          model = await AutoModelForCausalLM.from_pretrained(MODEL_ID, { device: 'webgpu', dtype: backend.dtype });
          gpuPerdida = false; vigiarGPU();
          out = await rodar(inp);
        } else throw e;
      }
      const lg = out.logits, V = lg.dims[lg.dims.length - 1], P = lg.dims[1];
      let row = lg.data.slice((P - 1) * V, P * V);
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
      // Teste de sanidade: logits NaN/infinitos ou <|pad|> no topo = GPU devolvendo lixo
      const topTok = tok.decode([topId]);
      if (!Number.isFinite(logZ) || topTok === '<|pad|>' || topId === tok.pad_token_id) {
        throw Object.assign(new Error(`saída inválida do modelo (top="${topTok}", logZ=${logZ})`), { webgpu: backend && backend.device === 'webgpu' });
      }
      postMessage({ id, ok: true, result: { logprobs, top: topTok, tokens: T, ms: Math.round(performance.now() - t0) } });
    }
  } catch (e) {
    postMessage({ id, ok: false, error: String(e && e.message || e), webgpu: !!(e && e.webgpu) || (backend && backend.device === 'webgpu' && type === 'decide') });
  }
};
