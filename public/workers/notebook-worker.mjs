// One worker per notebook: the Python namespace survives between cells.
let runtime;
let running = false;
let modelRequestId = 0;
const modelRequests = new Map();
self.addEventListener('message', async ({data}) => {
  if (data?.type === 'model-response') {
    const request = modelRequests.get(data.requestId);
    if (request) {
      modelRequests.delete(data.requestId);
      if (data.error) request.reject(Error(data.error));
      else request.resolve(JSON.stringify(data.result));
    }
    return;
  }
  if (running || typeof data?.code !== 'string' || !Number.isInteger(data.runId)) return;
  running = true;
  const {code, runId} = data;
  let output = '';
  const append = text => { if (output.length < 100000) output += text.slice(0,100000-output.length)+'\n'; };
  try {
    if (!runtime) {
      self.postMessage({type:'loading',runId});
      const indexURL = new URL('../vendor/pyodide/',import.meta.url).href;
      runtime = import(new URL('pyodide.mjs',indexURL).href).then(async ({loadPyodide}) => {
        const p = await loadPyodide({indexURL: indexURL.startsWith('file:') ? new URL(indexURL).pathname : indexURL});
        p.registerJsModule('browser_models', {
          embed: texts => {
            const values = texts.toJs();
            if (!Array.isArray(values) || values.length !== 2 || values.some(x => typeof x !== 'string' || !x.trim() || x.length > 1000)) throw Error('Provide two nonempty strings of at most 1000 characters.');
            const requestId = ++modelRequestId;
            return new Promise((resolve, reject) => {
              modelRequests.set(requestId, {resolve, reject});
              self.postMessage({type:'model-request', requestId, texts:values});
            });
          },
        });
        await p.runPythonAsync(`import json as _nb_json\n_nb_plots = []\ndef display_plot(x, y, title='', xlabel='', ylabel=''):\n    if len(x) != len(y) or not 0 < len(x) <= 2000:\n        raise ValueError('A plot needs 1..2000 paired points')\n    _nb_plots.append(dict(x=list(map(float,x)), y=list(map(float,y)), title=str(title), xlabel=str(xlabel), ylabel=str(ylabel)))`);
        return p;
      });
    }
    const p = await runtime;
    p.setStdout({batched:append}); p.setStderr({batched:append});
    await p.runPythonAsync('_nb_plots = []');
    const result = await p.runPythonAsync(code);
    if (result !== undefined && result !== null) append(String(result));
    result?.destroy?.();
    const plots = JSON.parse(p.runPython('_nb_json.dumps(_nb_plots, allow_nan=False)'));
    self.postMessage({type:'result',runId,output,plots});
  } catch(e) {
    self.postMessage({type:'error',runId,output:output+String(e)});
  } finally {running=false;}
});
