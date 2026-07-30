let runtimePromise;

async function getRuntime() {
  if (!runtimePromise) {
    const indexURL = new URL("../vendor/pyodide/", import.meta.url).href;
    runtimePromise = import(new URL("pyodide.mjs", indexURL).href).then(
      ({ loadPyodide }) => loadPyodide({ indexURL }),
    );
  }
  return runtimePromise;
}

self.addEventListener("message", async (event) => {
  const { code, runId } = event.data ?? {};
  if (typeof code !== "string" || typeof runId !== "number") return;

  self.postMessage({ runId, type: "loading" });

  try {
    const pyodide = await getRuntime();
    const stdout = [];
    const stderr = [];

    pyodide.setStdout({ batched: (text) => stdout.push(text) });
    pyodide.setStderr({ batched: (text) => stderr.push(text) });

    const result = await pyodide.runPythonAsync(code);
    const returnValue =
      result === undefined || result === null ? "" : String(result);
    if (result && typeof result.destroy === "function") result.destroy();

    self.postMessage({
      runId,
      type: "result",
      output: [...stdout, ...stderr, returnValue].filter(Boolean).join("\n"),
    });
  } catch (error) {
    self.postMessage({
      runId,
      type: "error",
      output: error instanceof Error ? error.message : String(error),
    });
  }
});
