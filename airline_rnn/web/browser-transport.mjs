// Run the trained model in a worker so the eight stages never block the UI.
const worker = new Worker(new URL('./browser-worker.mjs', import.meta.url), {type: 'module'});

async function* request(data, signal) {
  const channel = new MessageChannel(), queue = [];
  let wake, finished = false;
  const notify = event => {queue.push(event); wake?.();};
  const fail = () => notify({type: 'error', error: 'Không khởi tạo được mô hình trong trình duyệt.'});
  const abort = () => {finished = true; wake?.();};
  channel.port1.onmessage = event => notify(event.data);
  worker.addEventListener('error', fail);
  signal?.addEventListener('abort', abort, {once: true});
  if (signal?.aborted) finished = true;
  const timer = setTimeout(() => notify({type: 'error', error: 'Tải mô hình quá lâu. Kiểm tra mạng và tải lại trang.'}), 120000);
  if (!finished) worker.postMessage(data, [channel.port2]);
  try {
    while (!finished) {
      if (!queue.length) await new Promise(resolve => {wake = resolve;});
      wake = undefined;
      while (queue.length && !finished) {
        const event = queue.shift();
        if (event.type === 'end') {finished = true; break;}
        if (event.type === 'error') throw Error(event.error);
        yield event;
      }
    }
  } finally {
    clearTimeout(timer);
    channel.port1.close();
    worker.removeEventListener('error', fail);
    signal?.removeEventListener('abort', abort);
  }
}

window.AirlineDemoBackend = {
  async project() {
    const response = await fetch(new URL('./project-info.json', import.meta.url));
    if (!response.ok) throw Error('Không tải được thông tin đề tài.');
    return response.json();
  },
  async health() {
    for await (const event of request({type: 'health'})) if (event.status) return event;
    throw Error('Mô hình chưa sẵn sàng.');
  },
  analyze(text, signal) {return request({type: 'analyze', text}, signal);},
};
await import('./app.js');
