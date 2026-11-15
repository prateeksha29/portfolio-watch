// Fin News service worker: makes the app installable and keeps a copy of every page for offline use.
// - Pages: fetched fresh when the server answers; otherwise (or after 10 s) the saved copy is shown.
// - Saving: the pages ask for a sync on load; at most every 20 minutes the worker fetches the list of all pages
//   (/offline-urls) and saves each one, dropping pages no longer listed.
// - Forms (Ask, refresh, login) always go to the server; the pages disable them while offline.
// Bump VERSION when this file or the static files change.
const VERSION = "v2";
const STATIC = `static-${VERSION}`;
const PAGES = "pages";  // kept across versions: the saved pages stay valid
const META = "meta";
const PRECACHE = ["/offline", "/static/icon-192.png", "/static/apple-touch-icon.png"];
const SYNC_EVERY_MS = 20 * 60 * 1000;
const NAV_TIMEOUT_MS = 10000;
const PARALLEL = 3;

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(STATIC).then((c) => c.addAll(PRECACHE)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => ![STATIC, PAGES, META].includes(k)).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()));
  sync(true);  // save everything right after install or an update
});

self.addEventListener("message", (event) => {
  if (event.data === "sync") event.waitUntil(sync(false));
  if (event.data === "sync-now") event.waitUntil(sync(true));
});

let running = null;

function sync(force) {
  running = running || doSync(force).catch(() => {}).finally(() => { running = null; });
  return running;
}

async function doSync(force) {
  const meta = await caches.open(META);
  const last = await meta.match("/__last-sync");
  if (!force && last && Date.now() - Number(await last.text()) < SYNC_EVERY_MS) return;
  // redirect: "manual" so a login redirect (expired cookie) is never saved as a page
  const list = await fetch("/offline-urls", { redirect: "manual", cache: "no-store" });
  if (!list.ok || list.type !== "basic") return;
  const { urls } = await list.json();
  const pages = await caches.open(PAGES);
  const queue = [...urls];
  let saved = 0;
  const worker = async () => {
    while (queue.length) {
      const url = queue.shift();
      try {
        const resp = await fetch(url, { redirect: "manual", cache: "no-store", headers: { "X-Offline-Sync": "1" } });
        if (resp.ok && resp.type === "basic") { await pages.put(url, resp); saved++; }
      } catch (e) { /* server went away mid-sync: keep the older copy */ }
    }
  };
  await Promise.all(Array.from({ length: PARALLEL }, worker));
  if (!saved) return;
  const keep = new Set(urls.map((u) => new URL(u, self.location.origin).href));
  for (const req of await pages.keys()) if (!keep.has(req.url)) await pages.delete(req);
  await meta.put("/__last-sync", new Response(String(Date.now())));
}

function fromCache(req) {
  return caches.match(req, { ignoreVary: true });
}

self.addEventListener("fetch", (event) => {
  const req = event.request;
  const url = new URL(req.url);
  if (req.method !== "GET" || url.origin !== self.location.origin) return;  // forms go straight through

  if (url.pathname.startsWith("/static/")) {  // icons: cache first
    event.respondWith(fromCache(req).then((hit) => hit || fetch(req).then((resp) => {
      if (resp.ok) caches.open(STATIC).then((c) => c.put(req, resp.clone()));
      return resp;
    })));
    return;
  }

  if (req.mode === "navigate") {  // pages: network first, saved copy when the server doesn't answer
    event.respondWith((async () => {
      const network = fetch(req).then((resp) => {
        // Only real pages are saved: not the login redirect (opaqueredirect) or errors.
        if (resp.ok && resp.type === "basic") {
          const copy = resp.clone();
          caches.open(PAGES).then((c) => c.put(url.href, copy));
        }
        return resp;
      });
      const timeout = new Promise((resolve) => setTimeout(resolve, NAV_TIMEOUT_MS));
      try {
        const first = await Promise.race([network, timeout]);
        if (first) return first;
        const saved = await fromCache(url.href);
        return saved || await network;  // slow server and nothing saved: keep waiting
      } catch (e) {
        return (await fromCache(url.href)) || (await caches.match("/offline"));
      }
    })());
  }
});
