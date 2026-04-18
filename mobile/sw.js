const CACHE_NAME = 'fraudshield-v1';
const ASSETS = ['/mobile/', '/mobile/index.html', '/mobile/style.css', '/mobile/app.js'];

self.addEventListener('install', e => {
    e.waitUntil(caches.open(CACHE_NAME).then(c => c.addAll(ASSETS)));
});

self.addEventListener('fetch', e => {
    e.respondWith(
        caches.match(e.request).then(r => r || fetch(e.request))
    );
});
