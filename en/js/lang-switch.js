(function () {
  function targetPath() {
    var p = location.pathname;
    if (p.startsWith('/en/')) return p.replace(/^\/en\//, '/');
    if (p === '/en') return '/';
    if (p === '/') return '/en/';
    return '/en' + p;
  }

  function build() {
    if (document.getElementById('cipher-lang-switch')) return;
    var box = document.createElement('div');
    box.id = 'cipher-lang-switch';
    box.style.cssText = 'position:fixed;right:16px;top:82px;z-index:2147483647;background:rgba(0,0,0,.62);color:#fff;padding:8px 12px;border-radius:12px;font-size:13px;backdrop-filter:blur(4px)';

    var a = document.createElement('a');
    var isEn = location.pathname.startsWith('/en/');
    a.textContent = isEn ? '切换中文' : 'Switch English';
    a.href = targetPath() + location.search + location.hash;
    a.style.cssText = 'color:#fff;text-decoration:none;font-weight:600';

    box.appendChild(a);
    document.body.appendChild(box);
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', build);
  else build();
})();
