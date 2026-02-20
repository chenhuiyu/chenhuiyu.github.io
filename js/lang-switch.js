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
    box.style.cssText = 'position:fixed;top:16px;right:16px;z-index:2147483647;background:#111;color:#fff;border-radius:10px;padding:8px 12px;font-size:13px;box-shadow:0 6px 20px rgba(0,0,0,.25)';

    var isEn = location.pathname.startsWith('/en/');
    var a = document.createElement('a');
    a.href = targetPath();
    a.textContent = isEn ? '切换中文' : 'Switch English';
    a.style.cssText = 'color:#fff;text-decoration:none;font-weight:600;';
    box.appendChild(a);
    document.body.appendChild(box);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', build);
  } else {
    build();
  }
  setTimeout(build, 1200);
})();
