(function () {
  function isEn() { return location.pathname.startsWith('/en/'); }
  function targetPath() {
    var p = location.pathname;
    if (isEn()) return p.replace(/^\/en\//, '/');
    if (p === '/') return '/en/';
    return '/en' + p;
  }

  function build() {
    if (document.getElementById('cipher-lang-switch')) return;
    var box = document.createElement('div');
    box.id = 'cipher-lang-switch';
    box.style.cssText = 'position:fixed;right:16px;top:16px;z-index:99999;background:#111;color:#fff;padding:8px 12px;border-radius:10px;font-size:13px;box-shadow:0 8px 24px rgba(0,0,0,.25)';

    var a = document.createElement('a');
    a.textContent = isEn() ? '切换中文' : 'Switch English';
    a.href = targetPath();
    a.style.cssText = 'color:#fff;text-decoration:none;font-weight:600';

    box.appendChild(a);
    document.body.appendChild(box);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', build);
  } else {
    build();
  }
})();
