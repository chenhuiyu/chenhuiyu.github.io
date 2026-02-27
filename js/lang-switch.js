(function () {
  function isEnPath(pathname) {
    return pathname === '/en' || pathname === '/en/' || pathname.indexOf('.en/') !== -1 || pathname.startsWith('/en/');
  }

  function toCounterpart(pathname) {
    var p = pathname;

    // homepage switch
    if (p === '/' || p === '') return '/en/';
    if (p === '/en' || p === '/en/') return '/';

    // these sections don't have /en/* routes in current Hexo setup
    // avoid sending users to guaranteed 404 pages
    var noEnPrefixes = ['/categories', '/tags', '/archives', '/travel'];
    for (var i = 0; i &lt; noEnPrefixes.length; i++) {
      var prefix = noEnPrefixes[i];
      if (p === prefix || p.indexOf(prefix + '/') === 0) return '/en/';
      if (p === '/en' + prefix || p.indexOf('/en' + prefix + '/') === 0) return '/';
    }

    // post/page with explicit language suffix
    if (p.indexOf('.zh-CN/') !== -1) return p.replace('.zh-CN/', '.en/');
    if (p.indexOf('.en/') !== -1) return p.replace('.en/', '.zh-CN/');

    // generic fallback for /en/*
    if (p.startsWith('/en/')) return p.replace(/^\/en\//, '/');

    // generic fallback for zh -&gt; en
    return '/en' + p;
  }

  function createSwitchLink() {
    var a = document.createElement('a');
    var en = isEnPath(location.pathname);

    a.id = 'cipher-lang-switch';
    a.href = toCounterpart(location.pathname) + location.search + location.hash;
    a.textContent = en ? '中文' : 'EN';
    a.title = en ? '切换到中文' : 'Switch to English';
    a.setAttribute('aria-label', a.title);

    a.style.cssText = [
      'display:inline-flex',
      'align-items:center',
      'justify-content:center',
      'height:32px',
      'padding:0 12px',
      'border-radius:999px',
      'border:1px solid rgba(255,255,255,.42)',
      'color:#fff',
      'background:rgba(0,0,0,.28)',
      'font-size:13px',
      'font-weight:700',
      'text-decoration:none',
      'line-height:1',
      'backdrop-filter:blur(4px)'
    ].join(';');

    a.addEventListener('mouseenter', function () {
      a.style.background = 'rgba(0,0,0,.45)';
    });
    a.addEventListener('mouseleave', function () {
      a.style.background = 'rgba(0,0,0,.28)';
    });

    return a;
  }

  function mountInNav(a) {
    var nav = document.getElementById('nav');
    if (!nav) return false;

    if (document.getElementById('cipher-lang-switch')) return true;

    var wrap = document.createElement('div');
    wrap.id = 'cipher-lang-switch-wrap';
    wrap.style.cssText = [
      'position:absolute',
      'right:16px',
      'top:50%',
      'transform:translateY(-50%)',
      'z-index:10'
    ].join(';');

    if (!nav.style.position) nav.style.position = 'relative';

    wrap.appendChild(a);
    nav.appendChild(wrap);
    return true;
  }

  function mountFallback(a) {
    if (document.getElementById('cipher-lang-switch')) return;
    var wrap = document.createElement('div');
    wrap.id = 'cipher-lang-switch-wrap';
    wrap.style.cssText = 'position:fixed;right:16px;top:82px;z-index:2147483647';
    wrap.appendChild(a);
    document.body.appendChild(wrap);
  }

  function init() {
    var link = createSwitchLink();
    if (!mountInNav(link)) {
      mountFallback(link);
      return;
    }

    // Re-mount after async navbar updates / pjax
    setTimeout(function () {
      if (!document.getElementById('cipher-lang-switch')) mountFallback(createSwitchLink());
    }, 1200);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
