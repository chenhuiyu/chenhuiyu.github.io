(function () {
  function togglePath(path) {
    if (path.startsWith('/en/')) return path.replace(/^\/en\//, '/');
    if (path === '/en') return '/';
    if (path === '/') return '/en/';
    return '/en' + path;
  }

  function exists(url) {
    return fetch(url, { method: 'HEAD' }).then(r =&gt; r.ok).catch(() =&gt; false);
  }

  function buildBtn() {
    var wrap = document.createElement('div');
    wrap.style.cssText = 'position:fixed;right:18px;bottom:92px;z-index:99999;background:rgba(0,0,0,.62);color:#fff;border-radius:999px;padding:6px 10px;font-size:12px;backdrop-filter:blur(4px)';

    var btn = document.createElement('a');
    var isEn = location.pathname.startsWith('/en/');
    btn.textContent = isEn ? '中文' : 'English';
    btn.href = '#';
    btn.style.cssText = 'color:#fff;text-decoration:none;';
    btn.onclick = async function (e) {
      e.preventDefault();
      var targetPath = togglePath(location.pathname);
      var target = location.origin + targetPath + location.search + location.hash;
      if (await exists(target)) {
        location.href = target;
      } else {
        alert('该页面暂无对应语言版本');
      }
    };

    wrap.appendChild(btn);
    document.body.appendChild(wrap);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', buildBtn);
  } else {
    buildBtn();
  }
})();
