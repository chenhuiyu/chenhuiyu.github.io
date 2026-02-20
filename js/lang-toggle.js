(function () {
  const wrap = document.createElement('div');
  wrap.style.cssText = 'position:fixed;right:18px;bottom:92px;z-index:99999;background:rgba(0,0,0,.62);color:#fff;border-radius:999px;padding:6px 10px;font-size:12px;backdrop-filter:blur(4px)';

  const zh = document.createElement('a');
  zh.href = window.location.href.replace(/https:\/\/translate\.google\.[^/]+\/translate\?[^#]+&amp;u=([^&amp;]+).*/, function(_, u){return decodeURIComponent(u)});
  zh.textContent = '中文';
  zh.style.cssText = 'color:#fff;text-decoration:none;margin-right:8px;';

  const sep = document.createElement('span');
  sep.textContent = '|';
  sep.style.opacity = '.65';

  const en = document.createElement('a');
  en.href = 'https://translate.google.com/translate?sl=zh-CN&amp;tl=en&amp;u=' + encodeURIComponent(window.location.href);
  en.target = '_blank';
  en.rel = 'noopener noreferrer';
  en.textContent = 'English';
  en.style.cssText = 'color:#fff;text-decoration:none;margin-left:8px;';

  wrap.appendChild(zh); wrap.appendChild(sep); wrap.appendChild(en);
  document.addEventListener('DOMContentLoaded', function(){ document.body.appendChild(wrap); });
})();
