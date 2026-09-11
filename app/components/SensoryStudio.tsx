'use client';

import { useEffect, useRef, useState } from 'react';

type OrientationAPI = typeof DeviceOrientationEvent & { requestPermission?: () => Promise<string> };
const palettes = [
  { name: 'Aurora · 极光', hue: 180 },
  { name: 'Ember · 余烬', hue: 20 },
  { name: 'Iris · 鸢尾', hue: 265 },
];
const clamp = (n: number, min: number, max: number) => Math.min(max, Math.max(min, n));

export function SensoryStudio() {
  const canvas = useRef<HTMLCanvasElement>(null);
  const video = useRef<HTMLVideoElement>(null);
  const stage = useRef<HTMLDivElement>(null);
  const stream = useRef<MediaStream | null>(null);
  const cameraEpoch = useRef(0);
  const orientationEpoch = useRef(0);
  const orientationOff = useRef<(() => void) | null>(null);
  const orientationTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const target = useRef({ x: 0, y: 0 });
  const engine = useRef({ paused: false, palette: 0, camera: false, visible: true, dirty: true });
  const [paused, setPaused] = useState(false);
  const [palette, setPalette] = useState(0);
  const [camera, setCamera] = useState(false);
  const [requesting, setRequesting] = useState(false);
  const [tilt, setTilt] = useState(false);
  const [message, setMessage] = useState('Move your pointer. Tilt your world. / 移动鼠标，拨动一片光。');

  function stopCamera() {
    cameraEpoch.current++;
    stream.current?.getTracks().forEach(track => track.stop());
    stream.current = null;
    if (video.current) video.current.srcObject = null;
    engine.current.camera = false;
    engine.current.dirty = true;
    setCamera(false);
    setRequesting(false);
  }
  async function toggleCamera() {
    if (camera || requesting) { stopCamera(); setMessage('Camera off. / 摄像头已关闭。'); return; }
    if (!navigator.mediaDevices?.getUserMedia) { setMessage('Camera unavailable here. Try a modern HTTPS browser. / 当前浏览器不支持摄像头，可继续玩光场。'); return; }
    const ticket = ++cameraEpoch.current;
    setRequesting(true);
    setMessage('Allow your camera to become a living pointillist portrait. / 允许摄像头，把自己变成流动的点彩肖像。');
    try {
      const media = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } }, audio: false });
      if (ticket !== cameraEpoch.current) { media.getTracks().forEach(track => track.stop()); return; }
      if (document.hidden || !engine.current.visible) { media.getTracks().forEach(track => track.stop()); stopCamera(); setMessage('Camera cancelled because the artwork is not visible. / 作品不在视野内，已取消摄像头。'); return; }
      stream.current = media;
      media.getVideoTracks().forEach(track => track.addEventListener('ended', () => { if (stream.current === media) { stopCamera(); setMessage('Camera disconnected. / 摄像头连接已结束。'); } }, { once: true }));
      if (!video.current) { stopCamera(); return; }
      video.current.srcObject = media;
      await video.current.play();
      if (ticket !== cameraEpoch.current) return;
      engine.current.camera = true;
      engine.current.dirty = true;
      setCamera(true);
      setRequesting(false);
      setMessage('You, made of light. No recording, no upload. / 此刻的你，由光组成。不录制、不上传。');
    } catch {
      if (ticket !== cameraEpoch.current) return;
      stopCamera();
      setMessage('Camera not available or permission declined. The light field still works. / 摄像头不可用或未获授权，仍可探索光场。');
    }
  }
  async function toggleTilt() {
    const ticket = ++orientationEpoch.current;
    if (tilt) {
      orientationOff.current?.(); orientationOff.current = null;
      if (orientationTimer.current) clearTimeout(orientationTimer.current);
      setTilt(false); target.current = { x: 0, y: 0 }; engine.current.dirty = true;
      setMessage('Tilt off. / 倾斜感应已关闭。'); return;
    }
    if (typeof DeviceOrientationEvent === 'undefined') { setMessage('No motion sensor available. Use the pointer or arrow keys. / 未找到方向传感器，可使用鼠标或方向键。'); return; }
    try {
      const api = DeviceOrientationEvent as OrientationAPI;
      if (api.requestPermission && await api.requestPermission() !== 'granted') { setMessage('Motion permission declined. / 未获动作感应授权，可继续用鼠标或触摸。'); return; }
      if (ticket !== orientationEpoch.current) return;
      let baseline: { beta: number; gamma: number } | null = null;
      const listener = (e: DeviceOrientationEvent) => {
        if (engine.current.paused) return;
        if (e.beta === null || e.gamma === null) return;
        if (!baseline) { baseline = { beta: e.beta, gamma: e.gamma }; if (orientationTimer.current) clearTimeout(orientationTimer.current); setMessage('Gently tilt your phone. / 轻轻倾斜手机，让光跟随你。'); }
        const x = clamp((e.gamma - baseline.gamma) / 30, -1, 1);
        const y = clamp((e.beta - baseline.beta) / 30, -1, 1);
        const angle = (screen.orientation?.angle ?? 0) * Math.PI / 180;
        target.current = { x: x * Math.cos(angle) + y * Math.sin(angle), y: y * Math.cos(angle) - x * Math.sin(angle) };
        engine.current.dirty = true;
      };
      orientationOff.current?.();
      window.addEventListener('deviceorientation', listener);
      orientationOff.current = () => window.removeEventListener('deviceorientation', listener);
      setTilt(true);
      setMessage('Waiting for movement… / 等待方向数据…');
      orientationTimer.current = setTimeout(() => { if (!baseline) { orientationOff.current?.(); orientationOff.current = null; setTilt(false); setMessage('No sensor data received. Use the pointer or arrow keys. / 未收到传感器数据，请用鼠标或方向键。'); } }, 5000);
    } catch { setMessage('Motion unavailable in this browser. / 此浏览器暂时无法读取方向数据。'); }
  }

  useEffect(() => {
    const surface = canvas.current;
    const container = stage.current;
    if (!surface || !container) return;
    const ctx = surface.getContext('2d');
    if (!ctx) return;
    const sample = document.createElement('canvas');
    sample.width = 96; sample.height = 64;
    const sampleCtx = sample.getContext('2d', { willReadFrequently: true });
    const reduced = matchMedia('(prefers-reduced-motion: reduce)');
    let width = 1, height = 1, frame = 0, last = 0, time = 0;
    let currentX = 0, currentY = 0;
    const resize = () => {
      const rect = container.getBoundingClientRect();
      width = Math.max(1, rect.width); height = Math.max(1, rect.height);
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      surface.width = Math.round(width * ratio); surface.height = Math.round(height * ratio);
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      engine.current.dirty = true;
    };
    const observer = new ResizeObserver(resize); observer.observe(container); resize();
    const visibility = new IntersectionObserver(entries => { engine.current.visible = entries[0].isIntersecting; engine.current.dirty = true; if (!entries[0].isIntersecting && stream.current) { stopCamera(); setMessage('Camera stopped when the artwork left view. / 作品移出视野，摄像头已关闭。'); } });
    visibility.observe(container);
    const onHidden = () => { if (document.hidden) { stopCamera(); setMessage('Camera stopped in the background. / 页面进入后台，摄像头已关闭。'); } engine.current.dirty = true; };
    document.addEventListener('visibilitychange', onHidden);
    function draw(now: number) {
      frame = requestAnimationFrame(draw);
      if (!ctx || document.hidden || !engine.current.visible || now - last < 40) return;
      const state = engine.current;
      if ((state.paused || reduced.matches) && !state.dirty) return;
      if (!state.paused && !reduced.matches) time += Math.min((now - last) / 1000, 0.05);
      last = now; state.dirty = false;
      currentX += (target.current.x - currentX) * 0.12;
      currentY += (target.current.y - currentY) * 0.12;
      ctx.fillStyle = '#0c1018'; ctx.fillRect(0, 0, width, height);
      const hue = palettes[state.palette].hue;
      const glow = ctx.createRadialGradient(width * .52, height * .5, 0, width * .52, height * .5, width * .55);
      glow.addColorStop(0, `hsla(${hue},60%,24%,.3)`); glow.addColorStop(1, 'transparent');
      ctx.fillStyle = glow; ctx.fillRect(0, 0, width, height);
      if (state.camera && video.current && sampleCtx && video.current.readyState >= 2) {
        const v = video.current;
        const ratio = Math.max(96 / v.videoWidth, 64 / v.videoHeight);
        sampleCtx.drawImage(v, (96 - v.videoWidth * ratio) / 2, (64 - v.videoHeight * ratio) / 2, v.videoWidth * ratio, v.videoHeight * ratio);
        const pixels = sampleCtx.getImageData(0, 0, 96, 64).data;
        const scale = Math.min(width / 104, height / 72);
        for (let y = 0; y < 64; y++) for (let x = 0; x < 96; x++) {
          const offset = (y * 96 + x) * 4;
          const light = (.2126 * pixels[offset] + .7152 * pixels[offset + 1] + .0722 * pixels[offset + 2]) / 255;
          if (light < .045) continue;
          const px = width / 2 + (47.5 - x) * scale + currentX * 10;
          const py = height / 2 + (y - 31.5) * scale + currentY * 10;
          ctx.fillStyle = `hsla(${hue + light * 65},65%,${25 + light * 65}%,${.4 + light * .6})`;
          ctx.beginPath(); ctx.arc(px, py, Math.max(.3, light * scale * .47), 0, Math.PI * 2); ctx.fill();
        }
      } else {
        // Deterministic parametric ribbons: a living mathematical surface, not a bitmap.
        const columns = width < 600 ? 86 : 136;
        for (let row = 0; row < 32; row++) {
          for (let col = 0; col < columns; col++) {
            const u = col / (columns - 1), v = row / 31;
            const wave = Math.sin(u * 8 + time * .35 + v * 3) * .16 + Math.cos(u * 4 - time * .2) * .07;
            const px = width * (.07 + u * .86) + currentX * Math.sin(v * Math.PI) * 42;
            const py = height * (.23 + v * .5 + wave) + currentY * 35 * Math.sin(u * Math.PI) + Math.sin(u * 12 + v * 5 + time * .4) * 12;
            const distance = Math.hypot(px / width - (.5 + target.current.x * .4), py / height - (.5 + target.current.y * .35));
            const intensity = Math.max(0, 1 - distance * 3);
            ctx.fillStyle = `hsla(${hue + u * 65 + v * 15},72%,${45 + intensity * 35}%,${.25 + Math.sin(v * Math.PI) * .55})`;
            ctx.beginPath(); ctx.arc(px, py, .65 + intensity * 1.3, 0, Math.PI * 2); ctx.fill();
          }
        }
      }
      ctx.fillStyle = '#aeb8c9'; ctx.font = '12px monospace';
      ctx.fillText('YVETTE CHEN / SENSORY STUDIES', 22, height - 22);
    }
    frame = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(frame); observer.disconnect(); visibility.disconnect();
      document.removeEventListener('visibilitychange', onHidden);
      orientationEpoch.current++;
      cameraEpoch.current++; stream.current?.getTracks().forEach(track => track.stop());
      orientationOff.current?.(); if (orientationTimer.current) clearTimeout(orientationTimer.current);
    };
  }, []);

  function save() {
    canvas.current?.toBlob(blob => {
      if (!blob) return;
      const url = URL.createObjectURL(blob); const link = document.createElement('a');
      link.href = url; link.download = 'yvette-sensory-study.png'; link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    }, 'image/png');
  }
  return <section className="sensory-studio" id="sensory-studio" aria-labelledby="sensory-title">
    <div className="sensory-heading"><p className="section-kicker">An open-ended experiment · 感官实验室</p><span className="sensory-edition">STUDY 001 / LIGHT & PRESENCE</span></div>
    <div className="sensory-intro"><h2 id="sensory-title">A little less scrolling.<br/><em>A little more wonder.</em></h2><p>让光随你的手指流动，<br/>让你的轮廓成为作品。<br/><span>Move, tilt, or step into the frame.</span></p></div>
    <div className="sensory-stage" ref={stage}>
      <canvas ref={canvas} tabIndex={0} aria-label="Interactive light artwork. Move the pointer, touch the surface, or use arrow keys. 互动光场：鼠标、触摸或方向键控制。"
        onPointerMove={e => { if (tilt || paused) return; const r = e.currentTarget.getBoundingClientRect(); target.current = { x: clamp((e.clientX-r.left)/r.width*2-1,-1,1), y: clamp((e.clientY-r.top)/r.height*2-1,-1,1) }; engine.current.dirty = true; }}
        onPointerLeave={() => { if (!tilt && !paused) { target.current = {x:0,y:0}; engine.current.dirty = true; } }}
        onKeyDown={e => { if (paused) return; const directions:Record<string,[number,number]> = {ArrowLeft:[-.15,0],ArrowRight:[.15,0],ArrowUp:[0,-.15],ArrowDown:[0,.15]}; const d=directions[e.key]; if(d){e.preventDefault();target.current={x:clamp(target.current.x+d[0],-1,1),y:clamp(target.current.y+d[1],-1,1)};engine.current.dirty=true;} }}>
        An interactive artwork of flowing light. / 由流动光点组成的互动作品。
      </canvas>
      <video ref={video} muted playsInline className="sensory-video" aria-hidden="true"/>
      <div className="sensory-caption" aria-hidden="true"><span>{camera?'02 / LIVING PORTRAIT':'01 / FIELD OF LIGHT'}</span><span>{camera?'你，就是作品。':'Touch the intangible.'}</span></div>
    </div>
    <div className="sensory-controls">
      <div className="sensory-tools"><button onClick={toggleCamera} aria-pressed={camera}>{requesting?'Cancel / 取消':camera?'Close camera / 关闭摄像头':'Become the art / 摄像头肖像'}</button><button onClick={toggleTilt} aria-pressed={tilt}>{tilt?'Tilt off / 关闭倾斜':'Enable tilt / 手机倾斜'}</button><button onClick={() => { engine.current.paused=!paused;engine.current.dirty=true;setPaused(!paused); }} aria-pressed={paused}>{paused?'Resume / 继续':'Pause / 暂停'}</button><button onClick={save}>Keep this moment / 保存画面</button></div>
      <label className="sensory-palette">Palette / 配色<select value={palette} onChange={e=>{const p=Number(e.target.value);setPalette(p);engine.current.palette=p;engine.current.dirty=true;}}>{palettes.map((p,i)=><option key={p.name} value={i}>{p.name}</option>)}</select></label>
    </div>
    <p className="sensory-status" role="status" aria-live="polite">{message}</p>
    <p className="sensory-privacy">Camera is optional. Images stay on your device; no microphone, no face recognition. / 摄像头仅在你开启后使用，图像留在本机，不开启麦克风、不识别人脸。<br/>Motion follows your accessibility preferences. Pause freezes the artwork; Close camera releases the camera. / 尊重系统减少动态效果设置；暂停画面不会关闭摄像头，请使用关闭按钮。</p>
  </section>;
}
