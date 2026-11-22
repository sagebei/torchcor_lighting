/* TorchCor Lighting website: mobile menu, platform detection and download links. */
(() => {
  'use strict';

  // Every download is the newest app release on GitHub. Its files have the same names in
  // every release, so /releases/latest/download/<name> downloads the newest one directly
  // (the build workflow keeps "latest" on the newest app release); the GitHub API adds the
  // version, date and sizes when it answers.
  const REPO = 'sagebei/torchcor_lighting';
  const GH = `https://github.com/${REPO}`;
  // `alt`: what the download page offers next to the file when it is the visitor's
  const FILES = {
    'macos-arm64.dmg': { os: 'mac', label: 'macOS', detail: 'Apple silicon, macOS 14 or later', alt: 'Intel Macs' },
    'macos-x86_64.dmg': { os: 'mac', label: 'macOS', detail: 'Intel, macOS 14 or later', alt: 'Apple silicon' },
    'windows-x64-setup.exe': { os: 'windows', label: 'Windows', detail: 'Installer for Windows 10 or 11, 64-bit', alt: 'the portable zip' },
    'windows-x64.zip': { os: 'windows', label: 'Windows', detail: 'Portable, Windows 10 or 11, 64-bit' },
    'linux-x86_64.deb': { os: 'linux', label: 'Linux', detail: '.deb for Ubuntu 22.04 and Debian 12 or newer', alt: 'the .tar.gz for other distributions' },
    'linux-x86_64.tar.gz': { os: 'linux', label: 'Linux', detail: 'x86_64 with glibc 2.35 or newer, any distribution', alt: 'the .deb for Ubuntu and Debian' },
  };

  const $ = (s, root = document) => root.querySelector(s);
  const $$ = (s, root = document) => Array.from(root.querySelectorAll(s));

  /* ---------- Mobile menu ---------- */
  const nav = $('.nav');
  const toggle = $('.nav__toggle');
  if (nav && toggle) {
    const setOpen = (open) => {
      nav.classList.toggle('is-open', open);
      toggle.setAttribute('aria-expanded', String(open));
    };
    toggle.addEventListener('click', () => setOpen(!nav.classList.contains('is-open')));
    $$('.nav__links a').forEach((a) => a.addEventListener('click', () => setOpen(false)));
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape') setOpen(false); });
    document.addEventListener('click', (e) => { if (!nav.contains(e.target)) setOpen(false); });
  }

  /* ---------- Platform ---------- */
  function detectOS() {
    const ua = navigator.userAgent || '';
    const platform = (navigator.userAgentData && navigator.userAgentData.platform) || navigator.platform || '';
    if (/iPhone|iPad|iPod|Android/i.test(ua) || (/Mac/i.test(platform) && navigator.maxTouchPoints > 1)) return 'mobile';
    if (/Win/i.test(platform)) return 'windows';
    if (/Mac/i.test(platform)) return 'mac';
    if (/Linux|X11/i.test(platform) && !/CrOS/i.test(ua)) return 'linux';
    return 'other';
  }
  // The visitor's file: offered first on both pages and downloaded straight from the
  // overview's hero button. None for phones, ChromeOS or ARM Linux (no build there yet).
  function detectFile(system) {
    const ua = navigator.userAgent || '';
    if (system === 'mac') return 'macos-arm64.dmg'; // Intel Macs: see the check below
    if (system === 'windows') return 'windows-x64-setup.exe'; // ARM PCs run it emulated
    if (system !== 'linux' || /aarch64|armv\d/i.test(`${navigator.platform || ''} ${ua}`)) return null;
    // Firefox names the distribution; only Debian's family installs a .deb
    return /\b(Fedora|Red Hat|CentOS|SUSE|Arch Linux|Manjaro|Gentoo)\b/i.test(ua) ? 'linux-x86_64.tar.gz' : 'linux-x86_64.deb';
  }
  const os = detectOS();
  let primary = detectFile(os);

  // Download page: make the visitor's file the filled button, first in its card.
  function promote(key) {
    const btn = $(`.platform [data-asset="${key}"]`);
    if (!btn) return;
    $$('[data-asset]', btn.parentNode).forEach((b) => b.classList.toggle('btn--ghost', b !== btn));
    btn.parentNode.prepend(btn);
  }

  /* ---------- Release ---------- */
  const release = { version: '', date: '', page: `${GH}/releases/latest`, assets: {} };

  function link(key, checksum) {
    const a = release.assets[key];
    if (a) return checksum ? a.sha || release.page : a.url;
    return `${GH}/releases/latest/download/TorchCor-Lighting-${key}${checksum ? '.sha256' : ''}`;
  }

  function render() {
    $$('[data-version-label]').forEach((el) => {
      el.textContent = release.version ? `Version ${release.version}` : 'Latest version';
    });
    $$('[data-release-date]').forEach((el) => { el.textContent = release.date ? ` · ${release.date}` : ''; });
    $$('[data-asset]').forEach((a) => { a.href = link(a.dataset.asset); });
    $$('[data-checksum]').forEach((a) => { a.href = link(a.dataset.checksum, true); });
    $$('[data-size]').forEach((el) => {
      const a = release.assets[el.dataset.size];
      el.textContent = a && a.size ? `${Math.round(a.size / 1e6)} MB` : '';
    });
    $$('[data-file]').forEach((el) => {  // a file's name in a command, once it is known
      const a = release.assets[el.dataset.file];
      if (a) el.textContent = a.name;
    });

    // Overview page: the hero button downloads the visitor's file; the line under it names
    // the file and links to every other download. Phones and others keep the download page.
    const hero = $('[data-hero-download]');
    const heroSub = $('[data-hero-sub]');
    if (hero && primary) {
      const f = FILES[primary];
      const a = release.assets[primary];
      hero.href = link(primary);
      $('[data-hero-label]', hero).textContent = `Download for ${f.label}`;
      if (heroSub) {
        heroSub.textContent = `${f.detail}${a && a.size ? ` · ${Math.round(a.size / 1e6)} MB` : ''} · `;
        const all = document.createElement('a');
        all.href = 'download.html';
        all.textContent = 'All downloads';
        heroSub.append(all);
      }
    }

    // Download page: highlight the visitor's platform and offer its file first.
    $$('.platform').forEach((card) => card.classList.toggle('is-recommended', card.dataset.os === os));
    const box = $('[data-primary]');
    if (box && primary) {
      const f = FILES[primary];
      const btn = $('[data-primary-link]', box);
      btn.href = link(primary);
      btn.dataset.asset = primary;
      $('[data-primary-label]', box).textContent = `Download for ${f.label}`;
      $('[data-primary-sub]', box).textContent = `${f.detail} · Other platforms and ${f.alt} below`;
      box.hidden = false;
    }
  }

  if (primary) promote(primary);

  // Chromium browsers can tell Intel Macs apart; everyone else gets Apple silicon first.
  if (os === 'mac' && navigator.userAgentData && navigator.userAgentData.getHighEntropyValues) {
    navigator.userAgentData.getHighEntropyValues(['architecture']).then((v) => {
      if (v.architecture !== 'x86') return;
      primary = 'macos-x86_64.dmg';
      promote(primary);
      render();
    }).catch(() => {});
  }

  function readCache(key) {
    try {
      const c = JSON.parse(sessionStorage.getItem(key) || 'null');
      return c && Date.now() - c.t < 10 * 60 * 1000 ? c : null;  // a new release shows within 10 min
    } catch (e) { return null; }
  }

  async function loadLatestRelease() {
    const KEY = 'tcl-latest-app-release';
    let data = readCache(KEY);
    if (!data) {
      let res;
      try {
        // the newest app release, tagged v1.2.3: the repository's "latest" may be a demo data
        // release (data-v1.0), which holds no app
        res = await fetch(`https://api.github.com/repos/${REPO}/releases?per_page=30`, {
          headers: { Accept: 'application/vnd.github+json' },
        });
      } catch (e) { return; } // offline: the direct links to the latest files stay
      if (!res.ok) return; // private, or rate-limited (60 requests an hour): the same
      const list = await res.json();
      const j = (Array.isArray(list) ? list : []).find(
        (r) => !r.draft && !r.prerelease && /^v\d/i.test(r.tag_name || ''));
      if (!j) return; // no app release yet
      data = {
        t: Date.now(),
        tag: j.tag_name,
        date: j.published_at,
        page: j.html_url,
        assets: (j.assets || []).map((a) => [a.name, a.browser_download_url, a.size]),
      };
      try { sessionStorage.setItem(KEY, JSON.stringify(data)); } catch (e) { /* storage off */ }
    }
    const version = String(data.tag || '').replace(/^v/i, '');
    if (!/^\d/.test(version)) return;
    release.version = version;
    release.page = data.page || release.page;
    release.date = data.date
      ? new Date(data.date).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' })
      : '';
    release.assets = {};
    Object.keys(FILES).forEach((key) => {
      const file = data.assets.find(([name]) => name.endsWith(`-${key}`));
      const sum = data.assets.find(([name]) => name.endsWith(`-${key}.sha256`));
      if (file) release.assets[key] = { name: file[0], url: file[1], size: file[2], sha: sum ? sum[1] : null };
    });
    render();
  }

  render();
  loadLatestRelease();

  /* ---------- Copy buttons ---------- */
  $$('[data-copy]').forEach((btn) => {
    btn.addEventListener('click', async () => {
      try { await navigator.clipboard.writeText(btn.dataset.copy); } catch (e) { return; }
      const old = btn.textContent;
      btn.textContent = 'Copied';
      btn.classList.add('is-done');
      setTimeout(() => { btn.textContent = old; btn.classList.remove('is-done'); }, 1600);
    });
  });

  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* ---------- Pressure-volume loops that follow the electromechanics video ---------- */
  const pvData = $('#pv-data');
  const chart = $('#pv-chart');
  if (pvData && chart) {
    const data = JSON.parse(pvData.textContent);
    const X = (v) => 38 + ((v - 60) / 130) * 212;   // volume 60-190 mL
    const Y = (p) => 196 - (p / 120) * 182;          // pressure 0-120 mmHg
    // Labels clear of the curves: LV in the empty middle of its loop, RV just right of its loop.
    const label = (sel, x, y, anchor) => {
      const el = $(sel);
      el.setAttribute('x', x.toFixed(1));
      el.setAttribute('y', y.toFixed(1));
      el.setAttribute('text-anchor', anchor);
    };
    const mid = (a) => (Math.min(...a) + Math.max(...a)) / 2;
    label('#pv-name-lv', X(mid(data.lv.V)), Y(Math.max(...data.lv.p) * 0.55) + 4, 'middle');
    label('#pv-name-rv', X(Math.max(...data.rv.V)) + 6, Y(mid(data.rv.p)) + 4, 'start');
    // Each loop is traced frame by frame as the heart beats; the white dot leads the line.
    const f = data.frames;
    const n = f.lvV.length;
    const trace = (V, p, upto) => {
      let d = '';
      for (let k = 0; k <= upto; k += 1) d += `${k ? ' L' : 'M'}${X(V[k]).toFixed(1)} ${Y(p[k]).toFixed(1)}`;
      return d;
    };
    const lv = { path: $('#pv-lv'), prev: $('#pv-lv-prev'), dot: $('#pv-dot-lv'), V: f.lvV, p: f.lvp };
    const rv = { path: $('#pv-rv'), prev: $('#pv-rv-prev'), dot: $('#pv-dot-rv'), V: f.rvV, p: f.rvp };
    let shown = -1;
    const place = (i) => {
      if (i === shown) return;
      const newBeat = i < shown;
      shown = i;
      [lv, rv].forEach((c) => {
        if (newBeat) { // let the loop just completed fade out behind the new one
          c.prev.setAttribute('d', `${trace(c.V, c.p, n - 1)}Z`);
          c.prev.classList.remove('is-fading');
          void c.prev.getBoundingClientRect();
          c.prev.classList.add('is-fading');
        }
        c.path.setAttribute('d', trace(c.V, c.p, i));
        c.dot.setAttribute('cx', X(c.V[i]).toFixed(1));
        c.dot.setAttribute('cy', Y(c.p[i]).toFixed(1));
      });
    };
    if (reduceMotion) {
      // still page: the complete loops, with the dots at the poster frame (mid-systole)
      [lv, rv].forEach((c) => c.path.setAttribute('d', `${trace(c.V, c.p, n - 1)}Z`));
      const i = Math.round(n * 0.25);
      [lv, rv].forEach((c) => {
        c.dot.setAttribute('cx', X(c.V[i]).toFixed(1));
        c.dot.setAttribute('cy', Y(c.p[i]).toFixed(1));
      });
    } else {
      place(0);
    }
    const video = $('#em-video');
    if (video && !reduceMotion) {
      let raf = 0;
      const tick = () => {
        if (video.duration) place(Math.min(n - 1, Math.floor((video.currentTime / video.duration) * n)));
        raf = requestAnimationFrame(tick);
      };
      const start = () => { if (!raf) raf = requestAnimationFrame(tick); };
      const stop = () => { cancelAnimationFrame(raf); raf = 0; };
      if ('IntersectionObserver' in window) {
        new IntersectionObserver(([entry]) => (entry.isIntersecting ? start() : stop())).observe(chart);
      } else {
        start();
      }
    }
  }

  /* ---------- Respect reduced motion: stills instead of loops ---------- */
  if (reduceMotion) {
    $$('svg').forEach((svg) => {
      if (!svg.pauseAnimations) return;
      svg.pauseAnimations();
      const still = parseFloat(svg.dataset.still || '');
      if (!Number.isNaN(still) && svg.setCurrentTime) svg.setCurrentTime(still);
    });
    $$('video[autoplay]').forEach((video) => { video.removeAttribute('autoplay'); video.pause(); });
  }
})();
