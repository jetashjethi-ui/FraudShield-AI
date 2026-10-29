// Animated wireframe "void" for the hero. Three.js r181 is vendored locally
// (assets/vendor/three) so no CDN or build step is needed.

const THREE_URL = new URL('../vendor/three/three.module.min.js', import.meta.url).href;

const NOISE_GLSL = /* glsl */ `
  vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
  vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
  vec4 permute(vec4 x) { return mod289(((x * 34.0) + 1.0) * x); }
  vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }

  float snoise(vec3 v) {
    const vec2 C = vec2(1.0 / 6.0, 1.0 / 3.0);
    const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
    vec3 i = floor(v + dot(v, C.yyy));
    vec3 x0 = v - i + dot(i, C.xxx);
    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min(g.xyz, l.zxy);
    vec3 i2 = max(g.xyz, l.zxy);
    vec3 x1 = x0 - i1 + C.xxx;
    vec3 x2 = x0 - i2 + C.yyy;
    vec3 x3 = x0 - D.yyy;
    i = mod289(i);
    vec4 p = permute(permute(permute(
      i.z + vec4(0.0, i1.z, i2.z, 1.0))
      + i.y + vec4(0.0, i1.y, i2.y, 1.0))
      + i.x + vec4(0.0, i1.x, i2.x, 1.0));
    float n_ = 0.142857142857;
    vec3 ns = n_ * D.wyz - D.xzx;
    vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_);
    vec4 x = x_ * ns.x + ns.yyyy;
    vec4 y = y_ * ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);
    vec4 b0 = vec4(x.xy, y.xy);
    vec4 b1 = vec4(x.zw, y.zw);
    vec4 s0 = floor(b0) * 2.0 + 1.0;
    vec4 s1 = floor(b1) * 2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));
    vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
    vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;
    vec3 p0 = vec3(a0.xy, h.x);
    vec3 p1 = vec3(a0.zw, h.y);
    vec3 p2 = vec3(a1.xy, h.z);
    vec3 p3 = vec3(a1.zw, h.w);
    vec4 norm = taylorInvSqrt(vec4(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
    p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;
    vec4 m = max(0.6 - vec4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), 0.0);
    m = m * m;
    return 42.0 * dot(m * m, vec4(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3)));
  }
`;

const VERTEX = /* glsl */ `
  uniform float uTime;
  uniform float uAmplitude;
  uniform float uFrequency;
  uniform float uSpeed;
  uniform float uHover;
  varying float vDisp;
  varying float vRim;
  ${NOISE_GLSL}
  void main() {
    float t = uTime * uSpeed;
    vec3 p = position * uFrequency;
    float n = snoise(p + vec3(0.0, t * 0.12, 0.0)) * 0.62
            + snoise(p * 2.1 - vec3(t * 0.08)) * 0.28
            + snoise(p * 4.3 + vec3(t * 0.05, 0.0, 0.0)) * 0.10;
    float breath = 1.0 + sin(uTime * 0.55) * 0.14 + uHover * 0.35;
    float disp = n * uAmplitude * breath;
    vDisp = disp;
    vec3 displaced = position + normal * disp;
    vec4 mv = modelViewMatrix * vec4(displaced, 1.0);
    vec3 viewNormal = normalize(normalMatrix * normal);
    vRim = 1.0 - abs(dot(viewNormal, normalize(-mv.xyz)));
    gl_Position = projectionMatrix * mv;
  }
`;

const FRAGMENT = /* glsl */ `
  uniform float uOpacity;
  uniform float uFade;
  uniform float uHover;
  uniform float uAccentMix;
  uniform vec3 uAccent;
  varying float vDisp;
  varying float vRim;
  void main() {
    float b = clamp(0.22 + vDisp * 1.9 + vRim * 0.38, 0.0, 1.0);
    vec3 col = mix(vec3(0.14), vec3(0.82), b);
    float blue = smoothstep(0.6, 1.0, vRim) * smoothstep(0.04, 0.17, vDisp);
    col = mix(col, uAccent, blue * (0.28 + uHover * 0.35) * uAccentMix);
    float alpha = uOpacity * (0.4 + 0.6 * b) * uFade;
    gl_FragColor = vec4(col, alpha);
  }
`;

const hero = document.getElementById('hero');
const mount = document.getElementById('hero-canvas');
const initButton = document.getElementById('initialize');
const motionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');

function useFallback(reason) {
  if (!hero) return;
  hero.setAttribute('data-hero-state', 'fallback');
  hero.setAttribute('data-fallback-reason', reason);
}

function hasWebGL() {
  try {
    const canvas = document.createElement('canvas');
    return Boolean(canvas.getContext('webgl2') || canvas.getContext('webgl'));
  } catch {
    return false;
  }
}

async function start() {
  const THREE = await import(THREE_URL);
  const compact = window.matchMedia('(max-width: 768px), (pointer: coarse)').matches;

  const renderer = new THREE.WebGLRenderer({
    antialias: !compact,
    alpha: true,
    powerPreference: 'high-performance',
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, compact ? 1.5 : 2));
  renderer.setClearColor(0x000000, 0);
  renderer.domElement.setAttribute('aria-hidden', 'true');
  mount.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(34, 1, 0.1, 60);
  const group = new THREE.Group();
  scene.add(group);

  const makeMaterial = ({ opacity, amplitude, frequency, speed, accent }) =>
    new THREE.ShaderMaterial({
      uniforms: {
        uTime: { value: 0 },
        uAmplitude: { value: amplitude },
        uFrequency: { value: frequency },
        uSpeed: { value: speed },
        uOpacity: { value: opacity },
        uHover: { value: 0 },
        uFade: { value: 1 },
        uAccentMix: { value: accent },
        uAccent: { value: new THREE.Color(0x2f62ff) },
      },
      vertexShader: VERTEX,
      fragmentShader: FRAGMENT,
      wireframe: true,
      transparent: true,
      depthWrite: false,
    });

  const core = new THREE.Mesh(
    new THREE.IcosahedronGeometry(1, compact ? 26 : 56),
    makeMaterial({ opacity: 0.62, amplitude: 0.15, frequency: 1.25, speed: 1, accent: 1 }),
  );
  const shell = new THREE.Mesh(
    new THREE.IcosahedronGeometry(1.17, compact ? 8 : 16),
    makeMaterial({ opacity: 0.13, amplitude: 0.08, frequency: 0.8, speed: 0.6, accent: 0.4 }),
  );
  // A dark inner body hides the far side of the wireframe so the void reads with depth.
  const occluder = new THREE.Mesh(
    new THREE.IcosahedronGeometry(0.82, 6),
    new THREE.MeshBasicMaterial({ color: 0x060606 }),
  );
  occluder.renderOrder = 0;
  core.renderOrder = 1;
  shell.renderOrder = 2;
  group.add(occluder, core, shell);

  const materials = [core.material, shell.material];
  const state = {
    time: 0,
    rotY: 0,
    pointer: { x: 0, y: 0 },
    pointerTarget: { x: 0, y: 0 },
    hover: 0,
    hoverPointer: 0,
    hoverControl: 0,
    push: 0,
    pushTarget: 0,
    initializedAt: 0,
    baseZ: 5.6,
    inView: true,
    running: false,
    last: 0,
    raf: 0,
    reduced: motionQuery.matches,
  };

  function resize() {
    const width = mount.clientWidth;
    const height = mount.clientHeight;
    if (!width || !height) return;
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    const tanHalf = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
    const diameter = 2.5;
    const fitHeight = diameter / 0.86 / (2 * tanHalf);
    const fitWidth = diameter / 0.94 / (2 * tanHalf * camera.aspect);
    state.baseZ = Math.max(fitHeight, fitWidth);
    camera.updateProjectionMatrix();
    if (!state.running) renderFrame(0);
  }

  function renderFrame(dt) {
    const ease = (rate) => 1 - Math.exp(-dt * rate);
    if (state.reduced) {
      state.pointer.x = state.pointer.y = 0;
      state.hover = 0;
      state.push = state.pushTarget;
    } else {
      state.time += dt;
      state.rotY += dt * 0.05;
      state.pointer.x += (state.pointerTarget.x - state.pointer.x) * ease(2.4);
      state.pointer.y += (state.pointerTarget.y - state.pointer.y) * ease(2.4);
      const hoverTarget = Math.max(state.hoverPointer, state.hoverControl);
      state.hover += (hoverTarget - state.hover) * ease(3);
      state.push += (state.pushTarget - state.push) * ease(3.4);
    }

    for (const material of materials) {
      material.uniforms.uTime.value = state.time;
      material.uniforms.uHover.value = state.hover;
      material.uniforms.uFade.value = 1 - state.push * 0.55;
    }

    group.rotation.y = state.rotY + state.pointer.x * 0.28;
    group.rotation.x = 0.18 + Math.sin(state.time * 0.15) * 0.05 - state.pointer.y * 0.2;
    shell.rotation.y = -state.time * 0.03;
    shell.rotation.z = state.time * 0.012;
    group.scale.setScalar(1 + state.push * 0.42 + state.hover * 0.03);

    camera.position.set(state.pointer.x * 0.18, state.pointer.y * 0.12, state.baseZ * (1 - state.push * 0.38));
    camera.lookAt(0, 0, 0);
    renderer.render(scene, camera);
  }

  function tick(now) {
    if (!state.running) return;
    const dt = state.last ? Math.min((now - state.last) / 1000, 0.05) : 0.016;
    state.last = now;
    renderFrame(dt);
    state.raf = requestAnimationFrame(tick);
  }

  function updateLoop() {
    const shouldRun = state.inView && !document.hidden && !state.reduced;
    if (shouldRun && !state.running) {
      state.running = true;
      state.last = 0;
      state.raf = requestAnimationFrame(tick);
    } else if (!shouldRun && state.running) {
      state.running = false;
      cancelAnimationFrame(state.raf);
    }
  }

  window.addEventListener(
    'pointermove',
    (event) => {
      if (state.reduced) return;
      state.pointerTarget.x = (event.clientX / window.innerWidth) * 2 - 1;
      state.pointerTarget.y = -((event.clientY / window.innerHeight) * 2 - 1);
      const rect = hero.getBoundingClientRect();
      const dx = event.clientX - (rect.left + rect.width / 2);
      const dy = event.clientY - (rect.top + rect.height / 2);
      const radius = Math.min(rect.width, rect.height) * 0.42;
      state.hoverPointer = Math.hypot(dx, dy) < radius && rect.bottom > 0 ? 1 : 0;
    },
    { passive: true },
  );

  if (initButton) {
    const on = () => (state.hoverControl = 1);
    const off = () => (state.hoverControl = 0);
    initButton.addEventListener('pointerenter', on);
    initButton.addEventListener('pointerleave', off);
    initButton.addEventListener('focus', on);
    initButton.addEventListener('blur', off);
  }

  window.addEventListener('fs:initialize', () => {
    state.pushTarget = 1;
    state.initializedAt = performance.now();
    if (state.reduced) renderFrame(0);
  });

  window.addEventListener('fs:hero-return', () => {
    state.pushTarget = 0;
    if (state.reduced) renderFrame(0);
  });

  const observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        state.inView = entry.isIntersecting;
        const settled = performance.now() - state.initializedAt > 1400;
        if (entry.intersectionRatio >= 0.6 && state.pushTarget === 1 && settled) {
          state.pushTarget = 0;
        }
      }
      updateLoop();
    },
    { threshold: [0, 0.6] },
  );
  observer.observe(hero);

  document.addEventListener('visibilitychange', updateLoop);

  motionQuery.addEventListener('change', (event) => {
    state.reduced = event.matches;
    updateLoop();
    renderFrame(0);
  });

  renderer.domElement.addEventListener('webglcontextlost', (event) => {
    event.preventDefault();
    state.running = false;
    cancelAnimationFrame(state.raf);
    useFallback('context-lost');
  });

  new ResizeObserver(resize).observe(mount);
  resize();
  renderFrame(0);
  hero.setAttribute('data-hero-state', 'ready');
  updateLoop();
}

if (hero && mount) {
  if (!hasWebGL()) {
    useFallback('webgl-unavailable');
  } else {
    start().catch((error) => {
      console.warn('[FraudShield] 3D hero unavailable, showing static fallback.', error);
      useFallback('three-load-failed');
    });
  }
}
