/* ==========================================================================
   Ayris — общий движок нодового холста для вариантов редизайна.
   Взят 1:1 из design/mockups/node_editor_mockup.html и обобщён:
     • все кнопки тулбара необязательны — привязка только если есть в DOM;
     • слева кликабельный каталог блоков (#catalog) вместо списка команд;
     • добавлены «Показать всё» (#fitBtn) и «1:1» (#resetBtn);
     • стиль провода настраивается через window.AYRIS_WIRE (толщины/прозрачности);
     • тумблер темы убран — у каждого варианта своя подача.
   Логика холста (панорама, зум к курсору, порты, провода, инспектор) не тронута.
   ========================================================================== */
const ICONS = {
  trigger: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5 11a7 7 0 0 0 14 0M12 18v3"/></svg>',
  response: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 5h16v11H8l-4 4z"/></svg>',
  action: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M13 2 4 14h7l-1 8 9-12h-7z"/></svg>',
  condition: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"><path d="M12 3 21 12 12 21 3 12z"/></svg>',
  sound: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 9v6h4l5 4V5L8 9zM17 8a5 5 0 0 1 0 8"/></svg>',
};
/* Иконки категорий каталога — по одной на группу BlockCatalog (понятнее, чем 5 ролей) */
const _svg = p => `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${p}</svg>`;
const CAT_ICONS = {
  'Триггеры': _svg('<path d="M13 2 4 14h7l-1 8 9-12h-7z"/>'),
  'Голос / Звук': _svg('<path d="M4 9v6h4l5 4V5L8 9z"/><path d="M17 8a5 5 0 0 1 0 8"/>'),
  'Ввод': _svg('<rect x="2.5" y="6" width="19" height="12" rx="2"/><path d="M6 10h.01M9.5 10h.01M13 10h.01M16.5 10h.01M7 14h10"/>'),
  'Система': _svg('<circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3M4.9 4.9 7 7M17 17l2.1 2.1M19.1 4.9 17 7M7 17l-2.1 2.1"/>'),
  'Логика': _svg('<circle cx="6" cy="6" r="2.2"/><circle cx="6" cy="18" r="2.2"/><circle cx="18" cy="12" r="2.2"/><path d="M6 8.2v7.6M8.2 6H14a2 2 0 0 1 2 2v2M8.2 18H14a2 2 0 0 0 2-2v-2"/>'),
  'Поток': _svg('<path d="M4 12h13M13 6l6 6-6 6"/>'),
  'Сеть / Веб': _svg('<circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c3 3 3 15 0 18M12 3c-3 3-3 15 0 18"/>'),
  'Уведомления': _svg('<path d="M18 8a6 6 0 1 0-12 0c0 7-3 9-3 9h18s-3-2-3-9"/><path d="M13.7 21a2 2 0 0 1-3.4 0"/>'),
  'Яндекс Музыка': _svg('<path d="M9 18V6l10-2v10"/><circle cx="6" cy="18" r="2.6"/><circle cx="19" cy="16" r="2.6"/>'),
};
const catIcon = (cat, role) => CAT_ICONS[cat] || ICONS[role] || '';
const CLOCK ='<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>';
const ROLES = {
  trigger:   { color:'var(--r-trigger)',   label:'триггер' },
  response:  { color:'var(--r-response)',   label:'ответ' },
  action:    { color:'var(--r-action)',     label:'действие' },
  condition: { color:'var(--r-condition)',  label:'условие' },
  sound:     { color:'var(--r-sound)',      label:'звук' },
};
function outsOf(n){ return n.role==='condition' ? ['then','else'] : ['out']; }
const OUT_META = { out:{label:'',color:null}, then:{label:'то',color:'var(--success)'}, else:{label:'иначе',color:'var(--error)'} };

/* Каталог = реальный BlockCatalog приложения (8 категорий).
   Цвет ноды берём по «роли» холста; категория задаёт роль для тона. */
const BLOCKS = [
  { cat:'Триггеры', items:[
    { role:'trigger', name:'Голосовая фраза', desc:'запуск по слову или фразе' },
    { role:'trigger', name:'Горячая клавиша', desc:'сочетание клавиш' },
    { role:'trigger', name:'Расписание', desc:'по времени или cron' },
  ]},
  { cat:'Голос / Звук', items:[
    { role:'sound', name:'Сказать голосом', desc:'произнести фразу (TTS)' },
    { role:'sound', name:'Проиграть звук', desc:'файл или библиотека' },
    { role:'sound', name:'Остановить звук', desc:'оборвать проигрывание' },
    { role:'sound', name:'Громкость', desc:'уровень системного звука' },
    { role:'sound', name:'Голос TTS', desc:'выбрать голос синтеза' },
  ]},
  { cat:'Ввод', items:[
    { role:'action', name:'Нажать клавишу', desc:'комбинация клавиш' },
    { role:'action', name:'Клавиша вниз', desc:'зажать клавишу' },
    { role:'action', name:'Клавиша вверх', desc:'отпустить клавишу' },
    { role:'action', name:'Ввести текст', desc:'эмуляция набора' },
    { role:'action', name:'Клик мышью', desc:'нажатие кнопки мыши' },
    { role:'action', name:'Двинуть мышь', desc:'переместить курсор' },
    { role:'action', name:'Перетащить мышью', desc:'drag от точки к точке' },
    { role:'action', name:'Колесо мыши', desc:'прокрутка' },
  ]},
  { cat:'Система', items:[
    { role:'response', name:'Запустить программу', desc:'exe или команда' },
    { role:'response', name:'Команда оболочки', desc:'shell-команда', danger:true },
    { role:'response', name:'Закрыть программу', desc:'завершить приложение' },
    { role:'response', name:'Фокус окна', desc:'вывести окно вперёд' },
    { role:'response', name:'Состояние окна', desc:'свернуть или развернуть' },
    { role:'response', name:'Яркость', desc:'яркость экрана' },
    { role:'response', name:'Wi-Fi', desc:'включить или выключить' },
    { role:'response', name:'Bluetooth', desc:'включить или выключить' },
    { role:'response', name:'Питание ПК', desc:'сон или выключение', danger:true },
    { role:'response', name:'Скриншот', desc:'снимок экрана' },
    { role:'response', name:'Записать в буфер', desc:'буфер обмена' },
    { role:'response', name:'Прочитать буфер', desc:'из буфера обмена' },
  ]},
  { cat:'Логика', items:[
    { role:'condition', name:'Если', desc:'ветвление по условию' },
    { role:'condition', name:'Переключатель', desc:'несколько веток' },
    { role:'condition', name:'Пока', desc:'цикл по условию' },
    { role:'condition', name:'Цикл', desc:'перебор элементов' },
    { role:'condition', name:'Попытка', desc:'перехват ошибок' },
    { role:'condition', name:'Задать переменную', desc:'запись значения' },
    { role:'condition', name:'Прочитать переменную', desc:'чтение значения' },
    { role:'condition', name:'Добавить в список', desc:'push в массив' },
    { role:'condition', name:'Элемент списка', desc:'по индексу' },
    { role:'condition', name:'Задать в словарь', desc:'ключ → значение' },
    { role:'condition', name:'Прочитать словарь', desc:'по ключу' },
  ]},
  { cat:'Поток', items:[
    { role:'trigger', name:'Ожидание', desc:'пауза или условие' },
    { role:'trigger', name:'Вызвать команду', desc:'запустить другую команду' },
    { role:'trigger', name:'Вернуть', desc:'вернуть значение' },
    { role:'trigger', name:'Прервать', desc:'выйти из цикла' },
    { role:'trigger', name:'Продолжить', desc:'к следующей итерации' },
    { role:'trigger', name:'Пауза', desc:'задержка в мс или с' },
  ]},
  { cat:'Сеть / Веб', items:[
    { role:'action', name:'Веб-запрос', desc:'HTTP-запрос', danger:true },
    { role:'action', name:'Открыть ссылку', desc:'URL в браузере' },
    { role:'action', name:'Поиск в вебе', desc:'поисковый запрос' },
  ]},
  { cat:'Уведомления', items:[
    { role:'response', name:'Показать уведомление', desc:'системный тост' },
    { role:'response', name:'Строка в оверлее', desc:'запись в оверлей' },
  ]},
  { cat:'Яндекс Музыка', items:[
    { role:'sound', name:'Играть', desc:'старт воспроизведения' },
    { role:'sound', name:'Пауза', desc:'пауза музыки' },
    { role:'sound', name:'Следующий трек', desc:'вперёд' },
    { role:'sound', name:'Предыдущий трек', desc:'назад' },
    { role:'sound', name:'Лайк', desc:'нравится' },
    { role:'sound', name:'Поиск в музыке', desc:'найти трек' },
    { role:'sound', name:'Плейлист', desc:'открыть плейлист' },
  ]},
];

let nodes = [
  { id:'n1', role:'trigger',  title:'Голосовой триггер', x:40,  y:180, summary:'Фраза: <b>«привет»</b> · fuzzy', params:{phrase:'привет', fuzzy:true} },
  { id:'n2', role:'response', title:'Ответ голосом',     x:300, y:180, summary:'Скажет: <b>«Добрый день»</b>', params:{text:'Добрый день'} },
  { id:'n3', role:'sound',    title:'Звук',              x:300, y:360, summary:'chime.wav · 0.4 с', params:{file:'chime.wav'} },
  { id:'n5', role:'condition',title:'Если утро',         x:560, y:180, summary:'Время <b>&lt; 12:00</b>', params:{cond:'hour < 12'} },
  { id:'n2b',role:'response', title:'Утренний ответ',    x:820, y:90,  summary:'Скажет: <b>«Доброе утро»</b>', params:{text:'Доброе утро'} },
  { id:'n4', role:'action',   title:'Открыть браузер',   x:820, y:280, summary:'URL: <b>ya.ru</b>', params:{url:'ya.ru'} },
];
let wires = [
  { id:'w1', from:'n1', out:'out', to:'n2', delay:0 },
  { id:'w2', from:'n2', out:'out', to:'n5', delay:1500 },
  { id:'w3', from:'n1', out:'out', to:'n3', delay:200 },
  { id:'w4', from:'n5', out:'then', to:'n2b', delay:0 },
  { id:'w5', from:'n5', out:'else', to:'n4', delay:0 },
];

let view = { x: 0, y: 0, k: 1 };
let snap = false;
let selected = null;

const $ = id => document.getElementById(id);
const world = $('world'), wiresSvg = $('wires'), canvas = $('canvas');
const NODE_W = 210, NODE_H = 84;
/* стиль провода задаёт вариант: массив пар [толщина, прозрачность] */
const WIRE = (window.AYRIS_WIRE && window.AYRIS_WIRE.glow) || [[8,0.14],[4,0.38],[2,0.92]];

/* ================= Рендер нод ================= */
function renderNodes() {
  document.querySelectorAll('.node').forEach(n=>n.remove());
  nodes.forEach(n => {
    const el = document.createElement('div');
    el.className = 'node' + (n.disabled?' disabled':'') + (n.danger?' danger':'') + (n.running?' running':'') + (selected===n.id?' selected':'');
    el.style.left = n.x+'px'; el.style.top = n.y+'px';
    el.style.setProperty('--role', ROLES[n.role].color);
    el.dataset.id = n.id;
    const inFilled  = wires.some(w=>w.to===n.id) ? ' filled':'';
    const outs = outsOf(n);
    let portsHtml = '';
    outs.forEach((o,i)=>{
      const filled = wires.some(w=>w.from===n.id && w.out===o) ? ' filled':'';
      const col = OUT_META[o].color || ROLES[n.role].color;
      const topPct = outs.length===1 ? 50 : (58 + i*24);
      portsHtml += `<div class="port out multi${filled}" data-out="${o}" style="top:${topPct}%; --pc:${col}; border-color:${col}; ${filled?'background:'+col+';':''}"></div>`;
      if(OUT_META[o].label) portsHtml += `<div class="port-label" style="top:calc(${topPct}% - 6px); color:${col}">${OUT_META[o].label}</div>`;
    });
    el.innerHTML = `
      <div class="accent-bar"></div>
      <div class="head">
        <div class="ico">${ICONS[n.role]}</div>
        <div style="flex:1; overflow:hidden;">
          <div class="title">${n.title}</div>
          <div class="role-label">${ROLES[n.role].label}</div>
        </div>
        <div class="kebab" data-kebab="${n.id}">⋯</div>
      </div>
      <div class="body"><div class="summary">${n.summary}</div></div>
      ${n.role!=='trigger' ? `<div class="port in${inFilled}"></div>`:''}
      ${portsHtml}
    `;
    world.appendChild(el);
  });
  measureAnchors();
  bindNodeDrag(); bindPorts(); bindKebabs();
}

/* Точные якоря портов: снимаем реальный центр каждого порта из DOM
   (учитывает высоту ноды, рамку роли и свечение) и переводим в
   несмасштабированные координаты мира — провод всегда попадает в кружок. */
let anchors = {};
function measureAnchors(){
  const k = (typeof view!=='undefined' && view.k) ? view.k : 1;
  anchors = {};
  nodes.forEach(n=>{
    const el = world.querySelector(`.node[data-id="${n.id}"]`); if(!el) return;
    const nr = el.getBoundingClientRect();
    const rec = {};
    el.querySelectorAll('.port').forEach(p=>{
      const key = p.classList.contains('in') ? 'in' : (p.dataset.out||'out');
      const pr = p.getBoundingClientRect();
      rec[key] = { dx:(pr.left+pr.width/2-nr.left)/k, dy:(pr.top+pr.height/2-nr.top)/k };
    });
    anchors[n.id] = rec;
  });
}

/* ================= Рендер проводов ================= */
function nodeById(id){ return nodes.find(n=>n.id===id); }
function outPos(node, out){
  const key = out || 'out';
  const a = anchors[node.id];
  if(a && a[key]) return { x: node.x + a[key].dx, y: node.y + a[key].dy };
  const outs = outsOf(node);
  const idx = Math.max(0, outs.indexOf(out));
  const topPct = outs.length===1 ? 50 : (58 + idx*24);
  return { x: node.x + NODE_W, y: node.y + NODE_H*(topPct/100) };
}
function inPos(node){
  const a = anchors[node.id];
  if(a && a.in) return { x: node.x + a.in.dx, y: node.y + a.in.dy };
  return { x: node.x, y: node.y + NODE_H/2 };
}
function bezier(p1,p2){
  const dx = Math.max(60, Math.abs(p2.x-p1.x)*0.5);
  return `M ${p1.x} ${p1.y} C ${p1.x+dx} ${p1.y}, ${p2.x-dx} ${p2.y}, ${p2.x} ${p2.y}`;
}
function glowPath(d,col){
  return WIRE.map(([w,o])=>`<path d="${d}" fill="none" style="stroke:${col}" stroke-width="${w}" stroke-linecap="round" opacity="${o}"/>`).join('');
}
function renderWires(live){
  let paths = '';
  document.querySelectorAll('.delay-chip,.wire-add').forEach(e=>e.remove());
  wires.forEach(w=>{
    const a = nodeById(w.from), b = nodeById(w.to);
    if(!a||!b) return;
    const p1 = outPos(a, w.out||'out'), p2 = inPos(b);
    const d = bezier(p1,p2);
    const col = OUT_META[w.out||'out'].color || ROLES[a.role].color;
    paths += glowPath(d, col);
    const mx = (p1.x+p2.x)/2, my = (p1.y+p2.y)/2;
    const chip = document.createElement('div');
    if (w.delay>0){
      chip.className='delay-chip';
      chip.style.left=mx+'px'; chip.style.top=my+'px';
      chip.innerHTML = `${CLOCK}${(w.delay/1000).toFixed(w.delay%1000?1:0)} с`;
      chip.onclick=(e)=>{ e.stopPropagation(); editDelay(w); };
    } else {
      chip.className='wire-add';
      chip.style.left=mx+'px'; chip.style.top=my+'px';
      chip.textContent='+'; chip.title='Добавить задержку';
      chip.onmouseenter=()=>chip.style.opacity=1;
      chip.onmouseleave=()=>chip.style.opacity=0;
      chip.onclick=(e)=>{ e.stopPropagation(); editDelay(w); };
    }
    world.appendChild(chip);
  });
  if(live && live.d) paths += `<path d="${live.d}" fill="none" style="stroke:${live.col}" stroke-width="2.5" stroke-linecap="round" stroke-dasharray="6 5" opacity="0.9"/>`;
  wiresSvg.innerHTML = paths;
  drawMinimap();
}
function editDelay(w){
  const v = prompt('Задержка на связи (мс):', w.delay);
  if(v!==null){ w.delay = Math.max(0, parseInt(v)||0); renderWires(); commit(); }
}

/* ================= Живой провод от порта ================= */
let liveWire = null;
function worldPoint(ev){
  const r = canvas.getBoundingClientRect();
  return { x:(ev.clientX-r.left-view.x)/view.k, y:(ev.clientY-r.top-view.y)/view.k };
}
function bindPorts(){
  document.querySelectorAll('.port.out').forEach(p=>{
    p.addEventListener('mousedown', e=>{
      e.stopPropagation();
      const nodeEl = p.closest('.node');
      const from = nodeEl.dataset.id, out = p.dataset.out||'out';
      const col = OUT_META[out].color || ROLES[nodeById(from).role].color;
      liveWire = { from, out, col };
      function move(ev){
        const p1 = outPos(nodeById(from), out), p2 = worldPoint(ev);
        document.querySelectorAll('.node').forEach(el=>{
          const over = el.matches(':hover') && el.dataset.id!==from && nodeById(el.dataset.id).role!=='trigger';
          el.style.outline = over ? '2px solid var(--success)' : '';
        });
        renderWires({d:bezier(p1,p2), col});
      }
      function up(ev){
        document.removeEventListener('mousemove',move); document.removeEventListener('mouseup',up);
        const tgt = document.elementsFromPoint(ev.clientX,ev.clientY).map(x=>x.closest&&x.closest('.node')).find(Boolean);
        document.querySelectorAll('.node').forEach(el=>el.style.outline='');
        if(tgt && tgt.dataset.id!==from && nodeById(tgt.dataset.id).role!=='trigger'){
          wires.push({ id:'w'+(Date.now()%100000), from, out, to:tgt.dataset.id, delay:0 });
          liveWire = null; renderNodes(); renderWires(); commit(); return;
        }
        liveWire = null; renderNodes(); renderWires();
      }
      document.addEventListener('mousemove',move); document.addEventListener('mouseup',up);
    });
  });
}

/* ================= Перетаскивание нод ================= */
function bindNodeDrag(){
  document.querySelectorAll('.node').forEach(el=>{
    const id = el.dataset.id;
    el.addEventListener('mousedown', e=>{
      if(e.target.closest('.port')) return;
      e.stopPropagation();
      selectNode(id);
      const n = nodeById(id);
      const sx = e.clientX, sy = e.clientY, ox = n.x, oy = n.y;
      function move(ev){
        let nx = ox + (ev.clientX-sx)/view.k;
        let ny = oy + (ev.clientY-sy)/view.k;
        if(snap){ nx = Math.round(nx/26)*26; ny = Math.round(ny/26)*26; }
        n.x = nx; n.y = ny;
        el.style.left=nx+'px'; el.style.top=ny+'px';
        renderWires();
      }
      function up(){ document.removeEventListener('mousemove',move); document.removeEventListener('mouseup',up); if(n.x!==ox||n.y!==oy) commit(); }
      document.addEventListener('mousemove',move); document.addEventListener('mouseup',up);
    });
  });
}

/* ================= Панорама + зум к курсору ================= */
canvas.addEventListener('mousedown', e=>{
  if(e.target.closest('.node')||e.target.closest('.delay-chip')) return;
  selectNode(null);
  canvas.classList.add('panning');
  const sx=e.clientX, sy=e.clientY, ox=view.x, oy=view.y;
  function move(ev){ view.x=ox+(ev.clientX-sx); view.y=oy+(ev.clientY-sy); applyView(); }
  function up(){ canvas.classList.remove('panning'); document.removeEventListener('mousemove',move); document.removeEventListener('mouseup',up); }
  document.addEventListener('mousemove',move); document.addEventListener('mouseup',up);
});
canvas.addEventListener('wheel', e=>{
  e.preventDefault();
  const factor = e.deltaY<0 ? 1.1 : 0.9;
  const nk = Math.min(2.2, Math.max(0.4, view.k*factor));
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX-rect.left, my = e.clientY-rect.top;
  view.x = mx - (mx-view.x)*(nk/view.k);
  view.y = my - (my-view.y)*(nk/view.k);
  view.k = nk; applyView();
},{passive:false});
function applyView(){
  world.style.transform = `translate(${view.x}px,${view.y}px) scale(${view.k})`;
  /* пан/зум сетки выражаем через переменные — вариант сам решает, сколько слоёв рисовать */
  canvas.style.setProperty('--vx', view.x+'px');
  canvas.style.setProperty('--vy', view.y+'px');
  canvas.style.setProperty('--vk', view.k);
}
function fitAll(){
  if(!nodes.length) return;
  let minX=1e9,minY=1e9,maxX=-1e9,maxY=-1e9;
  nodes.forEach(n=>{ minX=Math.min(minX,n.x); minY=Math.min(minY,n.y); maxX=Math.max(maxX,n.x+NODE_W); maxY=Math.max(maxY,n.y+NODE_H); });
  const pad=80; minX-=pad;minY-=pad;maxX+=pad;maxY+=pad;
  const r = canvas.getBoundingClientRect();
  const k = Math.min(2.2, Math.max(0.4, Math.min(r.width/(maxX-minX), r.height/(maxY-minY))));
  view.k = k; view.x = (r.width-(maxX-minX)*k)/2 - minX*k; view.y = (r.height-(maxY-minY)*k)/2 - minY*k;
  applyView();
}

/* ================= Выбор + инспектор ================= */
function selectNode(id){
  selected = id;
  document.querySelectorAll('.node').forEach(el=>el.classList.toggle('selected', el.dataset.id===id));
  renderInspector();
}
function field(label, value, type='input'){
  if(type==='textarea') return `<div class="field"><label>${label}</label><textarea>${value}</textarea></div>`;
  return `<div class="field"><label>${label}</label><input value="${value}"></div>`;
}
function renderInspector(){
  const insp = $('inspector'); if(!insp) return;
  if(!selected){ insp.innerHTML = '<div class="insp-empty">Выбери ноду, чтобы настроить её параметры</div>'; return; }
  const n = nodeById(selected);
  const role = ROLES[n.role];
  let body='';
  if(n.role==='trigger'){
    body = field('Фраза', n.params.phrase)
      + `<div class="toggle-row"><span>Нечёткое совпадение (fuzzy)</span><div class="sw on"></div></div>`
      + `<div class="toggle-row"><span>Регулярное выражение</span><div class="sw"></div></div>`
      + field('Приоритет','5');
  } else if(n.role==='response'){
    body = field('Текст ответа', n.params.text, 'textarea')
      + `<div class="field"><label>Голос</label><select><option>Ayris (по умолчанию)</option><option>Мужской</option></select></div>`
      + `<div class="row">${field('Скорость','1.0')}${field('Тон','0')}</div>`;
  } else if(n.role==='action'){
    body = `<div class="field"><label>Действие</label><select><option>Открыть браузер</option><option>Нажать клавишу</option><option>Запустить программу</option></select></div>`
      + field('URL', n.params.url)
      + `<div class="toggle-row"><span>Требует прав администратора</span><div class="sw"></div></div>`;
  } else if(n.role==='sound'){
    body = `<div class="field"><label>Источник</label><select><option>Файл</option><option>Библиотека</option><option>Синтез TTS</option></select></div>`
      + field('Файл', n.params.file)
      + `<div class="toggle-row"><span>Ждать окончания</span><div class="sw on"></div></div>`;
  }
  insp.innerHTML = `
    <div class="insp-head" style="--role:${role.color}">
      <div class="ico">${ICONS[n.role]}</div>
      <div><div class="t">${n.title}</div><div class="s">${role.label}</div></div>
    </div>
    <div class="insp-body">
      ${field('Название', n.title)}
      ${body}
      <div class="toggle-row"><span>Нода включена</span><div class="sw ${n.disabled?'':'on'}"></div></div>
    </div>`;
  insp.querySelectorAll('.sw').forEach(sw=>sw.onclick=()=>sw.classList.toggle('on'));
}

/* ================= Тулбар (все кнопки необязательны) ================= */
function on(id, ev, fn){ const el=$(id); if(el) el.addEventListener(ev, fn); }

/* ================= История (отмена / повтор) ================= */
let past=[], future=[];
function snapshot(){ return JSON.stringify({ nodes, wires, selected }); }
let _cur = snapshot();
function commit(){ past.push(_cur); if(past.length>200) past.shift(); future.length=0; _cur=snapshot(); updateHistoryButtons(); }
function applySnapshot(s){
  const o=JSON.parse(s); nodes=o.nodes; wires=o.wires; selected=o.selected;
  renderNodes(); renderWires(); renderInspector();
}
function undo(){ if(!past.length) return; future.push(_cur); _cur=past.pop(); applySnapshot(_cur); updateHistoryButtons(); }
function redo(){ if(!future.length) return; past.push(_cur); _cur=future.pop(); applySnapshot(_cur); updateHistoryButtons(); }
function updateHistoryButtons(){
  const u=$('undoBtn'), r=$('redoBtn');
  if(u){ u.disabled=!past.length; u.classList.toggle('disabled', !past.length); }
  if(r){ r.disabled=!future.length; r.classList.toggle('disabled', !future.length); }
}
on('undoBtn','click', undo);
on('redoBtn','click', redo);
document.addEventListener('keydown', e=>{
  if(e.target && e.target.matches && e.target.matches('input,textarea,select')) return;
  const k=e.key.toLowerCase();
  if((e.ctrlKey||e.metaKey) && k==='z'){ e.preventDefault(); e.shiftKey?redo():undo(); }
  else if((e.ctrlKey||e.metaKey) && k==='y'){ e.preventDefault(); redo(); }
});
function arrange(){
  const layer = {}; nodes.forEach(n=>layer[n.id]=0);
  let changed=true, guard=0;
  while(changed && guard++<20){ changed=false;
    wires.forEach(w=>{ if(layer[w.to] < layer[w.from]+1){ layer[w.to]=layer[w.from]+1; changed=true; } });
  }
  const byLayer = {};
  nodes.forEach(n=>{ (byLayer[layer[n.id]] ||= []).push(n); });
  Object.entries(byLayer).forEach(([l,ns])=>{ ns.forEach((n,i)=>{ n.x = 60 + (+l)*270; n.y = 120 + i*170; }); });
  renderNodes(); renderWires(); commit();
}
on('snapBtn','click', e=>{ snap=!snap; e.currentTarget.classList.toggle('active',snap); });
on('arrangeBtn','click', arrange);
on('fitBtn','click', fitAll);
on('resetBtn','click', ()=>{ view={x:40,y:40,k:1}; applyView(); });
on('saveBtn','click', ()=>{ const b=$('saveBtn'); const t=b.textContent; b.textContent='✓ Сохранено'; setTimeout(()=>b.textContent=t,1400); });
on('runBtn','click', ()=>{
  nodes.forEach(n=>n.running=false); renderNodes();
  const order = [...nodes].sort((a,b)=>a.x-b.x);
  let i=0;
  const t=setInterval(()=>{
    nodes.forEach(n=>n.running=false);
    if(i<order.length){ nodeById(order[i].id).running=true; i++; renderNodes(); }
    else { clearInterval(t); nodes.forEach(n=>n.running=false); renderNodes(); }
  },600);
});

/* ================= Кебаб → контекст-меню ================= */
function bindKebabs(){
  document.querySelectorAll('.kebab').forEach(k=>{
    k.addEventListener('click', e=>{ e.stopPropagation(); const r=k.getBoundingClientRect(); openCtx(r.left, r.bottom+4, k.dataset.kebab); });
  });
}
const ctx = $('ctx');
function openCtx(x, y, nodeId){
  if(!ctx) return;
  const onNode = !!nodeId;
  const n = onNode ? nodeById(nodeId) : null;
  const items = onNode ? [
    {t:'Переименовать', k:'F2', fn:()=>{ const v=prompt('Название ноды:', n.title); if(v){ n.title=v; renderNodes(); renderWires(); selectNode(n.id); commit();} }},
    {t:'Дублировать', k:'Ctrl+D', fn:()=>{ const id='n'+(Date.now()%100000); nodes.push({...n, id, x:n.x+30, y:n.y+30, title:n.title+' — копия'}); renderNodes(); renderWires(); selectNode(id); commit(); }},
    {t: n.disabled?'Включить':'Выключить', fn:()=>{ n.disabled=!n.disabled; renderNodes(); renderWires(); commit(); }},
    {sep:true},
    {t:'Удалить', k:'Del', danger:true, fn:()=>{ nodes=nodes.filter(x=>x.id!==n.id); wires=wires.filter(w=>w.from!==n.id&&w.to!==n.id); selected=null; renderNodes(); renderWires(); renderInspector(); commit(); }},
  ] : [
    {t:'Добавить ноду', fn:()=>{ const a=$('addBtn'); if(a) a.click(); }},
    {t:'Упорядочить', fn:arrange},
    {t:'Показать всё', fn:fitAll},
  ];
  ctx.innerHTML = items.map(it=> it.sep ? '<div class="sep"></div>'
    : `<div class="ci ${it.danger?'danger':''}">${it.t}${it.k?`<span class="k">${it.k}</span>`:''}</div>`).join('');
  let idx=0; ctx.querySelectorAll('.ci').forEach(el=>{ const it=items.filter(x=>!x.sep)[idx++]; el.onclick=()=>{ ctx.classList.remove('open'); it.fn(); }; });
  ctx.style.left=x+'px'; ctx.style.top=y+'px'; ctx.classList.add('open');
  /* меню — position:fixed; зажимаем в окно, чтобы не уезжало за нижний/правый край */
  const m=8, w=ctx.offsetWidth, h=ctx.offsetHeight;
  ctx.style.left=Math.max(m, Math.min(x, window.innerWidth - w - m))+'px';
  ctx.style.top =Math.max(m, Math.min(y, window.innerHeight - h - m))+'px';
}
canvas.addEventListener('contextmenu', e=>{
  e.preventDefault();
  const nodeEl = e.target.closest('.node');
  openCtx(e.clientX, e.clientY, nodeEl?nodeEl.dataset.id:null);
});
document.addEventListener('click', e=>{ if(ctx && !e.target.closest('#ctx')) ctx.classList.remove('open'); });

/* ================= Миникарта (необязательна) ================= */
const minimap = $('minimap'), mmToggle = $('mmToggle');
if($('mmClose')) $('mmClose').onclick = ()=>{ minimap.classList.add('hidden'); if(mmToggle) mmToggle.classList.add('show'); };
if(mmToggle) mmToggle.onclick = ()=>{ minimap.classList.remove('hidden'); mmToggle.classList.remove('show'); drawMinimap(); };
function drawMinimap(){
  if(!minimap || minimap.classList.contains('hidden')) return;
  const c = $('mmCanvas'); if(!c) return; const g = c.getContext('2d');
  g.clearRect(0,0,c.width,c.height);
  if(!nodes.length) return;
  let minX=1e9,minY=1e9,maxX=-1e9,maxY=-1e9;
  nodes.forEach(n=>{ minX=Math.min(minX,n.x); minY=Math.min(minY,n.y); maxX=Math.max(maxX,n.x+NODE_W); maxY=Math.max(maxY,n.y+NODE_H); });
  const pad=40; minX-=pad;minY-=pad;maxX+=pad;maxY+=pad;
  const s = Math.min(c.width/(maxX-minX), c.height/(maxY-minY));
  const css = getComputedStyle(document.documentElement);
  const resolve = v => v.startsWith('var') ? css.getPropertyValue(v.slice(4,-1)).trim() : v;
  wires.forEach(w=>{ const a=nodeById(w.from),b=nodeById(w.to); if(!a||!b)return;
    g.strokeStyle=resolve('var(--accent)'); g.globalAlpha=0.5; g.beginPath();
    g.moveTo((a.x+NODE_W-minX)*s,(a.y+NODE_H/2-minY)*s); g.lineTo((b.x-minX)*s,(b.y+NODE_H/2-minY)*s); g.stroke(); });
  g.globalAlpha=1;
  nodes.forEach(n=>{ g.fillStyle = resolve(ROLES[n.role].color);
    g.fillRect((n.x-minX)*s,(n.y-minY)*s, NODE_W*s, NODE_H*s); });
}

/* ================= Палитра «＋ Нода» (необязательна) ================= */
const palette = $('palette'), pList = $('pList'), pSearch = $('pSearch');
function addNode(role, name, danger){
  const id='n'+(Date.now()%100000);
  nodes.push({ id, role, title:name, danger:!!danger,
    x:(-view.x+340)/view.k, y:(-view.y+240)/view.k, summary:'Не настроено', params:{} });
  renderNodes(); renderWires(); selectNode(id); commit(); return id;
}
function renderPalette(q=''){
  if(!pList) return;
  q=q.trim().toLowerCase(); let html='', any=false;
  BLOCKS.forEach(cat=>{
    const hits = cat.items.filter(b=> !q || b.name.toLowerCase().includes(q) || b.desc.toLowerCase().includes(q));
    if(!hits.length) return; any=true;
    html += `<div class="p-cat">${cat.cat}</div>`;
    hits.forEach(b=>{ html += `<div class="p-item" data-role="${b.role}" data-name="${b.name}" data-danger="${b.danger?1:0}" style="--role:${ROLES[b.role].color}">
      <div class="ico">${catIcon(cat.cat, b.role)}</div><div class="meta"><div class="nm">${b.name}</div><div class="ds">${b.desc}</div></div></div>`; });
  });
  pList.innerHTML = any ? html : '<div class="p-empty">Ничего не найдено</div>';
  pList.querySelectorAll('.p-item').forEach(it=>it.onclick=()=>{
    addNode(it.dataset.role, it.dataset.name, it.dataset.danger==='1');
    palette.classList.remove('open');
  });
}
on('addBtn','click', e=>{
  if(!palette) return;
  const willOpen = !palette.classList.contains('open');
  palette.classList.toggle('open', willOpen);
  if(willOpen){
    const r=e.currentTarget.getBoundingClientRect();
    const pw=264, m=8;
    /* капсула снизу → открываем палитру ВВЕРХ (якорь по bottom), left зажимаем в окно */
    const left=Math.max(m, Math.min(r.left, window.innerWidth - pw - m));
    palette.style.left=left+'px';
    palette.style.right='auto';
    palette.style.top='auto';
    palette.style.bottom=(window.innerHeight - r.top + m)+'px';
    if(pSearch){pSearch.value=''; pSearch.focus();}
    renderPalette();
  }
});
if(pSearch) pSearch.oninput = ()=>renderPalette(pSearch.value);
document.addEventListener('click', e=>{ if(palette && !e.target.closest('#palette') && !e.target.closest('#addBtn')) palette.classList.remove('open'); });

/* ================= Левый каталог блоков (необязателен) ================= */
function renderCatalog(q=''){
  const el = $('catalog'); if(!el) return;
  q=q.trim().toLowerCase(); let html='';
  BLOCKS.forEach(cat=>{
    const hits = cat.items.filter(b=> !q || b.name.toLowerCase().includes(q) || b.desc.toLowerCase().includes(q));
    if(!hits.length) return;
    html += `<div class="cat-group"><div class="cat-title">${cat.cat}</div>`;
    hits.forEach(b=>{ html += `<div class="cat-item" data-role="${b.role}" data-name="${b.name}" data-danger="${b.danger?1:0}" style="--role:${ROLES[b.role].color}">
      <span class="ci-ico">${catIcon(cat.cat, b.role)}</span><span class="ci-meta"><span class="ci-nm">${b.name}</span><span class="ci-ds">${b.desc}</span></span></div>`; });
    html += `</div>`;
  });
  el.innerHTML = html || '<div class="cat-empty">Ничего не найдено</div>';
  el.querySelectorAll('.cat-item').forEach(it=>it.onclick=()=>addNode(it.dataset.role, it.dataset.name, it.dataset.danger==='1'));
}
if($('catalogSearch')) $('catalogSearch').oninput = e=>renderCatalog(e.target.value);

/* ================= Инициализация ================= */
applyView(); renderNodes(); renderWires(); renderCatalog(); selectNode('n2');
_cur = snapshot(); updateHistoryButtons();
