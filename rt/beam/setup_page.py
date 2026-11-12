"""The Beam setup panel of rt's status page (http://<laptop>:8765/beam), for setting the beam up on site from a phone.

One static page; its script asks /beam/setup.json ~5 times a second for the live view (the four modem lines, the
last byte messages, the beam's numbers) and sends the owner's actions as small JSON POSTs:
  /beam/setup/start   open the chosen port in monitor mode (the beam driver pauses meanwhile)
  /beam/setup/stop    leave setup without changes (the beam restarts with the saved settings)
  /beam/monitor/clear forget what the monitor saw (a fresh start for Suggest)
  /beam/suggest       the probe's guess from what the monitor saw (fills the form; nothing is applied)
  /beam/apply         validate + save to <config>.beam.toml + restart only the beam with these settings
  /beam/off           driver = off, saved
Labels are in English like the rest of the status page. If [status] setup_token is set, the page asks for it once
and keeps it in the phone's browser (localStorage); it goes in the X-Setup-Token header.
"""

SETUP_PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>rt beam setup</title>
<style>
body{font-family:system-ui,sans-serif;margin:0 12px 40px;color:#111}h1{font-size:1.2em}h2{font-size:1em;margin:1em 0 .3em}
.lines{display:flex;gap:6px}.line{flex:1;text-align:center;padding:10px 0;border-radius:6px;background:#ddd;font-weight:bold}
.line.on{background:#0a0;color:#fff}.line.on small{color:#fff}.line.none{background:#eee;color:#999}
table{border-collapse:collapse;width:100%;font-size:.85em}td,th{border-bottom:1px solid #ddd;padding:3px 4px;text-align:left}
label{display:block;margin:6px 0 2px;font-size:.9em}input,select{font-size:1em;width:100%;box-sizing:border-box;padding:6px}
.row{display:flex;gap:8px}.row>div{flex:1}button{font-size:1em;padding:10px 12px;margin:4px 4px 4px 0;border-radius:6px;border:1px solid #888}
.primary{background:#06c;color:#fff;border-color:#06c}.danger{background:#c00;color:#fff;border-color:#c00}
.box{background:#fdd;border:1px solid #c00;padding:6px;margin:6px 0}.note{background:#eef;border:1px solid #88c;padding:6px;margin:6px 0}
.ok{color:#070}.bad{color:#c00}small{color:#666}.mono{font-family:ui-monospace,monospace}
</style></head><body>
<h1>Beam setup</h1><small><a href="/">&larr; status page</a> &middot; changes here restart only the beam, never
detection, capture or recording</small>
<div id="msg"></div>
<h2>1. Port</h2>
<div class="row"><div><label>Port</label><select id="port"></select></div>
<div><label>Baud</label><select id="baud"></select></div></div>
<div class="row"><div><label><input type="checkbox" id="set_dtr" style="width:auto"> DTR on</label></div>
<div><label><input type="checkbox" id="set_rts" style="width:auto"> RTS on</label></div></div>
<small>DTR / RTS on can power an open-collector sensor's pull-up (see rt/README.md "Beam input").</small><br>
<button class="primary" onclick="act('/beam/setup/start', portForm())">Start monitor</button>
<button onclick="act('/beam/setup/stop', {})">Leave setup</button>
<h2>2. Live monitor <small id="monstate"></small></h2>
<div class="lines"><div class="line none" id="l_CTS">CTS</div><div class="line none" id="l_DSR">DSR</div>
<div class="line none" id="l_DCD">DCD</div><div class="line none" id="l_RI">RI</div></div>
<small>green = asserted. Wave a hand through the beam: one line should flip, or bytes should arrive.</small>
<table><thead><tr><th>time</th><th>bytes (hex)</th><th>text</th></tr></thead><tbody id="messages"></tbody></table>
<small id="changes"></small><br>
<button onclick="act('/beam/monitor/clear', {})">Clear</button>
<button class="primary" onclick="suggest()">Suggest</button>
<div id="suggestion"></div>
<h2>3. Settings</h2>
<div class="row"><div><label>Driver</label><select id="driver"><option>off</option><option>modem-line</option>
<option>byte-stream</option></select></div>
<div><label>Line (modem-line)</label><select id="line"><option value="cts">CTS</option><option value="dsr">DSR</option>
<option value="cd">DCD</option><option value="ri">RI</option></select></div></div>
<div class="row"><div><label>Active (beam blocked =)</label><select id="active"><option value="high">high (asserted)</option>
<option value="low">low (not asserted)</option></select></div>
<div><label>Group gap (s)</label><input id="group_gap_s" inputmode="decimal"></div></div>
<div class="row"><div><label>Byte pattern (hex)</label><input id="byte_pattern" class="mono"></div>
<div><label>Release pattern (hex)</label><input id="byte_release_pattern" class="mono"></div></div>
<div class="row"><div><label>Stuck after (s)</label><input id="stuck_s" inputmode="decimal"></div><div></div></div>
<button class="primary" onclick="act('/beam/apply', settingsForm())">Apply</button>
<button class="danger" onclick="if(confirm('Turn the beam off?'))act('/beam/off', {})">Turn off</button>
<h2>4. Beam now</h2><div id="beam"></div>
<table><thead><tr><th>break</th><th>edges</th><th>duration</th><th>pass</th></tr></thead><tbody id="breaks"></tbody></table>
<p><small id="footer"></small></p>
<script>
var BAUDS=[300,600,1200,2400,4800,9600,14400,19200,38400,57600,115200,230400,460800,921600];
var loaded=false, editing=false;
function $(id){return document.getElementById(id)}
function esc(s){return String(s==null?'':s).replace(/[&<>"]/g,function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]})}
function hms(t){if(t==null)return '';var d=new Date(t*1000);return d.toTimeString().slice(0,8)+'.'+String(d.getMilliseconds()).padStart(3,'0')}
function token(){return localStorage.getItem('rtSetupToken')||''}
function show(html,cls){$('msg').innerHTML=html?'<div class="'+(cls||'note')+'">'+html+'</div>':''}
BAUDS.forEach(function(b){var o=document.createElement('option');o.value=b;o.text=b;$('baud').appendChild(o)});
document.querySelectorAll('input,select').forEach(function(el){el.addEventListener('focus',function(){editing=true});
  el.addEventListener('blur',function(){editing=false})});
function portForm(){return {port:$('port').value,baud:+$('baud').value,set_dtr:$('set_dtr').checked,set_rts:$('set_rts').checked}}
function settingsForm(){var f=portForm();['driver','line','active','byte_pattern','byte_release_pattern','group_gap_s','stuck_s']
  .forEach(function(k){f[k]=$(k).value});return f}
function fill(s){if(!s)return;['driver','line','active','byte_pattern','byte_release_pattern','group_gap_s','stuck_s']
  .forEach(function(k){if(s[k]!=null)$(k).value=s[k]});if(s.baud)$('baud').value=s.baud;
  if(s.port)setPort(s.port);if(s.set_dtr!=null)$('set_dtr').checked=s.set_dtr;if(s.set_rts!=null)$('set_rts').checked=s.set_rts}
function setPort(p){var sel=$('port');if(![].some.call(sel.options,function(o){return o.value==p})){var o=document.createElement('option');
  o.value=p;o.text=p+' (not found now)';sel.appendChild(o)}sel.value=p}
function post(url,body){return fetch(url,{method:'POST',headers:{'Content-Type':'application/json','X-Setup-Token':token()},
  body:JSON.stringify(body)}).then(function(r){return r.json().then(function(j){j.http=r.status;return j})})}
function act(url,body){post(url,body).then(function(j){
  if(j.http==403){var t=prompt('Setup PIN / token ([status] setup_token):');if(t!=null){localStorage.setItem('rtSetupToken',t);return act(url,body)}}
  if(j.ok){show(esc(j.message||'done'),'note')}else{show((j.errors||['failed']).map(esc).join('<br>'),'box')}
  refresh()}).catch(function(e){show('request failed: '+esc(e),'box')})}
function suggest(){post('/beam/suggest',{}).then(function(j){
  if(j.http==403){var t=prompt('Setup PIN / token ([status] setup_token):');if(t!=null){localStorage.setItem('rtSetupToken',t);return suggest()}}
  if(!j.ok){show((j.errors||['failed']).map(esc).join('<br>'),'box');return}
  var g=j.suggestion;var h='<div class="note"><b>Suggestion:</b> '+(g.driver?esc(g.config.join(', ')):'nothing usable seen')+
  '<ul>'+g.notes.map(function(n){return '<li>'+esc(n)+'</li>'}).join('')+'</ul>'+(g.driver?'Filled into the settings below: check them, then Apply.':'')+'</div>';
  $('suggestion').innerHTML=h;if(g.driver){fill(g.settings)}})}
function refresh(){fetch('/beam/setup.json').then(function(r){return r.json()}).then(function(d){
  var sel=$('port'),cur=sel.value;if(!editing){sel.innerHTML='';d.ports.forEach(function(p){var o=document.createElement('option');
    o.value=p.device;o.text=p.device+(p.cp210x?' - CP2102 (10C4:EA60)':'')+' '+p.description;sel.appendChild(o)});
    if(cur)setPort(cur)}
  if(!loaded){fill(d.settings);loaded=true;var here=d.ports.map(function(p){return p.device});
    // First visit: the monitored port, else the saved one if it exists now, else the CP2102, else the first port.
    var pick=d.monitor?d.monitor.port:(here.indexOf(d.settings.port)>=0?d.settings.port:
      ((d.ports.filter(function(p){return p.cp210x})[0]||d.ports[0]||{}).device));if(pick)setPort(pick)}
  var m=d.monitor;$('monstate').innerHTML=m?('<span class="'+(m.state=='open'?'ok':'bad')+'">'+esc(m.state)+'</span> '+esc(m.detail)):'(off: press Start monitor)';
  ['CTS','DSR','DCD','RI'].forEach(function(k){var v=m?m.lines[k]:null;$('l_'+k).className='line '+(v==null?'none':(v?'on':''));
    $('l_'+k).innerHTML=k+(m?'<br><small>'+m.transitions[k]+' changes</small>':'')});
  $('messages').innerHTML=m?m.messages.slice(0,15).map(function(x){return '<tr><td>'+hms(x.t_wall)+'</td><td class="mono">'+esc(x.hex)+
    '</td><td class="mono">'+esc(x.text)+'</td></tr>'}).join(''):'';
  $('changes').innerHTML=m&&m.changes.length?'last line changes: '+m.changes.slice(0,6).map(function(c){return hms(c.t_wall)+' '+c.line+
    (c.asserted?' on':' off')}).join(', '):'';
  var b=d.beam;var p=b.poll||{};
  $('beam').innerHTML=(d.error?'<div class="box">'+esc(d.error)+'</div>':'')+(d.setup_mode?'<div class="note">setup mode: the beam driver is paused'+
    ' (it restarts by itself after '+d.setup_idle_s+' s without this page open)</div>':'')+
    (b.enabled?'<span class="'+(b.state=='open'&&!b.stuck?'ok':'bad')+'">'+esc(b.stuck?'STUCK':b.state)+'</span> '+esc(b.driver)+' on '+esc(b.port)+
    ', line '+esc(b.line)+', level <b>'+esc(b.level==1?'ACTIVE (blocked)':b.level==0?'clear':'unknown')+'</b><br><small>'+b.edges_per_min+' edges/min &middot; '+
    b.breaks_total+' breaks &middot; poll '+(p.rate_hz==null?'-':p.rate_hz+' Hz, max gap '+p.max_gap_ms+' ms')+'</small>':'beam off');
  $('breaks').innerHTML=(b.last_breaks||[]).map(function(x){return '<tr><td>'+esc((x.t_beam_iso||'').slice(11,23))+'</td><td>'+x.n_edges+
    '</td><td>'+x.duration_s+' s</td><td>'+(x.pass_id==null?'':'pass '+x.pass_id+(x.number!=null?' #'+esc(x.number):''))+'</td></tr>'}).join('');
  $('footer').innerHTML='saved to '+esc(d.override_file)+' &middot; applied '+d.restarts+' time(s) this run';
}).catch(function(){$('monstate').innerHTML='<span class="bad">rt does not answer</span>'})}
refresh();setInterval(refresh,200);
</script></body></html>"""


def render_setup_page():
    return SETUP_PAGE
