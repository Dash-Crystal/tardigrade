"""Build standalone 2D replay HTML with embedded radar image and match data."""
import sys, json, base64
sys.path.insert(0, 'hci128_repo')
from read_v2 import load
from pathlib import Path

tard_file = sys.argv[1] if len(sys.argv) > 1 else 'tardigrade_v2.1_out/0.tard'

df = load(tard_file)
map_name = df.attrs.get('map_name', 'de_mirage')

frames = []
for t in sorted(df['tick'].unique())[::4]:  # 32Hz sampling for smoother movement
    td = df[(df['tick']==t) & (df['hp']>0)]
    fr = [{'id':int(r['agent']),'x':float(r['x']),'y':float(r['y']),
           'yaw':float(r['abs_yaw']),'team':int(r['team']),'hp':int(r['hp']),
           'fire':int(r.get('action',0))} for _,r in td.iterrows()]
    if fr: frames.append(fr)

radar_path = Path(f'radar/map/{map_name}.png')
if not radar_path.exists():
    radar_path = Path(f'radar/map/default.png')
radar_b64 = base64.b64encode(radar_path.read_bytes()).decode()

MAPS = {
    'de_mirage':  {'xRef':-3230,'yRef':1713,'scale':5.0},
    'de_dust2':   {'xRef':-2476,'yRef':3239,'scale':4.4},
    'de_inferno': {'xRef':-2087,'yRef':3870,'scale':4.9},
    'de_ancient': {'xRef':-2953,'yRef':2164,'scale':5.0},
    'de_anubis':  {'xRef':-2796,'yRef':3328,'scale':5.22},
    'de_nuke':    {'xRef':-3453,'yRef':2887,'scale':7.0},
    'de_vertigo': {'xRef':-3168,'yRef':1762,'scale':4.0},
}
cfg = MAPS.get(map_name, MAPS['de_mirage'])

frames_json = json.dumps(frames, separators=(',',':'))
cfg_json = json.dumps(cfg)
max_frame = len(frames) - 1

html = f"""<!DOCTYPE html>
<title>Match Replay — {map_name}</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#111;overflow:hidden}}
canvas{{display:block;width:100vw;height:100vh}}
#tick{{position:fixed;top:8px;left:8px;color:#666;font:9px monospace;z-index:10}}
#bar{{position:fixed;bottom:8px;left:8px;right:8px;z-index:10;display:flex;gap:6px;align-items:center}}
#bar button{{background:#222;border:1px solid #333;color:#888;font:9px monospace;padding:3px 8px;cursor:pointer}}
#bar input{{flex:1;height:2px;-webkit-appearance:none;background:#333}}
#bar input::-webkit-slider-thumb{{-webkit-appearance:none;width:8px;height:8px;background:#0f0}}
#bar .s{{color:#444;font-size:8px}}
</style>
<div id="tick">0</div>
<canvas id="c"></canvas>
<div id="bar">
<button onclick="P=!P">⏯</button>
<input type="range" id="sc" min="0" max="{max_frame}" value="0" oninput="f=+this.value;if(!P)D(f)">
<span class="s" id="sp">1x</span>
<button onclick="sp=Math.max(1,sp-1);document.getElementById('sp').textContent=sp+'x'">−</button>
<button onclick="sp=Math.min(16,sp+1);document.getElementById('sp').textContent=sp+'x'">+</button>
</div>
<script>
const F={frames_json};
const C={cfg_json};
const c=document.getElementById('c'),x=c.getContext('2d'),SC=document.getElementById('sc');
let f=0,P=true,sp=1,fc=0;
const T={{}};
const I=new Image();
I.src="data:image/png;base64,{radar_b64}";

function w2r(a,b){{return[(a-C.xRef)/C.scale,(C.yRef-b)/C.scale]}}

function R(){{c.width=c.clientWidth*devicePixelRatio;c.height=c.clientHeight*devicePixelRatio;x.scale(devicePixelRatio,devicePixelRatio)}}
R();onresize=R;

function D(i){{
  const W=c.clientWidth,H=c.clientHeight;
  x.clearRect(0,0,W,H);
  const s=Math.min(W,H)/1024;
  const ox=(W-1024*s)/2,oy=(H-1024*s)/2;

  x.globalAlpha=0.85;
  if(I.complete)x.drawImage(I,ox,oy,1024*s,1024*s);
  x.globalAlpha=1;

  document.getElementById('tick').textContent=i+'/'+F.length;
  SC.value=i;

  const fr=F[i];if(!fr)return;
  const TC={{2:'#4a9eff',3:'#ff6b4a'}};

  fr.forEach(p=>{{
    const[rx,ry]=w2r(p.x,p.y);
    const sx=ox+rx*s,sy=oy+ry*s;
    const col=TC[p.team]||'#888';

    // Trail
    if(!T[p.id])T[p.id]=[];
    T[p.id].push([sx,sy]);
    if(T[p.id].length>25)T[p.id].shift();
    for(let t=1;t<T[p.id].length;t++){{
      x.globalAlpha=t/T[p.id].length*0.3;
      x.strokeStyle=col;x.lineWidth=1.5;
      x.beginPath();x.moveTo(T[p.id][t-1][0],T[p.id][t-1][1]);
      x.lineTo(T[p.id][t][0],T[p.id][t][1]);x.stroke();
    }}
    x.globalAlpha=1;

    // FOV cone
    const yr=-p.yaw*Math.PI/180;
    x.fillStyle=col+'18';
    x.beginPath();x.moveTo(sx,sy);
    x.arc(sx,sy,35,yr-0.9,yr+0.9);x.closePath();x.fill();

    // Aim line
    x.strokeStyle=col+'66';x.lineWidth=1;
    x.beginPath();x.moveTo(sx,sy);
    x.lineTo(sx+Math.cos(yr)*25,sy+Math.sin(yr)*25);x.stroke();

    // Player dot
    x.fillStyle=col;x.shadowColor=col;x.shadowBlur=6;
    x.beginPath();x.arc(sx,sy,4,0,Math.PI*2);x.fill();
    x.shadowBlur=0;

    // Fire flash
    if(p.fire){{x.fillStyle='#ff3';x.beginPath();x.arc(sx,sy,6,0,Math.PI*2);x.fill()}}

    // HP label
    x.fillStyle='#555';x.font='7px monospace';
    x.fillText(p.hp,sx+6,sy-6);
  }});
}}

function A(){{requestAnimationFrame(A);fc++;
  if(P&&fc%6===0){{D(f);f=(f+sp)%F.length}}}}
I.onload=()=>A();
document.onkeydown=e=>{{if(e.key===' '){{P=!P;e.preventDefault()}}}};
</script>"""

Path('replay_standalone.html').write_text(html)
print(f'Built: {len(frames)} frames, map={map_name}, {Path("replay_standalone.html").stat().st_size/1e6:.1f}MB')
