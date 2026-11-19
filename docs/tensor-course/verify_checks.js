// Runs the page's own "check it" code in Node (DOM-free part) on every cell of every real tensor.
const fs=require('fs'),vm=require('vm'),path=require('path');
const html=fs.readFileSync(path.join(__dirname,'index.html'),'utf8');
const re=/<script(?![^>]*src)[^>]*>([\s\S]*?)<\/script>/g;let m;const ctx=vm.createContext({console,Math,Number,Object,Array,String,JSON,Error});
while((m=re.exec(html)))vm.runInContext(m[1],ctx);
const C=ctx.__course;let bad=0;
const walk=(T,f)=>{const n=T.shape.length,idx=new Array(n).fill(0);const total=T.size||1;for(let c=0;c<total;c++){let r=c;const o={};for(let j=n-1;j>=0;j--){o[T.axes[j]]=r%T.shape[j];r=Math.floor(r/T.shape[j])}f(o)}};
const rows=[];
for(const k of Object.keys(C.R)){
  if(!C.hasChk(k))continue;const T=C.R[k];let mx=0,cnt=0;
  walk(T,o=>{const r=C.CHK[k](o);const d=Math.abs(r.computed-r.stored);if(!(d<1e-4)){bad++;console.log('MISMATCH',k,JSON.stringify(o),r.computed,r.stored)}mx=Math.max(mx,d);cnt++});
  rows.push([k,cnt,mx]);
}
rows.forEach(r=>console.log(r[0].padEnd(12),String(r[1]).padStart(5),'cells  max diff',r[2].toExponential(2)));
// graph: every node with a check function, every parent key must exist
for(const n of C.GN){for(const p of n.par)if(!C.R[p.id])console.log('bad parent',n.id,p.id);for(const w of (n.w||[]))if(!C.R[w])console.log('bad weight',n.id,w)}
// toy checks
let toyBad=0;for(const [name,tab] of Object.entries(C.toyChecks)){if(typeof tab!=='object')continue;for(const k of Object.keys(tab)){const T=C.TOYS[{P:'P',S:'Sh',S2:'St',dot:'dd',Y:'Y',X3:'X3',Xh:'Xh'}[k]];walk(T,o=>{const r=tab[k](o,T);if(Math.abs(r.computed-r.stored)>1e-9){toyBad++;console.log('TOY MISMATCH',k,o)}})}}
// the three named cells
const three=[['q_lin',{t:2,nd:21}],['weights',{n:2,t:3,s:1}],['res2',{t:3,h:4}]];
for(const [k,o] of three){const r=C.CHK[k](o);console.log('CELL',k,JSON.stringify(o),'recomputed',r.computed,'stored',r.stored,'diff',Math.abs(r.computed-r.stored).toExponential(2))}
const r2=C.CHK.act_up({t:3,i:7});console.log('CELL act_up {t:3,i:7} recomputed',r2.computed,'stored',r2.stored);
console.log(bad||toyBad?'FAIL':'ALL CHECKS PASS','real mismatches',bad,'toy mismatches',toyBad);
process.exit(bad||toyBad?1:0);
