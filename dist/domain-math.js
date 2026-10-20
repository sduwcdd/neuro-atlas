import {softmax} from './math.js';
export const ctcCollapse=path=>path.split(' ').filter((v,i,a)=>i===0||v!==a[i-1]).filter(v=>v!=='_').join('');
export const ppoObjective=(ratio,adv,epsilon)=>Math.min(ratio*adv,Math.max(1-epsilon,Math.min(1+epsilon,ratio))*adv);
export function gcnAggregate(layers){const A=Array.from({length:4},(_,i)=>Array.from({length:4},(_,j)=>i===j||Math.abs(i-j)===1?1:0)),degree=A.map(r=>r.reduce((a,b)=>a+b)),N=A.map((r,i)=>r.map((v,j)=>v/Math.sqrt(degree[i]*degree[j])));let h=[1,0,2,1];for(let l=0;l<layers;l++)h=N.map(r=>r.reduce((s,v,j)=>s+v*h[j],0));return h}
export const receptiveField=(layers,base)=>1+2*Array.from({length:layers},(_,i)=>base**i).reduce((a,b)=>a+b,0);
export function attentionExample(query,causal){return softmax(Array.from({length:5},(_,i)=>causal&&i>query?-Infinity:Math.cos((i-query)*.8)))}
export function ganLoss(mu,threshold){const z=[-1,-.5,0,.5,1],sig=x=>1/(1+Math.exp(-2*(x-threshold))),real=z.map(v=>2+.5*v),fake=z.map(v=>mu+.5*v);return {real,fake,d:-real.reduce((s,x)=>s+Math.log(sig(x)),0)/5-fake.reduce((s,x)=>s+Math.log(1-sig(x)),0)/5,g:-fake.reduce((s,x)=>s+Math.log(sig(x)),0)/5}}
