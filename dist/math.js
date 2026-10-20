export const sigmoid=x=>1/(1+Math.exp(-x));
export const activate=(x,kind='ReLU')=>kind==='Sigmoid'?sigmoid(x):kind==='Tanh'?Math.tanh(x):Math.max(0,x);
export const softmax=a=>{const m=Math.max(...a),e=a.map(x=>Math.exp(x-m)),s=e.reduce((a,b)=>a+b,0);return e.map(x=>x/s)};
export const mean=a=>a.reduce((s,x)=>s+x,0)/a.length;
export function normalize(a){const m=mean(a),v=mean(a.map(x=>(x-m)**2));return a.map(x=>(x-m)/Math.sqrt(v+1e-5))}
export const conv=(a,k,row,col)=>k.reduce((s,r,i)=>s+r.reduce((s,v,j)=>s+v*a[row+i][col+j],0),0);
export const pool=(a,mode='Max')=>mode==='Max'?Math.max(...a):mean(a);
export const loss=(p,y,mode='MSE')=>mode==='MAE'?Math.abs(p-y):(p-y)**2;
export function chain(w,x=2,target=3){const h=w*x,y=h*.5,L=(y-target)**2,g=2*(y-target)*.5*x;return {h,y,L,g}}
export function optimize(kind,lr,steps=24){let w=3,v=0,m=0,s=0;const path=[w];for(let i=1;i<=steps;i++){const g=2*w;if(kind==='Momentum'){v=.8*v+g;w-=lr*v}else if(kind==='Adam'){m=.9*m+.1*g;s=.999*s+.001*g*g;w-=lr*(m/(1-.9**i))/(Math.sqrt(s/(1-.999**i))+1e-8)}else w-=lr*g;path.push(w)}return path}
export const trainData=Array.from({length:12},(_,i)=>{const x=i/11;return [x,2*x+1+.08*Math.sin(i*2)]});
export function trainingStep(state,lr){let gw=0,gb=0;for(const [x,y] of trainData){const e=state.w*x+state.b-y;gw+=2*e*x/trainData.length;gb+=2*e/trainData.length}state.w-=lr*gw;state.b-=lr*gb;state.step++;state.loss=mean(trainData.map(([x,y])=>(state.w*x+state.b-y)**2));state.history.push(state.loss);return state}
export function polynomialFit(degree,lambda=0){const xs=Array.from({length:10},(_,i)=>-1+2*i/9),ys=xs.map((x,i)=>Math.sin(2.4*x)+.24*Math.sin(i*2.3+1));const n=degree+1,A=Array.from({length:n},(_,i)=>Array.from({length:n+1},(_,j)=>j===n?xs.reduce((s,x,k)=>s+x**i*ys[k],0):xs.reduce((s,x)=>s+x**(i+j),0)+(i===j&&i>0?lambda:0)));for(let i=0;i<n;i++){let p=i;for(let k=i+1;k<n;k++)if(Math.abs(A[k][i])>Math.abs(A[p][i]))p=k;[A[i],A[p]]=[A[p],A[i]];const d=A[i][i]||1e-12;for(let j=i;j<=n;j++)A[i][j]/=d;for(let k=0;k<n;k++)if(k!==i){const f=A[k][i];for(let j=i;j<=n;j++)A[k][j]-=f*A[i][j]}}const coef=A.map(r=>r[n]),predict=x=>coef.reduce((s,c,i)=>s+c*x**i,0);return {xs,ys,predict,train:mean(xs.map((x,i)=>(predict(x)-ys[i])**2)),validation:mean(Array.from({length:80},(_,i)=>{const x=-1+2*(i+.5)/80;return (predict(x)-Math.sin(2.4*x))**2}))}}
