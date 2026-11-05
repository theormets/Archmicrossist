// Model conversion check only. Fresh-upload preprocessing is a separate gate.
const ORT_VERSION = '1.22.0';
const MODEL_REVISION = 'c2bb04a';
const MODEL_URL = `https://huggingface.co/Xenova/dinov2-small/resolve/${MODEL_REVISION}/onnx/model.onnx`;
const MIN_COSINE = 0.9995;
const $ = id => document.getElementById(id);
const log = text => { $('log').textContent += `${text}\n`; };
const status = text => { $('status').textContent = text; };
let report;

async function binary(url, label, stream=false, expectedSha='') {
  let modelCache;
  if(expectedSha && globalThis.caches){
    try{
      modelCache=await caches.open('archmicrossist-verified-models-v1');
      const saved=await modelCache.match(url);
      if(saved){
        const bytes=await saved.arrayBuffer();
        if(await sha(bytes)===expectedSha){status('Visual encoder: using saved model on this device…');return bytes;}
        await modelCache.delete(url);
      }
    }catch(error){console.info('Local model storage unavailable; using normal loading.',error);}
  }
  const res=await fetch(url,{cache:'force-cache'});
  if(!res.ok) throw new Error(`${label}: HTTP ${res.status}. Check that the file/folder was uploaded.`);
  if(!stream || !res.body) return res.arrayBuffer();
  const reader=res.body.getReader(), chunks=[]; let size=0;
  while(true){const {done,value}=await reader.read();if(done)break;chunks.push(value);size+=value.length;status(`${label}: ${(size/1e6).toFixed(1)} MB loaded${expectedSha?' · preparing local copy':''}…`);}
  const bytes=new Uint8Array(size);let offset=0;for(const c of chunks){bytes.set(c,offset);offset+=c.length;}
  if(expectedSha){
    if(await sha(bytes.buffer)!==expectedSha)throw new Error('Model checksum mismatch; local copy was not saved.');
    if(modelCache){try{await modelCache.put(url,new Response(bytes,{headers:{'Content-Type':'application/octet-stream'}}));}catch(error){console.info('Could not save model locally.',error);}}
  }
  return bytes.buffer;
}
async function json(url){const r=await fetch(url,{cache:'no-store'});if(!r.ok)throw new Error(`Missing ${url} (HTTP ${r.status})`);return r.json();}
async function sha(buffer){return [...new Uint8Array(await crypto.subtle.digest('SHA-256',buffer))].map(v=>v.toString(16).padStart(2,'0')).join('');}
function normalize(values){let sq=0;for(const v of values){if(!Number.isFinite(v))throw new Error('Non-finite embedding.');sq+=v*v;}if(sq<=0)throw new Error('Zero embedding.');const n=Math.sqrt(sq);return Float32Array.from(values,v=>v/n);}
function dot(a,b){let result=0;for(let i=0;i<a.length;i++)result+=a[i]*b[i];return result;}
function isRetracted(row){return ['true','1','1.0'].includes(String(row.source_retraction_flag).toLowerCase());}
function nearest(query,test,metadata,library,dimension){
  const ranked=[];
  for(let i=0;i<metadata.length;i++){
    const r=metadata[i];if(isRetracted(r)||String(r.source_status).toLowerCase()!=='approved'||r.microscopy_major!==test.microscopy)continue;
    let score=0;for(let j=0;j<dimension;j++)score+=query[j]*library[i*dimension+j];
    ranked.push({row:i,image_key:r.image_key,paper_key:r.paper_key,sample_key:r.sample_key,score});
  }
  ranked.sort((a,b)=>b.score-a.score||a.row-b.row);
  const result=[],seen=new Set();
  for(const r of ranked){const key=String(r.sample_key||'').trim()||`unlinked:${r.paper_key}`;if(seen.has(key))continue;seen.add(key);result.push(r);if(result.length===5)break;}
  return result;
}
function clsVector(outputs,dimension){
  let name=Object.keys(outputs).find(k=>k==='last_hidden_state');
  if(!name)name=Object.keys(outputs).find(k=>outputs[k].dims.length===3&&outputs[k].dims[2]===dimension);
  if(!name)throw new Error(`No CLS-compatible last-hidden-state output: ${Object.keys(outputs).join(', ')}`);
  const output=outputs[name];
  if(output.dims.length!==3||output.dims[0]!==1||output.dims[2]!==dimension)throw new Error(`Unexpected output shape: ${output.dims}`);
  return {name,shape:[...output.dims],vector:normalize(output.data.slice(0,dimension))};
}
function showRow(test,cosine,overlap,passed){
  const row=document.createElement('tr');
  for(const value of [test.test_id,test.microscopy,cosine.toFixed(6),`${overlap}/5`,passed?'MODEL PASS':'CHECK REQUIRED']){
    const cell=document.createElement('td');cell.textContent=value;row.appendChild(cell);
  }
  row.lastChild.className=passed?'pass':'fail';$('rows').appendChild(row);
}

export {binary,json,sha,normalize,dot,nearest,clsVector};
