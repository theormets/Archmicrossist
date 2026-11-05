import {decodeRGB,padRGB,tensorFromRGB} from './preprocess.js';
import {binary,json,sha,normalize,clsVector} from './runtime-core.js?v=archmicrossist1';
const $=id=>document.getElementById(id),say=s=>{$('status').textContent=s;};
const MODEL='./models/archmicrossist_dinov2_small_f2ca3b39879c.onnx';
const MODEL_SHA='f2ca3b39879cac317d017decfd297cf6619ddc9474a41893219ddb80f9a12679';
let dataPromise,encoderPromise,file,previewUrl,exportRows=[];
function node(tag,className,text){const n=document.createElement(tag);if(className)n.className=className;if(text!==undefined)n.textContent=text;return n;}
function label(r,k,fallback='Not reported'){const v=String(r[k]??'').trim();return v||fallback;}
function isRetracted(r){return ['true','1','1.0'].includes(String(r.retracted).toLowerCase());}
async function loadData(){
  if(dataPromise)return dataPromise;
  dataPromise=(async()=>{
    const config=await json('data/library.json'),metaBytes=await binary(config.metadata_file+'?sha256='+encodeURIComponent(config.metadata_sha256),'Reference metadata');
    if(await sha(metaBytes)!==config.metadata_sha256)throw new Error('Reference metadata checksum mismatch.');
    const rows=JSON.parse(new TextDecoder().decode(metaBytes)),buffer=await binary(config.embeddings_file,'Reference embeddings');
    if(await sha(buffer)!==config.embeddings_sha256)throw new Error('Reference embedding checksum mismatch.');
    const vectors=new Float32Array(buffer);
    if(rows.length!==config.shape[0]||vectors.length!==rows.length*384)throw new Error('Reference alignment mismatch.');
    return {config,rows,vectors};
  })().catch(e=>{dataPromise=null;throw e;});return dataPromise;
}
async function encoder(){
  if(encoderPromise)return encoderPromise;
  encoderPromise=(async()=>{
    say('Loading visual encoder. First search may take longer…');
    const ort=await import('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.wasm.min.mjs');
    ort.env.wasm.numThreads=1;ort.env.wasm.proxy=false;ort.env.wasm.wasmPaths='https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
    const bytes=await binary(MODEL,'Visual encoder',true,MODEL_SHA);if(await sha(bytes)!==MODEL_SHA)throw new Error('Wrong model file; retain the previously verified model.');
    const session=await ort.InferenceSession.create(bytes,{executionProviders:['wasm'],graphOptimizationLevel:'all'});return {ort,session};
  })().catch(e=>{encoderPromise=null;throw e;});return encoderPromise;
}
export function rankReferences(vector,rows,vectors,options){
  const ranked=[];
  for(let i=0;i<rows.length;i++){
    const r=rows[i],approved=r.status==='Approved',hold=r.status==='Hold';
    if(isRetracted(r)||(!approved&&!(options.hold&&hold)))continue;
    if(options.mode!=='Both'&&r.microscopy!==options.mode)continue;
    if(options.material&&options.material!=='All'&&r.material!==options.material)continue;
    if(options.exclude&&r.rgb_sha256===options.rgbHash&&r.width===options.width&&r.height===options.height)continue;
    let score=0;for(let j=0;j<384;j++)score+=vector[j]*vectors[i*384+j];ranked.push({r,score,index:i});
  }
  ranked.sort((a,b)=>b.score-a.score||a.index-b.index);
  const matches=[],seen=new Set();
  for(const item of ranked){const group=item.r.sample_key||`unlinked:${item.r.paper_key}`;if(seen.has(group))continue;seen.add(group);matches.push(item);if(matches.length===options.k)break;}
  // Select top-k by cosine first; only presentation order prioritizes confirmed source links.
  matches.sort((a,b)=>Number(a.r.status==='Hold')-Number(b.r.status==='Hold')||b.score-a.score||a.index-b.index);
  return {matches,candidates:ranked.length};
}
function card(item,rank){
  const r=item.r,hold=r.status==='Hold',article=node('article','card'),image=node('div','image-wrap'),img=node('img');
  img.src=r.image;img.alt='Published reference micrograph';img.loading='lazy';img.addEventListener('error',()=>{img.alt='Reference image not uploaded yet';});
  image.append(img,node('span','rank',`MATCH ${String(rank).padStart(2,'0')}`));
  const body=node('div','body'),top=node('div','card-top'),score=node('span','score','Cosine');
  score.append(node('b',null,Math.max(-1,Math.min(1,item.score)).toFixed(3)));
  top.append(node('span','badge'+(hold?' hold':''),hold?'Unverified · Hold':'Source-approved'),score);
  body.append(top,node('h3',null,hold?'Unverified reference':label(r,'material','Material not linked')),
    node('p','paper-title',label(r,'title')),node('p','authors',label(r,'authors').slice(0,145)+(label(r,'authors').length>145?'…':'')),node('div','meta',`${label(r,'year')} · Reference microscopy: ${r.microscopy} · ${r.paper_key}`));
  const source=r.source_record||{},details=node('div','record-sections');
  function section(title,fields,open=false){
    const panel=node('details'),dl=node('dl');panel.open=open;panel.append(node('summary',null,title));
    for(const [name,key] of fields)dl.append(node('dt',null,name),node('dd',null,label({...r,...source},key)));
    panel.append(dl);details.append(panel);
  }
  section('01 · Micrograph details',[
    ['Image ID','image_key'],['Microscopy','micrograph__microscopy_image_mode'],['Figure / panel','figure'],['PDF page','micrograph__pdf_page'],['Sample region','micrograph__sample_region'],['Printed scale','micrograph__scale_as_printed'],['Reported constituents','micrograph__reported_constituents'],['Reported morphology','micrograph__reported_morphology'],['Microstructure description','micrograph__microstructure_description'],['Authors’ interpretation','micrograph__author_interpretation'],['Preparation / etchant','preparation'],['Figure caption','caption']],true);
  section('02 · Sample & archaeological context',[
    ['Linked sample record','sample_key'],['Published sample label','sample__published_sample_label'],['Reported source material','source_material'],['Reported composition','sample__composition_as_reported'],['Artefact / object','sample__artefact_object_type'],['Findspot / site','sample__findspot_site'],['Region','sample__region_state'],['Country','sample__country'],['Period / culture','sample__period_culture'],['Dating as reported','sample__dating_as_reported'],['Dating caveats','sample__dating_uncertainty_caveats'],['Manufacturing origin','sample__manufacturing_origin'],['Sample notes','sample__sample_notes']]);
  section('03 · Source paper',[
    ['Title','title'],['Authors','authors'],['Year','year'],['Journal','paper__journal_source'],['Volume / issue / pages','paper__volume_issue_and_pages'],['DOI','doi'],['Paper ID','paper_key'],['Evidence locator','micrograph__text_evidence_locator']]);
  section('04 · Review & traceability',[
    ['Source review status','status'],['Hold reason','hold_reason'],['Image quality','micrograph__image_quality'],['Extraction notes','micrograph__extraction_notes'],['Sample link resolved','sample_link_resolved'],['Scope assessment','scope_track_v3'],['Context assessment','context_label_v3'],['Context evidence level','context_evidence_level_v3']]);
  const raw=node('details'),rawList=node('dl');raw.append(node('summary',null,'Complete original master record'));
  raw.append(node('p',null,'Source metadata and audit fields, shown as recorded. Empty fields are not reported. These are reference-image details, not measurements of your uploaded sample.'));
  for(const [key,value] of Object.entries(source))rawList.append(node('dt',null,key),node('dd',null,String(value).trim()||'Not reported'));
  raw.append(rawList);details.append(raw);
  let doi=label(r,'doi','').replace(/^https?:\/\/(dx\.)?doi\.org\//i,'').replace(/^doi:\s*/i,'').trim();
  details.append(node('p',null,'Display preview — consult the source paper for the original image.'));
  if(doi.startsWith('10.')){const a=node('a',null,'Open source paper ↗');a.href='https://doi.org/'+encodeURIComponent(doi).replace(/%2F/gi,'/');a.target='_blank';a.rel='noopener noreferrer';details.append(a);}
  body.append(details);article.append(image,body);return article;
}
function choose(selected){
  if(!selected)return;if(!/\.(png|jpe?g)$/i.test(selected.name)){say('Please use a PNG, JPG or JPEG image.');return;}
  if(selected.size>64*1024*1024){say('Please use an image under 64 MB.');return false;}
  selectTab('file');file=selected;if(previewUrl)URL.revokeObjectURL(previewUrl);previewUrl=URL.createObjectURL(file);
  $('query-preview').src=previewUrl;$('query-preview').hidden=false;$('upload-prompt').hidden=true;$('file-name').textContent=file.name;say('Image selected. Ready to trace similar micrographs.');
  // Prevent stale results being mistaken for the newly selected query.
  $('results').hidden=true;$('cards').replaceChildren();$('empty').hidden=false;$('download').hidden=true;$('score-note').hidden=true;$('result-title').textContent='Find your next connection.';$('result-note').textContent='Published images. Traceable sources.';exportRows=[];return true;
}
$('file').addEventListener('change',()=>choose($('file').files[0]));
for(const event of ['dragenter','dragover'])$('drop').addEventListener(event,e=>{e.preventDefault();$('drop').classList.add('drag');});
for(const event of ['dragleave','drop'])$('drop').addEventListener(event,e=>{e.preventDefault();$('drop').classList.remove('drag');});
$('drop').addEventListener('drop',e=>choose(e.dataTransfer.files[0]));
$('search').addEventListener('click',async()=>{
  if(!file){say('Choose a PNG, JPG or JPEG micrograph first.');return;}
  const selected=file,options={mode:'Both',hold:true,exclude:false,k:Number($('topk').value),material:'All'};
  $('search').disabled=true;$('file').disabled=true;
  try{
    say('Preparing your image…');const d=await loadData(),image=await decodeRGB(selected);
    options.rgbHash=await sha(image.rgb.buffer);options.width=image.width;options.height=image.height;
    const padded=padRGB(image.rgb,image.width,image.height),values=tensorFromRGB(padded.rgb,d.config.processor);
    const {ort,session}=await encoder();say('Reading microstructure patterns…');await new Promise(r=>setTimeout(r,30));
    const input=new ort.Tensor('float32',values,[1,3,224,224]);const output=await session.run({pixel_values:input});const vector=clsVector(output,384).vector;
    for(const tensor of Object.values(output))tensor.dispose?.();input.dispose?.();
    const {matches,candidates}=rankReferences(vector,d.rows,d.vectors,options);
    if(file!==selected){say('A new image was selected. Search again for the current image.');return;}
    $('results').hidden=false;$('cards').replaceChildren(...matches.map((item,i)=>card(item,i+1)));
    $('empty').hidden=matches.length>0;$('result-title').textContent=matches.length?'Closest references':'No references in this selection.';
    $('result-note').textContent=`${candidates.toLocaleString()} reference images searched · Approved references shown first; Hold references last`;
    $('score-note').hidden=!matches.length;$('download').hidden=!matches.length;
    exportRows=matches.map((item,i)=>({rank:i+1,cosine_similarity:item.score,...item.r,source_record:JSON.stringify(item.r.source_record||{})}));say(`${matches.length} matches ready. Follow the source to interpret them.`);
  }catch(e){say('Search stopped: '+e.message);console.error(e);}
  finally{$('search').disabled=false;$('file').disabled=false;}
});
$('download').addEventListener('click',()=>{
  if(!exportRows.length)return;const fields=Object.keys(exportRows[0]);
  function cell(v){let s=String(v??'');if(/^[=+\-@]/.test(s))s="'"+s;return '"'+s.replace(/"/g,'""')+'"';}
  const csv='\uFEFF'+[fields.map(cell).join(','),...exportRows.map(r=>fields.map(k=>cell(r[k])).join(','))].join('\r\n');
  const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'})),a=document.createElement('a');a.href=url;a.download='Archmicrossist_matches.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
function selectTab(tab){
  const url=tab==='url';$('file-panel').hidden=url;$('url-panel').hidden=!url;
  for(const id of ['tab-file','tab-url']){const active=(id==='tab-url')===url;$(id).classList.toggle('active',active);$(id).setAttribute('aria-selected',String(active));}
}
$('tab-file').addEventListener('click',()=>selectTab('file'));
$('tab-url').addEventListener('click',()=>{selectTab('url');$('image-url').focus();});
$('reset').addEventListener('click',()=>{
  if($('search').disabled||$('load-url').disabled){say('Please wait for the current operation to finish before resetting.');return;}
  file=undefined;exportRows=[];
  if(previewUrl){URL.revokeObjectURL(previewUrl);previewUrl=undefined;}
  $('file').value='';$('image-url').value='';$('query-preview').removeAttribute('src');$('query-preview').hidden=true;
  $('upload-prompt').hidden=false;$('file-name').textContent='PNG / JPG / JPEG · up to 64 MB';
  $('topk').value='5';
  $('results').hidden=true;$('cards').replaceChildren();$('empty').hidden=true;$('download').hidden=true;$('score-note').hidden=true;
  $('result-title').textContent='Closest references';$('result-note').textContent='';
  document.querySelector('.refine').open=false;selectTab('file');say('Reset complete. Upload a new micrograph.');
});

$('image-url').addEventListener('keydown',e=>{if(e.key==='Enter'){e.preventDefault();$('load-url').click();}});
$('load-url').addEventListener('click',async()=>{
  if($('search').disabled)return;
  $('load-url').disabled=true;
  try{
    const url=new URL($('image-url').value);
    if(url.protocol!=='https:'||url.username||url.password)throw new Error('Use a public HTTPS image URL without embedded credentials.');
    say('Loading linked image…');
    const response=await fetch(url.href,{mode:'cors',credentials:'omit',referrerPolicy:'no-referrer'});
    if(!response.ok)throw new Error('Image server returned HTTP '+response.status);
    if(Number(response.headers.get('Content-Length'))>64*1024*1024)throw new Error('Image exceeds 64 MB.');
    const reader=response.body.getReader(),chunks=[];let bytes=0;
    try{while(true){const next=await reader.read();if(next.done)break;bytes+=next.value.length;if(bytes>64*1024*1024){await reader.cancel();throw new Error('Image exceeds 64 MB.');}chunks.push(next.value);}}finally{reader.releaseLock();}
    const blob=new Blob(chunks),signature=new Uint8Array(await blob.slice(0,8).arrayBuffer());
    const png=[137,80,78,71,13,10,26,10].every((v,i)=>signature[i]===v);
    const jpeg=signature[0]===255&&signature[1]===216&&signature[2]===255;
    if(!png&&!jpeg)throw new Error('Use a PNG, JPG or JPEG image.');
    choose(new File([blob],png?'linked-micrograph.png':'linked-micrograph.jpg',{type:png?'image/png':'image/jpeg'}));
  }catch(error){say('Could not load URL: '+error.message+' If browser access is blocked, download the image and use Upload image.');}
  finally{$('load-url').disabled=false;}
});

loadData().catch(e=>say('Library setup: '+e.message));
