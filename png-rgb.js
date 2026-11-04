// Exact 8-bit PNG sample decoding, without canvas colour/premultiplication.
// PNG filtering follows the public PNG format specification (ISO/IEC 15948).
// Interlaced/16-bit PNG and transparency require a separate validated policy.
function orientation(bytes){
  if(!bytes)return 1;
  const d=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength);
  if(bytes.length<8)throw new Error('Invalid PNG EXIF metadata.');
  const little=bytes[0]===73&&bytes[1]===73;
  if(!little&&!(bytes[0]===77&&bytes[1]===77))throw new Error('Unsupported EXIF byte order.');
  if(d.getUint16(2,little)!==42)throw new Error('Unsupported EXIF header.');
  const start=d.getUint32(4,little);
  if(start+2>bytes.length)throw new Error('Invalid EXIF directory.');
  const count=d.getUint16(start,little);
  for(let i=0;i<count;i++){
    const p=start+2+i*12;if(p+12>bytes.length)throw new Error('Truncated EXIF directory.');
    if(d.getUint16(p,little)===274){
      if(d.getUint16(p+2,little)!==3||d.getUint32(p+4,little)!==1)throw new Error('Unsupported EXIF orientation encoding.');
      const value=d.getUint16(p+8,little);if(value<1||value>8)throw new Error('Invalid EXIF orientation.');return value;
    }
  }
  return 1;
}
function orient(rgb,width,height,value){
  if(value===1)return {rgb,width,height};
  const w=value>=5?height:width,h=value>=5?width:height,out=new Uint8Array(rgb.length);
  for(let y=0;y<h;y++)for(let x=0;x<w;x++){
    let a=x,b=y;
    switch(value){
      case 2:a=width-1-x;break;case 3:a=width-1-x;b=height-1-y;break;
      case 4:b=height-1-y;break;case 5:a=y;b=x;break;
      case 6:a=y;b=height-1-x;break;case 7:a=width-1-y;b=height-1-x;break;
      case 8:a=width-1-y;b=x;break;
    }
    out.set(rgb.subarray((b*width+a)*3,(b*width+a)*3+3),(y*w+x)*3);
  }
  return {rgb:out,width:w,height:h};
}
function predictor(a,b,c){const p=a+b-c,x=Math.abs(p-a),y=Math.abs(p-b),z=Math.abs(p-c);return x<=y&&x<=z?a:y<=z?b:c;}
export async function decodePNG8(file){
  if(file.size>64*1024*1024)throw new Error('PNG exceeds the 64 MB upload limit.');
  const bytes=new Uint8Array(await file.arrayBuffer()),view=new DataView(bytes.buffer);
  if(![137,80,78,71,13,10,26,10].every((v,i)=>bytes[i]===v))throw new Error('Not a PNG image.');
  let width,height,channels,type,palette,alpha,exif;const parts=[];
  for(let p=8;p+12<=bytes.length;){
    const length=view.getUint32(p),name=String.fromCharCode(...bytes.subarray(p+4,p+8));
    if(p+12+length>bytes.length)throw new Error('Truncated PNG chunk.');
    const data=bytes.subarray(p+8,p+8+length);
    if(name==='IHDR'){
      if(length!==13||width!==undefined)throw new Error('Invalid PNG header.');
      width=view.getUint32(p+8);height=view.getUint32(p+12);type=data[9];
      if(!width||!height||width*height>32_000_000)throw new Error('Invalid size or PNG exceeds the 32-megapixel limit.');
      if(data[8]!==8||data[12]!==0)throw new Error('This check supports 8-bit, non-interlaced PNG. Convert other PNG types to 8-bit RGB first.');
      if(data[10]!==0||data[11]!==0)throw new Error('Unsupported PNG compression/filter method.');
      channels=({0:1,2:3,3:1,4:2,6:4})[type];if(!channels)throw new Error('Unsupported PNG colour type.');
    }else if(name==='IDAT')parts.push(data);
    else if(name==='PLTE')palette=data;
    else if(name==='tRNS')alpha=data;
    else if(name==='eXIf')exif=data;
    else if(name==='IEND')break;
    p+=length+12;
  }
  if(!width||!parts.length)throw new Error('PNG header or image data missing.');
  if(type===3&&(!palette||palette.length%3))throw new Error('Invalid PNG palette.');
  const stride=width*channels,size=(stride+1)*height,decoded=new Uint8Array(size);
  const stream=new Blob(parts).stream().pipeThrough(new DecompressionStream('deflate')),reader=stream.getReader();let used=0;
  try{
    while(true){const {done,value}=await reader.read();if(done)break;if(used+value.length>size){await reader.cancel();throw new Error('PNG decoded-data length exceeds its dimensions.');}decoded.set(value,used);used+=value.length;}
  }finally{reader.releaseLock();}
  if(used!==size)throw new Error('PNG decoded-data length mismatch.');
  const samples=new Uint8Array(height*stride);
  for(let y=0;y<height;y++){
    const offset=y*(stride+1),filter=decoded[offset];if(filter>4)throw new Error('Invalid PNG scanline filter.');
    for(let x=0;x<stride;x++){
      const pos=y*stride+x,left=x>=channels?samples[pos-channels]:0,up=y?samples[pos-stride]:0,corner=y&&x>=channels?samples[pos-stride-channels]:0;
      const adjustment=filter===0?0:filter===1?left:filter===2?up:filter===3?Math.floor((left+up)/2):predictor(left,up,corner);
      samples[pos]=decoded[offset+1+x]+adjustment;
    }
  }
  const rgb=new Uint8Array(width*height*3);
  for(let i=0;i<width*height;i++){
    const p=i*channels;let r,g,b,a=255;
    if(type===0||type===4){r=g=b=samples[p];if(type===4)a=samples[p+1];else if(alpha&&alpha.length>=2&&r===alpha[0]*256+alpha[1])a=0;}
    else if(type===2||type===6){r=samples[p];g=samples[p+1];b=samples[p+2];if(type===6)a=samples[p+3];else if(alpha&&alpha.length>=6&&r===alpha[0]*256+alpha[1]&&g===alpha[2]*256+alpha[3]&&b===alpha[4]*256+alpha[5])a=0;}
    else{const index=samples[p],q=index*3;if(q+2>=palette.length)throw new Error('PNG palette index out of range.');r=palette[q];g=palette[q+1];b=palette[q+2];if(alpha&&index<alpha.length)a=alpha[index];}
    if(a!==255)throw new Error('Transparent pixels require a defined background policy; use an opaque RGB image.');
    rgb[i*3]=r;rgb[i*3+1]=g;rgb[i*3+2]=b;
  }
  const value=orientation(exif);
  return {...orient(rgb,width,height,value),format:'PNG',decoder:'PNG8 raw samples + EXIF orientation',exif_orientation:value};
}
