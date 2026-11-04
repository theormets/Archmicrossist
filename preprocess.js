// Independent RGB8 bicubic implementation for compatibility with Pillow's
// separable 8-bit BICUBIC resize and ImageOps.pad. See THIRD_PARTY_NOTES.txt.
// Browser decoding is tested separately from these pure-array operations.
import {decodePNG8} from './png-rgb.js';
const PRECISION=4194304, HALF=2097152;
export function roundEven(n){
  const lo=Math.floor(n),fraction=n-lo;
  return fraction===.5 ? lo+(lo%2) : Math.round(n);
}
function cubic(x){
  x=Math.abs(x);
  return x<1 ? (1.5*x-2.5)*x*x+1 : x<2 ? -.5*(((x-5)*x+8)*x-4) : 0;
}
function taps(sourceSize,destSize){
  const ratio=sourceSize/destSize,spread=Math.max(1,ratio),radius=2*spread,result=[];
  for(let d=0;d<destSize;d++){
    const middle=(d+.5)*ratio;
    const start=Math.max(0,Math.trunc(middle-radius+.5));
    const end=Math.min(sourceSize,Math.trunc(middle+radius+.5));
    const weights=[];let sum=0;
    for(let s=start;s<end;s++){const w=cubic((s-middle+.5)/spread);weights.push(w);sum+=w;}
    result.push({start,weights:Int32Array.from(weights,w=>{const v=sum?w/sum:w;return Math.trunc(v*PRECISION+(v<0?-.5:.5));})});
  }
  return result;
}
function byte(sum){return Math.max(0,Math.min(255,Math.floor(sum/PRECISION)));}
export function resizeRGB(input,width,height,newWidth,newHeight){
  if(input.length!==width*height*3)throw new Error('RGB array size mismatch.');
  if(![width,height,newWidth,newHeight].every(n=>Number.isInteger(n)&&n>0))throw new Error('Invalid image dimensions.');
  let horizontal=input;
  if(width!==newWidth){
    horizontal=new Uint8Array(newWidth*height*3);const table=taps(width,newWidth);
    for(let y=0;y<height;y++)for(let x=0;x<newWidth;x++){
      const t=table[x];
      for(let c=0;c<3;c++){
        let sum=HALF;
        for(let k=0;k<t.weights.length;k++)sum+=input[(y*width+t.start+k)*3+c]*t.weights[k];
        horizontal[(y*newWidth+x)*3+c]=byte(sum);
      }
    }
  }
  if(height===newHeight)return new Uint8Array(horizontal);
  const result=new Uint8Array(newWidth*newHeight*3),table=taps(height,newHeight);
  for(let y=0;y<newHeight;y++)for(let x=0;x<newWidth;x++){
    const t=table[y];
    for(let c=0;c<3;c++){
      let sum=HALF;
      for(let k=0;k<t.weights.length;k++)sum+=horizontal[((t.start+k)*newWidth+x)*3+c]*t.weights[k];
      result[(y*newWidth+x)*3+c]=byte(sum);
    }
  }
  return result;
}
export function padRGB(input,width,height){
  let w=224,h=224;
  if(width>height)h=roundEven(height/width*224);
  else if(width<height)w=roundEven(width/height*224);
  if(w<1||h<1)throw new Error('Image aspect ratio is too extreme for a useful 224-pixel input.');
  const resized=resizeRGB(input,width,height,w,h),result=new Uint8Array(224*224*3);result.fill(127);
  const left=roundEven((224-w)/2),top=roundEven((224-h)/2);
  for(let y=0;y<h;y++)result.set(resized.subarray(y*w*3,(y+1)*w*3),((y+top)*224+left)*3);
  return {rgb:result,width:224,height:224,resized_width:w,resized_height:h,pad_left:left,pad_top:top};
}
export function tensorFromRGB(rgb,config){
  if(rgb.length!==224*224*3)throw new Error('Expected a 224x224 RGB image.');
  const means=config.image_mean.map(Math.fround),stds=config.image_std.map(Math.fround);
  const scale=config.rescale_factor,plane=224*224,result=new Float32Array(plane*3);
  if(config.do_rescale!==true||config.do_normalize!==true)throw new Error('Unexpected saved processor settings.');
  for(let c=0;c<3;c++)for(let i=0;i<plane;i++){
    const rescaled=Math.fround(rgb[i*3+c]*scale);
    result[c*plane+i]=Math.fround(Math.fround(rescaled-means[c])/stds[c]);
  }
  return result;
}
export async function decodeRGB(file,{forceCanvas=false}={}){
  // Opaque RGB PNG/JPEG only for this initial upload check. Browser decoders
  // handle orientation. Other formats, transparency and 16-bit conversion
  // require separately defined and validated preprocessing policies.
  const head=new Uint8Array(await file.slice(0,26).arrayBuffer());
  const png=head.length>=26&&[137,80,78,71,13,10,26,10].every((b,i)=>head[i]===b);
  const jpeg=head.length>=3&&head[0]===255&&head[1]===216&&head[2]===255;
  if(!png&&!jpeg)throw new Error('Use an opaque PNG or JPEG. TIFF and other formats are not supported in this check.');
  if(png&&!forceCanvas)return decodePNG8(file);
  if(png&&head[24]===16)throw new Error('16-bit PNG requires a separately validated conversion; use an 8-bit RGB copy.');
  const bitmap=await createImageBitmap(file,{imageOrientation:'from-image',colorSpaceConversion:'none',premultiplyAlpha:'none'});
  try{
    if(bitmap.width*bitmap.height>32_000_000)throw new Error('Image exceeds the 32-megapixel browser test limit.');
    const canvas=document.createElement('canvas');canvas.width=bitmap.width;canvas.height=bitmap.height;
    const ctx=canvas.getContext('2d',{willReadFrequently:true,colorSpace:'srgb'});
    if(!ctx)throw new Error('Canvas decoding is unavailable.');
    ctx.drawImage(bitmap,0,0);const pixels=ctx.getImageData(0,0,canvas.width,canvas.height).data;
    const result=new Uint8Array(canvas.width*canvas.height*3);
    for(let i=0,j=0;i<pixels.length;i+=4,j+=3){
      if(pixels[i+3]!==255)throw new Error('Transparent pixels are not supported by this initial consistency test.');
      result[j]=pixels[i];result[j+1]=pixels[i+1];result[j+2]=pixels[i+2];
    }
    return {rgb:result,width:canvas.width,height:canvas.height,format:png?'PNG':'JPEG',decoder:'Browser bitmap/canvas'};
  }finally{bitmap.close();}
}
export function previewRGB(canvas,rgb){
  canvas.width=224;canvas.height=224;const ctx=canvas.getContext('2d'),image=ctx.createImageData(224,224);
  for(let i=0,j=0;i<rgb.length;i+=3,j+=4){image.data[j]=rgb[i];image.data[j+1]=rgb[i+1];image.data[j+2]=rgb[i+2];image.data[j+3]=255;}
  ctx.putImageData(image,0,0);
}
