# MetalloTrace

**Visual reference search for archaeometallurgy.**  
A research tool by [Mayon Padai](https://mayonpadai.github.io/).

**[Open MetalloTrace](https://theormets.github.io/MetalloTrace/)**

## What it does

Upload a microscope image to explore visually similar published references, their source-reported material and sample information, and original paper details. The reference collection includes archaeological metals, corrosion, slags, ceramics and associated materials, alongside some supporting or modern comparative references.

MetalloTrace supports literature discovery and expert comparison. It is designed for archaeometallurgy reference search; it is not a general-purpose metallurgy identification service. A retrieved reference does not establish the uploaded specimen's composition, age, provenance or identity.

## Using the tool

1. Upload a PNG, JPG or JPEG, or load a public HTTPS image URL that permits browser access.
2. Click **Trace similar micrographs**.
3. Inspect the reference images, cosine scores and expandable source details.
4. Follow the DOI links to consult the original papers, or export the results as CSV.

**More search options** controls the number of returned matches. **Reset** clears the selected image, URL, results and options while retaining the loaded model for subsequent searches.

Uploads are limited to 64 MB. The decoding pipeline has a 32-megapixel image limit. The exact PNG path supports opaque, 8-bit, non-interlaced PNGs. JPEG uses browser decoding and has not received the same pixel-equivalence validation as the calibration PNGs.

## How retrieval works

```text
Uploaded image -> consistent RGB preparation and padding
              -> pretrained DINOv2-small / 384-number CLS vector
              -> cosine comparison with saved reference vectors
              -> grouped reference matches and source details
```

The model is pretrained; this project does not train an SLM or a new material classifier. Query filenames, paper titles and material names do not determine visual similarity ranking.

The current library contains 2,371 image records from 224 papers. These are catalogue records, not a verified count of independent physical specimens. The previously identified retracted paper is excluded from this deployment.

All references in the deployed library are considered, including Approved and Hold records and all recorded microscopy methods. Returned matches are grouped by recorded sample; unlinked records are grouped by paper. After top matches are selected by cosine similarity, Approved references are displayed first and Hold references last. Display order therefore differs from pure cosine order.

**Hold** indicates unresolved source/sample information. **Approved** indicates the extraction review status, not confirmed identification of the uploaded specimen. Source fields may be missing, broad or uncertain. Full original master records are available within result details.

**Cosine similarity is not a material-identification probability.** For example, 0.85 does not mean 85% confidence that the sample is a particular metal.

## Deployment and privacy

This is a static GitHub Pages application. ONNX Runtime Web executes the exported DINOv2 model on the user's device using WebAssembly. Normal image search processes query images locally rather than uploading them to a Python inference backend.

The model is approximately 86.6 MB. The first search loads the model and initializes the runtime. Within the same page the model is reused; on later visits a verified local browser copy is reused when storage permits. Browser restrictions, eviction or cleared site data can require another download. Model initialization still takes time after a page reload.

The runtime loads from jsDelivr and fonts load from Google Fonts. Hosting services receive ordinary asset requests. Loading an image URL contacts that external image host. The app does not intentionally add uploaded images to the reference database.

Reference images displayed in results are compressed previews. Consult original papers for full-resolution figures and quantitative interpretation.

## Validation status

- Initial paper-held-out retrieval evaluation was conducted on a small pilot query set.
- The exported model and the upload preprocessing path were checked against saved reference calculations using ten calibration PNGs.
- These consistency checks are distinct from scientific identification performance.
- The current all-microscopy, Hold-inclusive search policy is broader than the earlier baseline configuration; baseline metrics must not be presented as remeasured performance of this version.
- Larger expert-reviewed benchmarks, independent laboratory specimens, and controlled comparisons with alternative retrieval methods remain necessary.

## Repository layout

```text
index.html                 Page structure
style.css                  Styling and responsive layout
app.js                     Upload, ranking, results and CSV export
runtime-core.js            Model loading, caching and vector utilities
preprocess.js              RGB preparation, padding and normalization
png-rgb.js                 PNG decoding
reference_embeddings.f32   Saved reference vectors in metadata row order
data/library.json          File hashes, shape and model configuration
data/references.json       Reference metadata and source records
models/                    Verified DINOv2-small ONNX export
images/part01 ... part05/   Reference display previews
assets/                    Branding and decorative artwork
THIRD_PARTY_NOTES.txt       Implementation references and notices
.nojekyll                  Static GitHub Pages serving
```

Keep metadata and embeddings aligned. Changes to metadata require updating its SHA-256 in `data/library.json`. Changes to the model, preprocessing or vector extraction require appropriate consistency checks and may require rebuilding reference embeddings.

GitHub Pages publishes the `main` branch root. Do not apply Git LFS tracking to deployment assets: Git LFS is not supported by GitHub Pages. All current deployment files fit below GitHub's ordinary per-file limit.

## Attribution and reuse

- Visual encoder: [DINOv2, Meta AI](https://github.com/facebookresearch/dinov2).
- Browser inference: [ONNX Runtime](https://onnxruntime.ai/).
- Team identity and supplied logo: Mayon Padai.
- Decorative microscope banner: generated artwork; it is not reference or analytical data.

Research-paper figures and metadata retain their original source attribution and applicable rights. This repository does not grant a blanket reuse license for third-party figures or model assets. Consult original sources and their licenses before redistribution. See `THIRD_PARTY_NOTES.txt` for additional implementation references.
