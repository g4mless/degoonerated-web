// Goonable image classifier — ONNX Runtime Web
// Model: MobileNetV3 (torchvision), input [1,3,224,224] NCHW float32, output "logits" [1,2]

const MODEL_URL = "goonable_model.onnx";

// Urutan kelas mengikuti ImageFolder torchvision (alfabetis).
// Kalau hasilnya kebalik, tinggal tukar urutan array ini.
const CLASSES = ["goonable", "normal"];

// Preprocessing torchvision standar (ImageNet)
const SIZE = 224;          // ukuran crop akhir
const RESIZE = 256;        // resize shorter-side sebelum center crop
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

const dropEl = document.getElementById("drop");
const fileEl = document.getElementById("file");
const previewEl = document.getElementById("preview");
const resultsEl = document.getElementById("results");
const verdictEl = document.getElementById("verdict");
const resetBtn = document.getElementById("reset");

let session = null;

async function init() {
  try {
    ort.env.wasm.numThreads = 1; // aman untuk file:// dan tanpa COOP/COEP
    session = await ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ["wasm"],
    });
    dropEl.style.opacity = "1";
  } catch (e) {
    console.error(e);
    dropEl.innerHTML =
      "Failed to load model. Make sure the page is opened via an HTTP server " +
      "(not by double-clicking the file). See README.";
  }
}

// ---- preprocessing ----
function preprocess(img) {
  // 1) resize aspect-preserving sehingga sisi terpendek = RESIZE
  const scale = RESIZE / Math.min(img.width, img.height);
  const rw = Math.round(img.width * scale);
  const rh = Math.round(img.height * scale);

  const canvas = document.createElement("canvas");
  canvas.width = rw;
  canvas.height = rh;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0, rw, rh);

  // 2) center crop SIZE x SIZE
  const sx = Math.floor((rw - SIZE) / 2);
  const sy = Math.floor((rh - SIZE) / 2);
  const { data } = ctx.getImageData(sx, sy, SIZE, SIZE); // RGBA

  // 3) HWC RGBA -> NCHW RGB float32 + normalization
  const out = new Float32Array(3 * SIZE * SIZE);
  const plane = SIZE * SIZE;
  for (let i = 0; i < plane; i++) {
    const r = data[i * 4] / 255;
    const g = data[i * 4 + 1] / 255;
    const b = data[i * 4 + 2] / 255;
    out[i] = (r - MEAN[0]) / STD[0];
    out[i + plane] = (g - MEAN[1]) / STD[1];
    out[i + 2 * plane] = (b - MEAN[2]) / STD[2];
  }
  return new ort.Tensor("float32", out, [1, 3, SIZE, SIZE]);
}

function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

async function classify(img) {
  if (!session) return;

  const tensor = preprocess(img);
  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];
  const result = await session.run({ [inputName]: tensor });
  const logits = Array.from(result[outputName].data);
  const probs = softmax(logits);

  render(probs);
}

function render(probs) {
  // map prob to class index
  const goonIdx = CLASSES.indexOf("goonable");
  const normIdx = CLASSES.indexOf("normal");
  const goon = probs[goonIdx];
  const norm = probs[normIdx];

  document.getElementById("goonPct").textContent = (goon * 100).toFixed(1) + "%";
  document.getElementById("normPct").textContent = (norm * 100).toFixed(1) + "%";
  document.getElementById("goonBar").style.width = goon * 100 + "%";
  document.getElementById("normBar").style.width = norm * 100 + "%";

  const topIdx = probs.indexOf(Math.max(...probs));
  verdictEl.textContent = `Prediction: ${CLASSES[topIdx]} (${(probs[topIdx] * 100).toFixed(1)}%)`;
  verdictEl.style.color = CLASSES[topIdx] === "goonable" ? "#ef4444" : "#0044b0";

  resultsEl.style.display = "block";
}

// ---- file handling ----
function loadFile(file) {
  if (!file || !file.type.startsWith("image/")) return;
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => {
    previewEl.src = url;
    previewEl.style.display = "block";
    classify(img);
  };
  img.src = url;
}

dropEl.addEventListener("click", () => fileEl.click());
fileEl.addEventListener("change", (e) => loadFile(e.target.files[0]));

dropEl.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropEl.classList.add("hover");
});
dropEl.addEventListener("dragleave", () => dropEl.classList.remove("hover"));
dropEl.addEventListener("drop", (e) => {
  e.preventDefault();
  dropEl.classList.remove("hover");
  loadFile(e.dataTransfer.files[0]);
});

resetBtn.addEventListener("click", () => {
  resultsEl.style.display = "none";
  previewEl.style.display = "none";
  previewEl.src = "";
  fileEl.value = "";
});

init();
