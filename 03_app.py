import cv2
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse

app = FastAPI(title="Diabetic Foot Digital Twin Server")

# Load artifacts
model = joblib.load("xgb_ulcer_model.pkl")
explainer = joblib.load("shap_explainer.pkl")
features = joblib.load("feature_names.pkl")
patient_db = pd.read_csv("digital_twin_patient_data.csv")

stream_index = 0

def extract_foot_temperatures_from_bytes(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    temp_map = 26.0 + (gray.astype(np.float32) / 255.0) * 10.0
    coords = cv2.findNonZero(mask)
    if coords is None:
        return 28.0, 28.0, 28.0 
    
    x, y, w, h = cv2.boundingRect(coords)
    
    toes_mask = mask[y : y + int(h * 0.33), x : x + w]
    midfoot_mask = mask[y + int(h * 0.33) : y + int(h * 0.66), x : x + w]
    heel_mask = mask[y + int(h * 0.66) : y + h, x : x + w]
    
    toes_temp = temp_map[y : y + int(h * 0.33), x : x + w][toes_mask > 0].mean() if np.any(toes_mask > 0) else 28.0
    mid_temp = temp_map[y + int(h * 0.33) : y + int(h * 0.66), x : x + w][midfoot_mask > 0].mean() if np.any(midfoot_mask > 0) else 28.0
    heel_temp = temp_map[y + int(h * 0.66) : y + h, x : x + w][heel_mask > 0].mean() if np.any(heel_mask > 0) else 28.0
    
    return float(toes_temp), float(mid_temp), float(heel_temp)

@app.post("/analyze-image")
async def analyze_uploaded_image(file: UploadFile = File(...)):
    contents = await file.read()
    toes_t, mid_t, heel_t = extract_foot_temperatures_from_bytes(contents)
    
    row = {
        "toes_temp": round(toes_t, 2),
        "midfoot_temp": round(mid_t, 2),
        "heel_temp": round(heel_t, 2),
        "age": 58,
        "hba1c": 8.2,
        "diabetes_years": 12
    }
    
    input_df = pd.DataFrame([row])[features]
    prob = float(model.predict_proba(input_df)[0][1])
    shap_vals = explainer.shap_values(input_df)[0]
    
    row["predicted_risk_pct"] = round(prob * 100, 2)
    row["shap_summary"] = dict(zip(features, np.round(shap_vals.tolist(), 2)))
    row["patient_id"] = "LIVE UPLOAD"
    row["file_name"] = file.filename
    return row

@app.get("/telemetry/stream-next")
def stream_next_packet():
    global stream_index
    if len(patient_db) == 0:
        return {}
    
    row = patient_db.iloc[stream_index].to_dict()
    stream_index = (stream_index + 1) % len(patient_db)
    
    input_df = pd.DataFrame([row])[features]
    prob = float(model.predict_proba(input_df)[0][1])
    shap_vals = explainer.shap_values(input_df)[0]
    
    row["predicted_risk_pct"] = round(prob * 100, 2)
    row["shap_summary"] = dict(zip(features, np.round(shap_vals.tolist(), 2)))
    return row

@app.get("/", response_class=HTMLResponse)
def twin_dashboard():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>UlcerSense Digital Twin</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <style>
            body { font-family: 'Inter', sans-serif; background-color: #070d18; }
            .glass-panel { background: rgba(15, 23, 42, 0.7); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.08); }
            .smooth-transition { transition: all 0.4s ease; }
        </style>
    </head>
    <body class="text-slate-100 p-6 min-h-screen flex flex-col items-center">
        <div class="w-full max-w-6xl space-y-6">
            
            <header class="glass-panel rounded-2xl p-5 flex justify-between items-center shadow-lg">
                <div>
                    <h1 class="text-2xl font-bold text-teal-400">UlcerSense Digital Twin</h1>
                    <p class="text-sm text-slate-400">Autonomous Micro-climate & DFU Predictive Telemetry</p>
                </div>
                <div class="flex gap-3 items-center">
                    <button onclick="injectHighRiskTest()" class="bg-red-600 hover:bg-red-500 text-white px-4 py-2 rounded text-sm font-bold transition-colors">
                        Test High Risk
                    </button>
                    <button id="pauseBtn" onclick="toggleStream()" class="bg-slate-700 hover:bg-slate-600 px-4 py-2 rounded text-sm font-bold transition-colors">
                        Pause Stream
                    </button>
                    <label class="cursor-pointer bg-teal-600 hover:bg-teal-500 text-white text-sm font-bold py-2 px-4 rounded transition-colors">
                        Upload Scan
                        <input type="file" class="hidden" accept="image/*" onchange="handleUpload(event)">
                    </label>
                </div>
            </header>

            <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                <!-- Spatial Heatmap Representation -->
                <div class="glass-panel rounded-2xl p-6 flex flex-col items-center">
                    <h2 class="text-xs uppercase tracking-wider text-slate-400 mb-4 font-semibold w-full border-b border-slate-700 pb-2 text-center">
                        Thermal Nodes
                    </h2>
                    <div class="relative w-36 h-[300px] bg-slate-950 border border-slate-700 rounded-t-[50px] rounded-b-[30px] flex flex-col justify-between p-3">
                        <div id="toesNode" class="w-full h-[30%] rounded-t-[40px] rounded-b-xl flex items-center justify-center font-bold text-xs text-slate-950 bg-emerald-500 smooth-transition">Toes</div>
                        <div id="midNode" class="w-full h-[30%] rounded-xl flex items-center justify-center font-bold text-xs text-slate-950 bg-emerald-500 smooth-transition">Mid</div>
                        <div id="heelNode" class="w-full h-[30%] rounded-t-xl rounded-b-[20px] flex items-center justify-center font-bold text-xs text-slate-950 bg-emerald-500 smooth-transition">Heel</div>
                    </div>
                    <div class="mt-4 w-full text-xs text-slate-300 space-y-1">
                        <div class="flex justify-between"><span>Toes:</span> <span id="valToes" class="font-mono">--°C</span></div>
                        <div class="flex justify-between"><span>Mid:</span> <span id="valMid" class="font-mono">--°C</span></div>
                        <div class="flex justify-between"><span>Heel:</span> <span id="valHeel" class="font-mono">--°C</span></div>
                    </div>
                </div>

                <!-- Live Stream Analytics -->
                <div class="md:col-span-2 flex flex-col gap-6">
                    <div class="glass-panel rounded-2xl p-5">
                        <div class="flex justify-between items-center mb-2">
                            <h2 class="text-xs uppercase text-slate-400 font-semibold tracking-wider">Live Risk Stream</h2>
                            <div id="alertBanner" class="px-3 py-1 text-xs font-bold rounded bg-slate-800 text-slate-300">
                                AWAITING TELEMETRY
                            </div>
                        </div>
                        <div class="h-48"><canvas id="riskChart"></canvas></div>
                    </div>

                    <div class="grid grid-cols-2 gap-4">
                        <div class="glass-panel rounded-2xl p-5">
                            <h3 class="text-xs uppercase text-slate-400 font-semibold mb-3 tracking-wider">Patient Profile</h3>
                            <div id="patientBadge" class="text-sm font-bold text-teal-400 mb-2">ID: Initializing...</div>
                            <div class="text-sm space-y-1 text-slate-300">
                                <div>HbA1c: <span id="valHbA1c" class="font-mono text-white">--</span></div>
                                <div>Age: <span id="valAge" class="font-mono text-white">--</span></div>
                                <div>Diabetes Duration: <span id="valDuration" class="font-mono text-white">-- yrs</span></div>
                            </div>
                        </div>
                        <div class="glass-panel rounded-2xl p-5">
                            <h3 class="text-xs uppercase text-slate-400 font-semibold mb-3 tracking-wider">AI Explainability (SHAP)</h3>
                            <div id="shapList" class="text-xs space-y-1.5 font-mono text-slate-300">Evaluating feature importance...</div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Historical Log Table -->
            <div class="glass-panel rounded-2xl p-5">
                <h2 class="text-xs uppercase text-slate-400 font-semibold mb-3 border-b border-slate-700 pb-2 tracking-wider">
                    Recent Scan Telemetry Log
                </h2>
                <div class="overflow-x-auto">
                    <table class="w-full text-left text-sm">
                        <thead class="text-xs text-slate-400 uppercase bg-slate-800/60">
                            <tr>
                                <th class="px-4 py-2">Timestamp</th>
                                <th class="px-4 py-2">Patient ID</th>
                                <th class="px-4 py-2">Max Foot Temp</th>
                                <th class="px-4 py-2">Predicted Risk</th>
                                <th class="px-4 py-2">Clinical Status</th>
                            </tr>
                        </thead>
                        <tbody id="historyTable" class="divide-y divide-slate-800">
                        </tbody>
                    </table>
                </div>
            </div>

        </div>

        <script>
            let isPaused = false;
            let telemetryInterval;
            let scanHistory = [];

            function toggleStream() {
                isPaused = !isPaused;
                const btn = document.getElementById('pauseBtn');
                btn.innerText = isPaused ? "Resume Stream" : "Pause Stream";
                btn.className = isPaused 
                    ? "bg-amber-600 hover:bg-amber-500 px-4 py-2 rounded text-sm font-bold transition-colors"
                    : "bg-slate-700 hover:bg-slate-600 px-4 py-2 rounded text-sm font-bold transition-colors";
            }

            function getColor(temp) {
                if (temp > 32.5) return 'rgb(239, 68, 68)';   // Red (Hotspot / inflammation)
                if (temp > 31.0) return 'rgb(245, 158, 11)';  // Amber
                return 'rgb(16, 185, 129)';                   // Emerald Green
            }

            const ctx = document.getElementById('riskChart').getContext('2d');
            const riskChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Predicted Ulcer Risk %',
                        data: [],
                        borderColor: 'rgb(45, 212, 191)',
                        backgroundColor: 'rgba(45, 212, 191, 0.1)',
                        borderWidth: 2,
                        tension: 0.3,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        y: { min: 0, max: 100, grid: { color: '#1e293b' } },
                        x: { grid: { display: false } }
                    }
                }
            });

            function updateDashboard(data) {
                document.getElementById('patientBadge').innerText = `${data.patient_id} (${data.file_name || 'N/A'})`;
                document.getElementById('toesNode').style.backgroundColor = getColor(data.toes_temp);
                document.getElementById('midNode').style.backgroundColor = getColor(data.midfoot_temp);
                document.getElementById('heelNode').style.backgroundColor = getColor(data.heel_temp);

                document.getElementById('valToes').innerText = `${data.toes_temp.toFixed(1)}°C`;
                document.getElementById('valMid').innerText = `${data.midfoot_temp.toFixed(1)}°C`;
                document.getElementById('valHeel').innerText = `${data.heel_temp.toFixed(1)}°C`;
                document.getElementById('valHbA1c').innerText = `${data.hba1c}%`;
                document.getElementById('valAge').innerText = data.age;
                document.getElementById('valDuration').innerText = `${data.diabetes_years} yrs`;

                const alertDiv = document.getElementById('alertBanner');
                const isCritical = data.predicted_risk_pct > 55;
                if (isCritical) {
                    alertDiv.className = 'px-3 py-1 text-xs font-bold rounded bg-red-950 border border-red-500 text-red-200 animate-pulse';
                    alertDiv.innerText = `CRITICAL RISK DETECTED (${data.predicted_risk_pct}%)`;
                } else {
                    alertDiv.className = 'px-3 py-1 text-xs font-bold rounded bg-emerald-950 border border-emerald-500 text-emerald-200';
                    alertDiv.innerText = `NOMINAL TWIN STATE (${data.predicted_risk_pct}%)`;
                }

                // Render full SHAP feature list
                if (data.shap_summary) {
                    const shapHtml = Object.entries(data.shap_summary)
                        .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
                        .slice(0, 4)
                        .map(([k, v]) => `
                            <div class="flex justify-between border-b border-slate-800 pb-1">
                                <span>${k}</span>
                                <span class="${v > 0 ? 'text-red-400' : 'text-emerald-400'}">${v > 0 ? '+' : ''}${v}</span>
                            </div>
                        `).join('');
                    document.getElementById('shapList').innerHTML = shapHtml;
                }

                // Update Risk Chart
                const timeStr = new Date().toLocaleTimeString();
                if (riskChart.data.labels.length > 14) {
                    riskChart.data.labels.shift();
                    riskChart.data.datasets[0].data.shift();
                }
                riskChart.data.labels.push(timeStr);
                riskChart.data.datasets[0].data.push(data.predicted_risk_pct);
                riskChart.update();

                // Append to Historical Telemetry Table
                const maxTemp = Math.max(data.toes_temp, data.midfoot_temp, data.heel_temp).toFixed(1);
                scanHistory.unshift(`
                    <tr class="hover:bg-slate-800/40">
                        <td class="px-4 py-2">${timeStr}</td>
                        <td class="px-4 py-2 font-medium text-teal-400">${data.patient_id}</td>
                        <td class="px-4 py-2">${maxTemp}°C</td>
                        <td class="px-4 py-2 font-bold ${isCritical ? 'text-red-400' : 'text-emerald-400'}">${data.predicted_risk_pct}%</td>
                        <td class="px-4 py-2"><span class="px-2 py-0.5 rounded text-xs ${isCritical ? 'bg-red-900/60 text-red-200' : 'bg-emerald-900/60 text-emerald-200'}">${isCritical ? 'High Risk' : 'Normal'}</span></td>
                    </tr>
                `);
                if (scanHistory.length > 6) scanHistory.pop();
                document.getElementById('historyTable').innerHTML = scanHistory.join('');
            }

            function injectHighRiskTest() {
                isPaused = true;
                const btn = document.getElementById('pauseBtn');
                btn.innerText = "Resume Stream";
                btn.className = "bg-amber-600 hover:bg-amber-500 px-4 py-2 rounded text-sm font-bold transition-colors";

                const testCriticalData = {
                    patient_id: "TEST_CRITICAL_HOTSPOT",
                    file_name: "Simulated_Ulcer_Risk.png",
                    toes_temp: 34.8,
                    midfoot_temp: 31.2,
                    heel_temp: 35.6,
                    age: 63,
                    hba1c: 10.2,
                    diabetes_years: 17,
                    predicted_risk_pct: 91.5,
                    shap_summary: {
                        heel_temp: 1.84,
                        toes_temp: 1.35,
                        hba1c: 1.25,
                        diabetes_years: 0.60,
                        age: 0.18,
                        midfoot_temp: 0.05
                    }
                };
                updateDashboard(testCriticalData);
            }

            async function handleUpload(e) {
                const file = e.target.files[0];
                if (!file) return;

                isPaused = true;
                const btn = document.getElementById('pauseBtn');
                btn.innerText = "Resume Stream";
                btn.className = "bg-amber-600 hover:bg-amber-500 px-4 py-2 rounded text-sm font-bold transition-colors";
                
                const fd = new FormData();
                fd.append('file', file);
                document.getElementById('patientBadge').innerText = "Processing Scan...";
                
                try {
                    const res = await fetch('/analyze-image', { method: 'POST', body: fd });
                    const result = await res.json();
                    updateDashboard(result);
                } catch(err) {
                    console.error("Upload parsing failure:", err);
                    document.getElementById('patientBadge').innerText = "Scan Upload Failed";
                }
            }

            async function pullData() {
                if (isPaused) return;
                try {
                    const res = await fetch('/telemetry/stream-next');
                    const packet = await res.json();
                    updateDashboard(packet);
                } catch(err) {
                    console.error("Telemetry streaming error:", err);
                }
            }

            telemetryInterval = setInterval(pullData, 2500);
            pullData();
        </script>
    </body>
    </html>
    """