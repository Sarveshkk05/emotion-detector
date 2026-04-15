const emotions = ['Happy', 'Sad', 'Angry', 'Surprised', 'Fearful', 'Disgusted', 'Neutral'];

const video = document.getElementById('camera-feed');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const pill = document.getElementById('emotion-pill');
const pillLabel = document.getElementById('pill-label');
const pillScore = document.getElementById('pill-score');
const emotionsBar = document.querySelector('.emotions-bar');

let emotionElements = {};

// Initialize Emotion UI
function createEmotionBars() {
    emotions.forEach(emotion => {
        const id = emotion.toLowerCase();
        const card = document.createElement('div');
        card.className = `emotion-card`;
        card.id = `card-${id}`;

        const header = document.createElement('div');
        header.className = 'emotion-header';
        
        const label = document.createElement('span');
        label.className = 'emotion-label';
        label.innerText = emotion;

        const score = document.createElement('span');
        score.className = 'emotion-score';
        score.innerText = '0%';
        score.id = `score-${id}`;

        header.appendChild(label);
        header.appendChild(score);

        const track = document.createElement('div');
        track.className = 'progress-track';
        
        const fill = document.createElement('div');
        fill.className = 'progress-fill';
        fill.id = `fill-${id}`;

        track.appendChild(fill);

        card.appendChild(header);
        card.appendChild(track);

        emotionsBar.appendChild(card);
        
        emotionElements[id] = { card, score, fill };
    });
}

// Start Camera setup
async function setupCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
            audio: false
        });
        video.srcObject = stream;
        
        return new Promise((resolve) => {
            video.onloadedmetadata = () => {
                resolve(video);
            };
        });
    } catch (err) {
        console.error("Error accessing camera:", err);
    }
}

// Ensure the canvas resolution matches the actual display size of the bounding rect
function resizeCanvas() {
    const rect = video.getBoundingClientRect();
    if(rect.width && rect.height) {
        // Set actual DOM size 
        canvas.style.width = `${rect.width}px`;
        canvas.style.height = `${rect.height}px`;
        // Set internal resolution strictly matching
        canvas.width = rect.width;
        canvas.height = rect.height;
    }
}

window.addEventListener('resize', resizeCanvas);


// Dashboard update function handling prediction data from the model
function updateDashboard(predictions) {
    if (!predictions || !predictions.box) {
        pill.classList.add('hidden');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        return;
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const { x, y, width, height } = predictions.box;
    
    // Scale bbox from intrinsic video coordinates to the canvas display size
    const scaleX = canvas.width / video.videoWidth;
    const scaleY = canvas.height / video.videoHeight;
    
    const drawX = x * scaleX;
    const drawY = y * scaleY;
    const drawW = width * scaleX;
    const drawH = height * scaleY;

    // Draw the simple clean purple bounding box
    ctx.strokeStyle = '#7F77DD';
    ctx.lineWidth = 2.5; 
    ctx.beginPath();
    
    if (ctx.roundRect) {
        ctx.roundRect(drawX, drawY, drawW, drawH, 12);
    } else {
        ctx.rect(drawX, drawY, drawW, drawH);
    }
    ctx.stroke();

    // Data Processing for Emotions
    let topEmotionInfo = null;
    let maxConf = -1;
    
    Object.keys(predictions.emotions).forEach(key => {
        let conf = predictions.emotions[key];
        // safety constrain values
        conf = Math.max(0, Math.min(1, conf));

        if (emotionElements[key]) {
            const percentage = Math.round(conf * 100);
            emotionElements[key].fill.style.width = `${percentage}%`;
            emotionElements[key].score.innerText = `${percentage}%`;
            emotionElements[key].card.classList.remove('active');
        }
        
        if (conf > maxConf) {
            maxConf = conf;
            topEmotionInfo = { label: key, score: conf };
        }
    });

    if (topEmotionInfo && maxConf > 0) {
        emotionElements[topEmotionInfo.label].card.classList.add('active');
        
        pillLabel.innerText = topEmotionInfo.label.charAt(0).toUpperCase() + topEmotionInfo.label.slice(1);
        pillScore.innerText = `${Math.round(topEmotionInfo.score * 100)}%`;
        
        // Position pill exactly above the bounding box element
        let pillX = drawX + (drawW / 2);
        let pillY = drawY;

        pill.style.left = `${pillX}px`;
        pill.style.top = `${pillY}px`;
        pill.classList.remove('hidden');
    }
}


// A mock tracking loop for demonstration purposes.
// This replaces an actual WebSockets or Polling fetch call to a model endpoint.
let time = 0;
function mockDetectionLoop() {
    if (video.videoWidth && video.videoHeight && canvas.width) {
        // Wobbling bounding box over the center
        const w = 240 + Math.sin(time * 0.05) * 20;
        const h = 300 + Math.cos(time * 0.05) * 20;
        const x = (video.videoWidth - w) / 2 + Math.sin(time * 0.02) * 50;
        const y = (video.videoHeight - h) / 2 + Math.cos(time * 0.03) * 50;

        // Oscillating emotion confident distribution
        const baseH = (Math.sin(time * 0.04) + 1) / 2; // oscillates 0 to 1
        
        const emotionsDist = {
            happy: 0.7 * baseH,
            sad: 0.1 * (1 - baseH),
            angry: 0.05,
            surprised: 0.2 * (1 - baseH),
            fearful: 0.05 * (1 - baseH),
            disgusted: 0.0,
            neutral: 0.3 * baseH + 0.1
        };

        // Normalization
        let total = 0;
        for (let k in emotionsDist) total += emotionsDist[k];
        for (let k in emotionsDist) emotionsDist[k] /= total;

        updateDashboard({
            box: { x, y, width: w, height: h },
            emotions: emotionsDist
        });
    }

    time++;
    // throttle FPS slightly for smoother appearance
    setTimeout(() => {
        requestAnimationFrame(mockDetectionLoop);
    }, 1000/30);
}


async function init() {
    createEmotionBars();
    await setupCamera();
    
    // Wait for the layout to settle before sizing canvas
    setTimeout(() => {
        resizeCanvas();
        requestAnimationFrame(mockDetectionLoop);
    }, 500);
}

// Start
document.addEventListener('DOMContentLoaded', init);
