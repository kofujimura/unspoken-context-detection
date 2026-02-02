// Global state
let analysisResult = null;
let chart = null;

// DOM elements
const uploadForm = document.getElementById('uploadForm');
const submitBtn = document.getElementById('submitBtn');
const resultsSection = document.getElementById('resultsSection');
const scoreThreshold = document.getElementById('scoreThreshold');
const thresholdValue = document.getElementById('thresholdValue');
const segmentsList = document.getElementById('segmentsList');

// Event listeners
uploadForm.addEventListener('submit', handleSubmit);
scoreThreshold.addEventListener('input', handleThresholdChange);

async function handleSubmit(e) {
    e.preventDefault();

    const formData = new FormData();
    formData.append('file', document.getElementById('file').files[0]);
    formData.append('video_id', document.getElementById('videoId').value);
    formData.append('language', document.getElementById('language').value);

    const chunkSize = document.getElementById('chunkSize').value;
    const overlap = document.getElementById('overlap').value;
    const model = document.getElementById('model').value;

    if (chunkSize) formData.append('chunk_size', chunkSize);
    if (overlap) formData.append('overlap', overlap);
    if (model) formData.append('model', model);

    submitBtn.disabled = true;
    submitBtn.textContent = 'Analyzing... (This may take a while)';

    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Analysis failed');
        }

        const data = await response.json();
        analysisResult = data;
        displayResults();
    } catch (error) {
        showError(`Error: ${error.message}`);
    } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = 'Analyze';
    }
}

function displayResults() {
    if (!analysisResult) return;

    // Show results section
    resultsSection.style.display = 'block';

    // Update stats
    const totalSegments = analysisResult.segments.length;
    const markedCount = analysisResult.segments.filter(s => s.is_marked).length;
    const markedPercent = ((markedCount / totalSegments) * 100).toFixed(1);

    document.getElementById('totalSegments').textContent = totalSegments;
    document.getElementById('markedCount').textContent = markedCount;
    document.getElementById('markedPercent').textContent = markedPercent + '%';
    document.getElementById('totalChunks').textContent = analysisResult.total_chunks || '-';
    document.getElementById('totalMarks').textContent = analysisResult.total_marks || '-';
    document.getElementById('chunkSizeValue').textContent = analysisResult.params?.chunk_size || '-';
    document.getElementById('overlapValue').textContent = analysisResult.params?.overlap || '-';

    // Render chart
    renderChart();

    // Render segments
    renderSegments();

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

function renderChart() {
    const ctx = document.getElementById('contextChart').getContext('2d');

    const labels = analysisResult.segments.map((s, i) => i);
    const uccScore = analysisResult.segments.map(s => s.ucc_score);
    const severity = analysisResult.segments.map(s => s.severity || 0);

    // Mark background colors
    const backgroundColors = analysisResult.segments.map(s =>
        s.is_marked ? 'rgba(231, 76, 60, 0.1)' : 'rgba(0, 0, 0, 0)'
    );

    if (chart) {
        chart.destroy();
    }

    chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'UCC Score',
                    data: uccScore,
                    borderColor: 'rgb(155, 89, 182)',
                    backgroundColor: 'rgba(155, 89, 182, 0.1)',
                    tension: 0.2,
                    yAxisID: 'y'
                },
                {
                    label: 'Severity',
                    data: severity,
                    borderColor: 'rgb(231, 76, 60)',
                    backgroundColor: 'rgba(231, 76, 60, 0.1)',
                    tension: 0.2,
                    yAxisID: 'y1',
                    stepped: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                title: {
                    display: true,
                    text: 'UCC Score and Severity Over Time'
                },
                tooltip: {
                    callbacks: {
                        afterLabel: function(context) {
                            const segment = analysisResult.segments[context.dataIndex];
                            if (segment.is_marked && segment.reasons && segment.reasons.length > 0) {
                                return 'Missing: ' + segment.reasons.join(', ');
                            }
                            return '';
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Segment Index'
                    }
                },
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: {
                        display: true,
                        text: 'UCC Score'
                    },
                    min: 0,
                    max: 1
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: {
                        display: true,
                        text: 'Severity (1-5)'
                    },
                    min: 0,
                    max: 5,
                    ticks: {
                        stepSize: 1
                    },
                    grid: {
                        drawOnChartArea: false
                    }
                }
            }
        }
    });
}

function renderSegments() {
    const threshold = parseFloat(scoreThreshold.value);
    const filtered = analysisResult.segments.filter(s => s.ucc_score >= threshold);

    segmentsList.innerHTML = '';

    filtered.forEach(segment => {
        const segmentDiv = document.createElement('div');
        segmentDiv.className = 'segment' + (segment.is_marked ? ' marked' : '');

        const scoreClass = segment.ucc_score > 0.7 ? ' high' : '';
        const severityClass = segment.severity ? ` severity-${segment.severity}` : '';

        segmentDiv.innerHTML = `
            <div class="segment-header">
                <span class="segment-time">#${segment.id} | ${formatTime(segment.start_ms)} - ${formatTime(segment.end_ms)}</span>
                <span>
                    <span class="segment-score${scoreClass}">Score: ${segment.ucc_score.toFixed(2)}</span>
                    ${segment.severity ? `<span class="severity-badge${severityClass}">Severity ${segment.severity}</span>` : ''}
                </span>
            </div>
            <div class="segment-text">${escapeHtml(segment.text)}</div>
            <div class="segment-details">
                ${segment.reasons && segment.reasons.length > 0 ? `
                    <div class="reasons">
                        <strong>Missing Prerequisites:</strong><br>
                        ${segment.reasons.map(r => `<span class="reason-tag">${escapeHtml(r)}</span>`).join('')}
                    </div>
                ` : ''}
                ${segment.micro_suggestion ? `
                    <div class="micro-suggestion">
                        <strong>💡 Suggestion:</strong> ${escapeHtml(segment.micro_suggestion)}
                    </div>
                ` : ''}
            </div>
        `;

        segmentDiv.addEventListener('click', () => {
            segmentDiv.classList.toggle('active');
        });

        segmentsList.appendChild(segmentDiv);
    });

    if (filtered.length === 0) {
        segmentsList.innerHTML = '<div class="loading">No segments match the current threshold.</div>';
    }
}

function handleThresholdChange(e) {
    const value = parseFloat(e.target.value);
    thresholdValue.textContent = value.toFixed(2);
    renderSegments();
}

function formatTime(ms) {
    const totalSeconds = Math.floor(ms / 1000);
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    const milliseconds = ms % 1000;

    return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}.${String(milliseconds).padStart(3, '0')}`;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error';
    errorDiv.textContent = message;

    const container = document.querySelector('.container');
    container.insertBefore(errorDiv, container.firstChild);

    setTimeout(() => {
        errorDiv.remove();
    }, 5000);
}
