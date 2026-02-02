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
    formData.append('k_minus', document.getElementById('k_minus').value);
    formData.append('k_plus', document.getElementById('k_plus').value);
    formData.append('lambda_param', document.getElementById('lambda').value);

    console.log('Submitting form with k_minus:', document.getElementById('k_minus').value, 'k_plus:', document.getElementById('k_plus').value);

    submitBtn.disabled = true;
    submitBtn.textContent = 'Analyzing...';

    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            body: formData
        });

        console.log('Response status:', response.status);

        const data = await response.json();
        console.log('Response data:', data);

        if (data.success) {
            analysisResult = data.result;
            console.log('Analysis result:', analysisResult);
            displayResults();
        } else {
            console.error('Analysis failed:', data.error);
            showError(data.error || 'Unknown error occurred');
        }
    } catch (error) {
        console.error('Error during analysis:', error);
        showError(`Network error: ${error.message}`);
    } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = 'Analyze';
    }
}

function displayResults() {
    console.log('displayResults called');
    if (!analysisResult) {
        console.error('No analysis result available');
        return;
    }

    console.log('Analysis result structure:', {
        segments: analysisResult.segments?.length,
        metadata: analysisResult.metadata,
        params: analysisResult.params
    });

    // Show results section
    resultsSection.style.display = 'block';

    try {
        // Update stats
        const totalSegments = analysisResult.segments.length;
        const markedCount = analysisResult.segments.filter(s => s.is_marked).length;
        const markedPercent = ((markedCount / totalSegments) * 100).toFixed(1);

        document.getElementById('totalSegments').textContent = totalSegments;
        document.getElementById('markedCount').textContent = markedCount;
        document.getElementById('markedPercent').textContent = markedPercent + '%';
        document.getElementById('threshold').textContent = analysisResult.metadata.threshold.toFixed(3);
        document.getElementById('kMinus').textContent = analysisResult.params.k_minus;
        document.getElementById('kPlus').textContent = analysisResult.params.k_plus;

        console.log('Stats updated successfully');

        // Render chart
        renderChart();
        console.log('Chart rendered');

        // Render segments
        renderSegments();
        console.log('Segments rendered');

        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    } catch (error) {
        console.error('Error in displayResults:', error);
        showError(`Display error: ${error.message}`);
    }
}

function renderChart() {
    const ctx = document.getElementById('contextChart').getContext('2d');

    const labels = analysisResult.segments.map((s, i) => i);
    const contextDebt = analysisResult.segments.map(s => s.scores.context_debt);
    const contextDebtDelta = analysisResult.segments.map(s => s.scores.context_debt_delta);
    const uccScore = analysisResult.segments.map(s => s.scores.ucc_score);
    const topicSimilarity = analysisResult.segments.map(s => s.scores.topic_similarity);

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
                    label: 'Context Debt (D)',
                    data: contextDebt,
                    borderColor: 'rgb(52, 152, 219)',
                    backgroundColor: 'rgba(52, 152, 219, 0.1)',
                    tension: 0.2,
                    yAxisID: 'y'
                },
                {
                    label: 'Context Debt Delta (ΔD)',
                    data: contextDebtDelta,
                    borderColor: 'rgb(231, 76, 60)',
                    backgroundColor: 'rgba(231, 76, 60, 0.1)',
                    tension: 0.2,
                    yAxisID: 'y'
                },
                {
                    label: 'UCC Score',
                    data: uccScore,
                    borderColor: 'rgb(46, 204, 113)',
                    backgroundColor: 'rgba(46, 204, 113, 0.1)',
                    tension: 0.2,
                    yAxisID: 'y1'
                },
                {
                    label: 'Topic Similarity',
                    data: topicSimilarity,
                    borderColor: 'rgb(155, 89, 182)',
                    backgroundColor: 'rgba(155, 89, 182, 0.1)',
                    tension: 0.2,
                    yAxisID: 'y1'
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
                    text: 'Context Debt Over Time'
                },
                tooltip: {
                    callbacks: {
                        afterLabel: function(context) {
                            const segment = analysisResult.segments[context.dataIndex];
                            if (segment.is_marked) {
                                return 'MARKED';
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
                        text: 'Context Debt / ΔD'
                    }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: {
                        display: true,
                        text: 'UCC Score / Topic Similarity'
                    },
                    min: 0,
                    max: 1,
                    grid: {
                        drawOnChartArea: false
                    }
                }
            }
        }
    });
}

function renderSegments(threshold = 0) {
    segmentsList.innerHTML = '';

    const filtered = analysisResult.segments.filter(s => s.scores.ucc_score >= threshold);

    filtered.forEach((segment, idx) => {
        const div = document.createElement('div');
        div.className = 'segment' + (segment.is_marked ? ' marked' : '');
        div.onclick = () => toggleSegmentDetails(div);

        const scoreClass = segment.scores.ucc_score > 0.7 ? 'high' : '';

        div.innerHTML = `
            <div class="segment-header">
                <span class="segment-time">${formatTime(segment.start_ms)} - ${formatTime(segment.end_ms)}</span>
                <span class="segment-score ${scoreClass}">UCC: ${segment.scores.ucc_score.toFixed(2)}</span>
            </div>
            <div class="segment-text">${segment.text}</div>
            <div class="segment-details">
                <div><strong>Topic Similarity:</strong> ${segment.scores.topic_similarity?.toFixed(3) || 'N/A'}</div>
                <div><strong>Context Debt:</strong> ${segment.scores.context_debt?.toFixed(3) || 'N/A'}</div>
                <div><strong>Context Debt Δ:</strong> ${segment.scores.context_debt_delta?.toFixed(3) || 'N/A'}</div>
                ${segment.is_marked && segment.reasons.length > 0 ? `
                    <div class="reasons">
                        <strong>Missing Prerequisites:</strong><br>
                        ${segment.reasons.map(r => `<span class="reason-tag">${r}</span>`).join('')}
                    </div>
                ` : ''}
            </div>
        `;

        segmentsList.appendChild(div);
    });

    if (filtered.length === 0) {
        segmentsList.innerHTML = '<p class="loading">No segments match the current threshold</p>';
    }
}

function toggleSegmentDetails(element) {
    element.classList.toggle('active');
}

function handleThresholdChange(e) {
    const value = parseFloat(e.target.value);
    thresholdValue.textContent = value.toFixed(2);
    renderSegments(value);
}

function formatTime(ms) {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);

    const s = seconds % 60;
    const m = minutes % 60;

    if (hours > 0) {
        return `${hours}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
    } else {
        return `${m}:${s.toString().padStart(2, '0')}`;
    }
}

function showError(message) {
    const error = document.createElement('div');
    error.className = 'error';
    error.textContent = message;
    resultsSection.insertBefore(error, resultsSection.firstChild);

    setTimeout(() => error.remove(), 5000);
}
