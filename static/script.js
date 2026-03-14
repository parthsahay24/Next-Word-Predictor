document.addEventListener('DOMContentLoaded', () => {
    const textInput = document.getElementById('text-input');
    const suggestionsList = document.getElementById('suggestions-list');
    const loader = document.getElementById('loader');
    const tempSlider = document.getElementById('temperature');
    const tempValue = document.getElementById('temp-value');
    const topkSlider = document.getElementById('top-k');
    const topkValue = document.getElementById('top-k-value');
    const sidePanel = document.getElementById('sidePanel');
    const panelClose = document.getElementById('panelClose');
    const iconBtns = document.querySelectorAll('.icon-btn');

    let debounceTimer;

    // ── Side panel logic ─────────────────────────────
    iconBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const panelWasOpen = sidePanel.classList.contains('open');
            const wasActive = btn.classList.contains('active');

            // Toggle panel
            if (panelWasOpen && wasActive) {
                sidePanel.classList.remove('open');
                iconBtns.forEach(b => b.classList.remove('active'));
            } else {
                sidePanel.classList.add('open');
                iconBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                // highlight relevant card
                const card = document.getElementById('card-' + btn.dataset.panel);
                if (card) {
                    document.querySelectorAll('.panel-card').forEach(c => c.style.background = '');
                    card.style.background = 'rgba(0, 212, 255, 0.15)';
                }
            }
        });
    });

    panelClose.addEventListener('click', () => {
        sidePanel.classList.toggle('open');
        if (!sidePanel.classList.contains('open')) {
            iconBtns.forEach(b => b.classList.remove('active'));
        }
    });

    // Close panel when clicking outside
    document.addEventListener('click', (e) => {
        if (!sidePanel.contains(e.target) && !e.target.closest('.side-icons')) {
            sidePanel.classList.remove('open');
            iconBtns.forEach(b => b.classList.remove('active'));
        }
    });

    // ── Slider values ────────────────────────────────
    tempSlider.addEventListener('input', (e) => {
        tempValue.textContent = parseFloat(e.target.value).toFixed(1);
        triggerPrediction();
    });

    topkSlider.addEventListener('input', (e) => {
        topkValue.textContent = e.target.value;
        triggerPrediction();
    });

    // ── Typing handler ───────────────────────────────
    textInput.addEventListener('input', () => {
        clearTimeout(debounceTimer);
        const text = textInput.value;

        if (text.trim().length === 0) {
            suggestionsList.innerHTML = '';
            loader.classList.remove('active');
            return;
        }

        if (!text.endsWith(' ')) {
            suggestionsList.innerHTML = '';
            return;
        }

        loader.classList.add('active');
        suggestionsList.innerHTML = '';
        debounceTimer = setTimeout(() => fetchPredictions(text), 400);
    });

    // Tab to autocomplete
    textInput.addEventListener('keydown', (e) => {
        if (e.key === 'Tab') {
            const first = suggestionsList.querySelector('.suggestion-chip');
            if (first) { e.preventDefault(); insertWord(first.dataset.word); }
        }
    });

    function triggerPrediction() {
        const text = textInput.value;
        if (text.trim().length > 0 && text.endsWith(' ')) {
            loader.classList.add('active');
            fetchPredictions(text);
        }
    }

    async function fetchPredictions(text) {
        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text,
                    temperature: parseFloat(tempSlider.value),
                    top_k: parseInt(topkSlider.value)
                })
            });
            const data = await response.json();
            loader.classList.remove('active');
            if (data.error) { console.error('API Error:', data.error); return; }
            renderSuggestions(data.predictions);
        } catch (err) {
            loader.classList.remove('active');
            console.error('Prediction error:', err);
        }
    }

    function renderSuggestions(predictions) {
        suggestionsList.innerHTML = '';
        if (!predictions || predictions.length === 0) return;
        predictions.forEach((pred, i) => {
            const li = document.createElement('li');
            li.className = 'suggestion-chip';
            li.dataset.word = pred.word;
            li.style.animationDelay = `${i * 0.05}s`;
            li.innerHTML = `<span class="chip-word">${pred.word}</span><span class="chip-prob">${pred.probability.toFixed(1)}%</span>`;
            li.addEventListener('click', () => insertWord(pred.word));
            suggestionsList.appendChild(li);
        });
    }

    function insertWord(word) {
        textInput.value += word + ' ';
        textInput.focus();
        suggestionsList.innerHTML = '';
        triggerPrediction();
    }
});
