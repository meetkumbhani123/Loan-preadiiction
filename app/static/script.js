document.getElementById('prediction-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const btnText = document.querySelector('.btn-text');
    const loader = document.querySelector('.loader');
    
    // UI state loading
    btnText.classList.add('hidden');
    loader.classList.remove('hidden');
    
    // Gather form data
    const formData = new FormData(e.target);
    const data = Object.fromEntries(formData.entries());
    
    // Type casting
    data.Age = parseInt(data.Age);
    data.Income = parseFloat(data.Income);
    data.LoanAmount = parseFloat(data.LoanAmount);
    data.CreditScore = parseFloat(data.CreditScore);
    data.MonthsEmployed = parseFloat(data.MonthsEmployed);
    data.NumCreditLines = parseFloat(data.NumCreditLines);
    data.InterestRate = parseFloat(data.InterestRate);
    data.LoanTerm = parseFloat(data.LoanTerm);
    data.DTIRatio = parseFloat(data.DTIRatio);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        displayResults(result, data);
    } catch (error) {
        console.error('Error fetching prediction:', error);
        alert('Failed to get prediction. Ensure the backend is running.');
    } finally {
        // UI reset
        btnText.classList.remove('hidden');
        loader.classList.add('hidden');
    }
});

function displayResults(result, data) {
    const panel = document.getElementById('results-panel');
    panel.classList.remove('hidden');
    
    // Smooth scroll to results
    panel.scrollIntoView({ behavior: 'smooth' });
    
    const probPercentage = (result.default_probability * 100).toFixed(1);
    const probElem = document.getElementById('prob-percentage');
    const statusElem = document.getElementById('status-message');
    const ringElem = document.querySelector('.prediction-ring');
    const confidenceVal = document.querySelectorAll('.metric-val')[0];
    const driverVal = document.querySelectorAll('.metric-val')[1];
    
    // Animate percentage
    animateValue(probElem, 0, probPercentage, 1000);
    
    // Update styling based on risk
    const isHighRisk = result.prediction === 1;
    
    if (isHighRisk) {
        ringElem.style.background = `conic-gradient(var(--danger) ${probPercentage}%, rgba(255,255,255,0.05) 0%)`;
        statusElem.textContent = 'HIGH RISK';
        statusElem.className = 'status-indicator status-high';
    } else {
        ringElem.style.background = `conic-gradient(var(--success) ${probPercentage}%, rgba(255,255,255,0.05) 0%)`;
        statusElem.textContent = 'LOW RISK';
        statusElem.className = 'status-indicator status-low';
    }
    
    // Setting confidence purely for aesthetics/demo
    confidenceVal.textContent = probPercentage > 80 || probPercentage < 20 ? 'High' : 'Moderate';
    driverVal.textContent = data.DTIRatio > 0.4 ? 'DTI Ratio' : 'Credit Score';
}

function animateValue(obj, start, end, duration) {
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        obj.innerHTML = (progress * (end - start) + start).toFixed(1) + '%';
        if (progress < 1) {
            window.requestAnimationFrame(step);
        }
    };
    window.requestAnimationFrame(step);
}
