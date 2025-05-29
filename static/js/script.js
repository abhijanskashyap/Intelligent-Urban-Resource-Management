document.addEventListener('DOMContentLoaded', () => {
    // Event listener for "Generate Data" button
    const generateDataBtn = document.getElementById('generateDataBtn');
    if (generateDataBtn) {
        generateDataBtn.addEventListener('click', async () => {
            const statusDiv = document.getElementById('dataGenerationStatus');
            statusDiv.textContent = 'Generating data... Please wait.';
            statusDiv.style.color = 'blue';
            try {
                const response = await fetch('/generate_data', { method: 'POST' });
                const result = await response.json();
                statusDiv.textContent = result.message;
                statusDiv.style.color = 'green';
            } catch (error) {
                console.error('Error generating data:', error);
                statusDiv.textContent = 'Error generating data.';
                statusDiv.style.color = 'red';
            }
        });
    }

    // Event listener for electricity prediction form
    const electricityPredictForm = document.getElementById('electricityPredictForm');
    if (electricityPredictForm) {
        electricityPredictForm.addEventListener('submit', async (e) => {
            e.preventDefault(); // Prevent default form submission

            const formData = new FormData(electricityPredictForm);
            // Convert FormData to a plain object for JSON
            const data = {};
            for (let [key, value] of formData.entries()) {
                // Ensure numeric values are parsed correctly
                if (['hour', 'dayofweek', 'month', 'traffic_density'].includes(key)) {
                    data[key] = parseInt(value);
                } else if (['water_consumption', 'waste_level'].includes(key)) {
                    data[key] = parseFloat(value);
                } else {
                    data[key] = value;
                }
            }

            const predictionResultDiv = document.getElementById('electricityPredictionResult');
            predictionResultDiv.textContent = 'Predicting...';
            predictionResultDiv.style.color = 'blue';

            try {
                // Send POST request to the new AJAX endpoint
                const response = await fetch('/predict_electricity_ajax', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json' // Specify JSON content type
                    },
                    body: JSON.stringify(data) // Send data as JSON string
                });

                const result = await response.json(); // Parse JSON response
                if (response.ok) { // Check if HTTP status is 2xx
                    predictionResultDiv.textContent = `Predicted Electricity Usage: ${result.prediction}`;
                    predictionResultDiv.style.color = 'green';
                } else {
                    // Handle API errors (e.g., 400 Bad Request from Flask)
                    predictionResultDiv.textContent = `Error: ${result.error || 'Unknown error'}`;
                    predictionResultDiv.style.color = 'red';
                    console.error('API Error:', result.error);
                }
            } catch (error) {
                // Handle network errors or issues with JSON parsing
                console.error('Error during electricity prediction fetch:', error);
                predictionResultDiv.textContent = 'Error during prediction. Please check console for details.';
                predictionResultDiv.style.color = 'red';
            }
        });
    }

    // Event listener for event classification form
    const eventClassifyForm = document.getElementById('eventClassifyForm');
    if (eventClassifyForm) {
        eventClassifyForm.addEventListener('submit', async (e) => {
            e.preventDefault(); // Prevent default form submission

            const formData = new FormData(eventClassifyForm);
            // Convert FormData to a plain object for JSON
            const data = {};
            for (let [key, value] of formData.entries()) {
                // Ensure numeric values are parsed correctly
                if (['electricity_usage', 'water_consumption', 'waste_level'].includes(key)) {
                    data[key] = parseFloat(value);
                } else if (['traffic_density'].includes(key)) {
                    data[key] = parseInt(value);
                } else {
                    data[key] = value;
                }
            }

            const classificationResultDiv = document.getElementById('eventClassificationResult');
            classificationResultDiv.textContent = 'Classifying...';
            classificationResultDiv.style.color = 'blue';

            try {
                // Send POST request to the new AJAX endpoint
                const response = await fetch('/classify_event_ajax', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json' // Specify JSON content type
                    },
                    body: JSON.stringify(data) // Send data as JSON string
                });

                const result = await response.json(); // Parse JSON response
                if (response.ok) { // Check if HTTP status is 2xx
                    classificationResultDiv.textContent = `Predicted Event/Anomaly: ${result.prediction}`;
                    classificationResultDiv.style.color = 'green';
                } else {
                    // Handle API errors
                    classificationResultDiv.textContent = `Error: ${result.error || 'Unknown error'}`;
                    classificationResultDiv.style.color = 'red';
                    console.error('API Error:', result.error);
                }
            } catch (error) {
                // Handle network errors or issues with JSON parsing
                console.error('Error during event classification fetch:', error);
                classificationResultDiv.textContent = 'Error during classification. Please check console for details.';
                classificationResultDiv.style.color = 'red';
            }
        });
    }
});