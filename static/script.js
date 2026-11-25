document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('upload-form');
    const imageInput = document.getElementById('image-input');
    const fileDropArea = document.getElementById('file-drop-area');
    const imagePreviewContainer = document.getElementById('image-preview-container');
    const imagePreview = document.getElementById('image-preview');
    const fileNameDisplay = document.getElementById('file-name');
    const loader = document.getElementById('loader');
    const resultsSection = document.getElementById('results-section');

    let selectedFile = null;

    // --- File Drop Area Logic ---
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        fileDropArea.addEventListener(eventName, preventDefaults, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        fileDropArea.addEventListener(eventName, () => fileDropArea.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        fileDropArea.addEventListener(eventName, () => fileDropArea.classList.remove('dragover'), false);
    });

    fileDropArea.addEventListener('drop', handleDrop, false);
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const file = dt.files[0];
        handleFile(file);
    }

    // --- File Input Logic ---
    imageInput.addEventListener('change', (e) => {
        handleFile(e.target.files[0]);
    });

    function handleFile(file) {
        if (file && file.type.startsWith('image/')) {
            selectedFile = file;
            fileNameDisplay.textContent = file.name;
            
            const reader = new FileReader();
            reader.onload = (e) => {
                imagePreview.src = e.target.result;
                imagePreviewContainer.classList.remove('hidden');
            };
            reader.readAsDataURL(file);
        } else {
            alert('Please select an image file.');
        }
    }

    // --- Form Submission Logic ---
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (!selectedFile) {
            alert('Please select an image file first.');
            return;
        }

        // Hide previous results, show loader
        resultsSection.classList.add('hidden');
        loader.classList.remove('hidden');

        const formData = new FormData();
        formData.append('image', selectedFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'An error occurred during analysis.');
            }

            const data = await response.json();
            displayResults(data);

        } catch (error) {
            console.error('Error:', error);
            alert(error.message);
        } finally {
            loader.classList.add('hidden');
        }
    });

    // --- Display Results ---
    function displayResults(data) {
        // Diagnosis Panel
        document.getElementById('prediction-class').textContent = data.prediction;
        document.getElementById('prediction-confidence').textContent = `${data.confidence}%`;

        // Probabilities Chart
        const probList = document.getElementById('prob-list');
        probList.innerHTML = '';
        data.probabilities.forEach(item => {
            const li = document.createElement('li');
            li.innerHTML = `
                <div class="prob-bar-container">
                    <span class="prob-label">${item.class}</span>
                    <div class="prob-bar">
                        <div class="prob-bar-inner" style="width: ${item.prob * 100}%;">
                            ${(item.prob * 100).toFixed(1)}%
                        </div>
                    </div>
                </div>
            `;
            probList.appendChild(li);
        });

        // Images Panel
        document.getElementById('original-image').src = data.original_image;
        document.getElementById('gradcam-image').src = data.gradcam_image;

        // Show the results
        resultsSection.classList.remove('hidden');
    }
});