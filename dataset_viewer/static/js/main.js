// Main application logic
window.loadDataset = async () => {
    const pathInput = document.getElementById('selected-path');
    const loadBtn = document.getElementById('load-btn');
    const galleryContainer = document.getElementById('gallery-container');
    
    loadBtn.disabled = true;
    loadBtn.textContent = 'Loading...';
    
    try {
        const formData = new FormData();
        formData.append('dataset_path', pathInput.value);
        
        const response = await fetch('/analyze_dataset', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        // Load first page
        await loadPage(1);
        
    } catch (error) {
        galleryContainer.innerHTML = `
            <div class="error">
                ${error.message || 'Failed to load dataset'}
            </div>
        `;
    } finally {
        loadBtn.disabled = false;
        loadBtn.textContent = 'Load Dataset';
    }
};

// Page navigation
window.changePage = async (direction) => {
    const currentPage = parseInt(document.getElementById('current-page').value);
    const totalPages = parseInt(document.getElementById('total-pages').value);
    const newPage = currentPage + direction;
    
    if (newPage >= 1 && newPage <= totalPages) {
        await loadPage(newPage);
    }
};

// Load specific page
async function loadPage(page) {
    const galleryContainer = document.getElementById('gallery-container');
    
    try {
        const formData = new FormData();
        formData.append('page', page);
        
        const response = await fetch('/get_page', {
            method: 'POST',
            body: formData
        });
        
        const html = await response.text();
        galleryContainer.innerHTML = html;
        
    } catch (error) {
        galleryContainer.innerHTML = `
            <div class="error">
                Failed to load page: ${error.message}
            </div>
        `;
    }
}

// Modal handling
window.showModal = (imageData) => {
    const modal = document.getElementById('imageModal');
    const modalImg = document.getElementById('modalImage');
    modal.style.display = 'block';
    modalImg.src = `data:image/jpeg;base64,${imageData}`;
};

window.hideModal = () => {
    const modal = document.getElementById('imageModal');
    modal.style.display = 'none';
};

// Path input validation
document.getElementById('selected-path')?.addEventListener('input', (e) => {
    const loadBtn = document.getElementById('load-btn');
    loadBtn.disabled = !e.target.value.trim();
});