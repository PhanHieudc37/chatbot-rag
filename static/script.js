// ===== CONSTANTS =====
const API_BASE_URL = 'http://127.0.0.1:3737';
let currentAbortController = null;

// ===== DOM ELEMENTS =====
const ai37Toggle = document.getElementById('ai37Toggle');
const ai37ChatWindow = document.getElementById('ai37ChatWindow');
const minimizeBtn = document.getElementById('minimizeBtn');
const clearChatBtn = document.getElementById('clearChatBtn');
const ai37Messages = document.getElementById('ai37Messages');
const ai37Input = document.getElementById('ai37Input');
const sendBtn = document.getElementById('sendBtn');
const stopBtn = document.getElementById('stopBtn');
const refreshBtn = document.getElementById('refreshBtn');
const statsBtn = document.getElementById('statsBtn');
const tablesList = document.getElementById('tablesList');
const tableModal = document.getElementById('tableModal');
const statsModal = document.getElementById('statsModal');

let currentTableName = '';
let currentPage = 1;

// ===== INITIALIZATION =====
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 AI 37 khởi động...');
    
    // loadTables(); // Tắt API tables (không có backend endpoint)
    checkHealth();
    setupEventListeners();
    autoResizeTextarea();
    // loadStats(); // Tắt stats (không có backend endpoint)
});

// ===== EVENT LISTENERS =====
function setupEventListeners() {
    // Toggle chatbot
    ai37Toggle.addEventListener('click', toggleChatWindow);
    
    // Minimize chatbot
    minimizeBtn.addEventListener('click', toggleChatWindow);
    
    // Clear chat
    clearChatBtn.addEventListener('click', clearChat);
    
    // Send message
    sendBtn.addEventListener('click', sendMessage);
    
    // Stop generation
    stopBtn.addEventListener('click', stopGeneration);
    
    // Enter to send (Shift+Enter for new line)
    ai37Input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // Refresh button
    refreshBtn.addEventListener('click', forceRefresh);
    
    // Stats button
    statsBtn.addEventListener('click', showStats);
    
    // Close modals
    window.addEventListener('click', (e) => {
        if (e.target === tableModal) {
            closeTableModal();
        }
        if (e.target === statsModal) {
            closeStatsModal();
        }
    });
}

// ===== CHATBOT TOGGLE =====
function toggleChatWindow() {
    ai37ChatWindow.classList.toggle('active');
    if (ai37ChatWindow.classList.contains('active')) {
        ai37Input.focus();
    }
}

// ===== CLEAR CHAT =====
function clearChat() {
    const welcome = document.querySelector('.ai37-welcome');
    ai37Messages.innerHTML = '';
    if (welcome) {
        ai37Messages.appendChild(welcome.cloneNode(true));
    }
}

// ===== AUTO-RESIZE TEXTAREA =====
function autoResizeTextarea() {
    ai37Input.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = Math.min(this.scrollHeight, 120) + 'px';
    });
}

// ===== SEND MESSAGE =====
async function sendMessage() {
    const message = ai37Input.value.trim();
    
    if (!message) return;
    
    // Disable input
    ai37Input.disabled = true;
    sendBtn.disabled = true;
    
    // Add user message
    addAI37Message('user', message);
    
    // Clear input
    ai37Input.value = '';
    ai37Input.style.height = 'auto';
    
    // Show typing indicator
    const typingId = showTypingIndicator();
    
    // Show stop button
    stopBtn.style.display = 'flex';
    sendBtn.style.display = 'none';
    
    // Create AbortController for stopping
    currentAbortController = new AbortController();
    
    try {
        // STREAMING MODE - Gõ từng chữ như ChatGPT
        const response = await fetch(`${API_BASE_URL}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                message: message,
                stream: true  // Bật streaming
            }),
            signal: currentAbortController.signal
        });
        
        removeTypingIndicator(typingId);
        
        // Tạo message container cho AI
        const messageDiv = document.createElement('div');
        messageDiv.className = 'ai37-message ai37-message-ai';
        
        const avatarDiv = document.createElement('div');
        avatarDiv.className = 'ai37-message-avatar';
        avatarDiv.textContent = '🤖';
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'ai37-message-content';
        
        const textDiv = document.createElement('div');
        textDiv.className = 'ai37-message-text';
        
        const timeDiv = document.createElement('div');
        timeDiv.className = 'ai37-message-time';
        timeDiv.textContent = new Date().toLocaleTimeString('vi-VN');
        
        contentDiv.appendChild(textDiv);
        contentDiv.appendChild(timeDiv);
        messageDiv.appendChild(avatarDiv);
        messageDiv.appendChild(contentDiv);
        ai37Messages.appendChild(messageDiv);
        
        // Scroll to bottom
        ai37Messages.scrollTop = ai37Messages.scrollHeight;
        
        // Đọc streaming response
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullText = '';
        
        while (true) {
            const {value, done} = await reader.read();
            if (done) break;
            
            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const jsonStr = line.slice(6);
                    
                    try {
                        const data = JSON.parse(jsonStr);
                        
                        if (data.content) {
                            // Thêm content vào text
                            fullText += data.content;
                            textDiv.innerHTML = formatMessageText(fullText);
                            
                            // Auto scroll
                            ai37Messages.scrollTop = ai37Messages.scrollHeight;
                        }
                        
                        if (data.error) {
                            textDiv.innerHTML = `❌ Lỗi: ${data.error}`;
                        }
                        
                        if (data.done) {
                            // Hoàn thành
                            break;
                        }
                    } catch (e) {
                        // Ignore parse errors
                    }
                }
            }
        }
        
    } catch (error) {
        removeTypingIndicator(typingId);
        
        if (error.name === 'AbortError') {
            addAI37Message('ai', '⏹️ Đã dừng tạo câu trả lời.');
        } else {
            console.error('Lỗi:', error);
            addAI37Message('ai', `❌ Lỗi kết nối: ${error.message}`);
        }
    } finally {
        // Re-enable input
        ai37Input.disabled = false;
        sendBtn.disabled = false;
        stopBtn.style.display = 'none';
        sendBtn.style.display = 'flex';
        ai37Input.focus();
        currentAbortController = null;
    }
}

// ===== STOP GENERATION =====
function stopGeneration() {
    if (currentAbortController) {
        currentAbortController.abort();
    }
}

// ===== ADD MESSAGE TO CHAT =====
function addAI37Message(sender, text, sql = null) {
    console.log('Adding message:', sender, text);
    console.log('Text type:', typeof text);
    console.log('Text length:', text ? text.length : 0);
    console.log('Text JSON:', JSON.stringify(text));
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `ai37-message ai37-message-${sender}`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'ai37-message-content';
    
    // Format text
    try {
        console.log('Before formatMessageText:', text);
        const formattedText = formatMessageText(text);
        console.log('After formatMessageText:', formattedText);
        contentDiv.innerHTML = formattedText;
    } catch (error) {
        console.error('Error formatting text:', error);
        console.error('Error details:', error.message, error.stack);
        contentDiv.textContent = text; // Fallback to plain text
    }
    
    // Add SQL block if present
    if (sql) {
        const sqlDiv = document.createElement('div');
        sqlDiv.className = 'ai37-sql-block';
        sqlDiv.innerHTML = `
            <div class="ai37-sql-title">🔍 SQL Query:</div>
            <code>${escapeHtml(sql)}</code>
        `;
        contentDiv.appendChild(sqlDiv);
    }
    
    // Add timestamp
    const timeDiv = document.createElement('div');
    timeDiv.className = 'ai37-message-time';
    timeDiv.textContent = new Date().toLocaleTimeString('vi-VN');
    contentDiv.appendChild(timeDiv);
    
    messageDiv.appendChild(contentDiv);
    ai37Messages.appendChild(messageDiv);
    
    // Scroll to bottom
    ai37Messages.scrollTop = ai37Messages.scrollHeight;
}

// ===== FORMAT MESSAGE TEXT =====
function formatMessageText(text) {
    console.log('Formatting text:', text);
    console.log('Text type in formatMessageText:', typeof text);
    console.log('Text length in formatMessageText:', text ? text.length : 0);
    console.log('Text JSON in formatMessageText:', JSON.stringify(text));
    
    try {
        let formatted = escapeHtml(text);
        console.log('After escapeHtml:', formatted);
    
    // Convert **bold**
    formatted = formatted.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    
        // Convert line breaks
        formatted = formatted.replace(/\n/g, '<br>');
        
        // Convert ```sql blocks
        formatted = formatted.replace(/```sql\s*([\s\S]*?)\s*```/gi, (match, code) => {
            return `<div class="ai37-sql-block">
                <div class="ai37-sql-title">SQL Query:</div>
                <code>${code.trim()}</code>
            </div>`;
        });
        
        // Convert ``` generic code blocks
        formatted = formatted.replace(/```([\s\S]*?)```/g, (match, code) => {
            return `<div class="ai37-sql-block"><code>${code.trim()}</code></div>`;
        });
        
        console.log('Final formatted text:', formatted);
        return formatted;
    } catch (error) {
        console.error('Error in formatMessageText:', error);
        return text; // Fallback
    }
}

// ===== ESCAPE HTML =====
function escapeHtml(text) {
    console.log('Escaping HTML for text:', text);
    console.log('Text type in escapeHtml:', typeof text);
    console.log('Text length in escapeHtml:', text ? text.length : 0);
    console.log('Text JSON in escapeHtml:', JSON.stringify(text));
    
    try {
        const div = document.createElement('div');
        console.log('Created div element');
        div.textContent = text;
        console.log('Set textContent');
        const result = div.innerHTML;
        console.log('Got innerHTML:', result);
        return result;
    } catch (error) {
        console.error('Error escaping HTML:', error);
        console.error('Error details:', error.message, error.stack);
        return text; // Fallback
    }
}

// ===== TYPING INDICATOR =====
function showTypingIndicator() {
    const id = 'typing-' + Date.now();
    const typingDiv = document.createElement('div');
    typingDiv.id = id;
    typingDiv.className = 'ai37-message ai37-message-ai';
    typingDiv.innerHTML = `
        <div class="ai37-message-content">
            <div class="ai37-typing">
                <div class="ai37-typing-dot"></div>
                <div class="ai37-typing-dot"></div>
                <div class="ai37-typing-dot"></div>
            </div>
        </div>
    `;
    ai37Messages.appendChild(typingDiv);
    ai37Messages.scrollTop = ai37Messages.scrollHeight;
    return id;
}

function removeTypingIndicator(id) {
    const element = document.getElementById(id);
    if (element) {
        element.remove();
    }
}

// ===== LOAD TABLES =====
async function loadTables() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/tables`);
        const data = await response.json();
        
        if (data.tables && data.tables.length > 0) {
            displayTables(data.tables);
            loadTableStats(data.tables);
            
            // Update total tables count
            document.getElementById('totalTables').textContent = data.tables.length;
        } else {
            tablesList.innerHTML = '<div class="loading">Không tìm thấy bảng nào</div>';
        }
    } catch (error) {
        console.error('Lỗi load tables:', error);
        tablesList.innerHTML = '<div class="loading">❌ Lỗi tải dữ liệu</div>';
    }
}

function displayTables(tables) {
    tablesList.innerHTML = '';
    
    tables.forEach(tableName => {
        const tableDiv = document.createElement('div');
        tableDiv.className = 'table-item';
        tableDiv.innerHTML = `
            <div class="table-item-name">📊 ${tableName}</div>
            <div class="table-item-count" id="count-${tableName}">Đang tải...</div>
        `;
        tableDiv.onclick = () => showTableDetails(tableName, 1);
        tablesList.appendChild(tableDiv);
    });
}

async function loadTableStats(tables) {
    let totalRecords = 0;
    
    for (const tableName of tables) {
        try {
            const response = await fetch(`${API_BASE_URL}/api/query`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: `SELECT COUNT(*) as count FROM ${tableName}` })
            });
            
            const data = await response.json();
            
            if (data.success && data.rows && data.rows.length > 0) {
                const count = data.rows[0].count;
                totalRecords += count;
                
                const countElement = document.getElementById(`count-${tableName}`);
                if (countElement) {
                    countElement.textContent = `${count.toLocaleString('vi-VN')} bản ghi`;
                }
            }
        } catch (error) {
            console.error('Lỗi load stats cho table:', error);
        }
    }
    
    // Update total records count
    document.getElementById('totalRecords').textContent = totalRecords.toLocaleString('vi-VN');
}

// ===== SHOW TABLE DETAILS WITH PAGINATION =====
async function showTableDetails(tableName, page = 1) {
    currentTableName = tableName;
    currentPage = page;
    
    const modalTableName = document.getElementById('modalTableName');
    const tableContent = document.getElementById('tableContent');
    const paginationControls = document.getElementById('paginationControls');
    
    modalTableName.textContent = `📊 Bảng: ${tableName}`;
    tableContent.innerHTML = '<div class="loading">Đang tải dữ liệu...</div>';
    paginationControls.innerHTML = '';
    
    tableModal.style.display = 'block';
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/table/${tableName}?page=${page}&per_page=100`);
        const data = await response.json();
        
        if (data.error) {
            tableContent.innerHTML = `<div class="loading">❌ ${data.error}</div>`;
            return;
        }
        
        if (data.rows.length === 0) {
            tableContent.innerHTML = '<div class="loading">Bảng không có dữ liệu</div>';
            return;
        }
        
        // Create table
        let html = '<table><thead><tr>';
        
        data.columns.forEach(col => {
            html += `<th>${col.name} <span style="color: #94a3b8; font-weight: normal;">(${col.type})</span></th>`;
        });
        html += '</tr></thead><tbody>';
        
        data.rows.forEach(row => {
            html += '<tr>';
            data.columns.forEach(col => {
                const value = row[col.name];
                html += `<td>${value !== null && value !== undefined ? escapeHtml(String(value)) : '<em style="color: #64748b;">NULL</em>'}</td>`;
            });
            html += '</tr>';
        });
        
        html += '</tbody></table>';
        tableContent.innerHTML = html;
        
        // Create pagination
        if (data.pagination && data.pagination.total_pages > 1) {
            createPagination(data.pagination);
        }
        
    } catch (error) {
        console.error('Lỗi:', error);
        tableContent.innerHTML = `<div class="loading">❌ Lỗi: ${error.message}</div>`;
    }
}

function createPagination(pagination) {
    const paginationControls = document.getElementById('paginationControls');
    let html = '';
    
    // Get current page (support both old and new format)
    const currentPage = pagination.current_page || pagination.page || 1;
    const totalPages = pagination.total_pages || 1;
    
    // Previous button
    html += `<button class="pagination-btn" ${currentPage === 1 ? 'disabled' : ''} 
             onclick="showTableDetails('${currentTableName}', ${currentPage - 1})">
             ← Trước
             </button>`;
    
    // Page numbers
    const startPage = Math.max(1, currentPage - 2);
    const endPage = Math.min(totalPages, currentPage + 2);
    
    if (startPage > 1) {
        html += `<button class="pagination-btn" onclick="showTableDetails('${currentTableName}', 1)">1</button>`;
        if (startPage > 2) {
            html += `<span class="pagination-info">...</span>`;
        }
    }
    
    for (let i = startPage; i <= endPage; i++) {
        html += `<button class="pagination-btn ${i === currentPage ? 'active' : ''}" 
                 onclick="showTableDetails('${currentTableName}', ${i})">
                 ${i}
                 </button>`;
    }
    
    if (endPage < totalPages) {
        if (endPage < totalPages - 1) {
            html += `<span class="pagination-info">...</span>`;
        }
        html += `<button class="pagination-btn" onclick="showTableDetails('${currentTableName}', ${totalPages})">
                 ${totalPages}
                 </button>`;
    }
    
    // Next button
    html += `<button class="pagination-btn" ${currentPage === totalPages ? 'disabled' : ''} 
             onclick="showTableDetails('${currentTableName}', ${currentPage + 1})">
             Sau →
             </button>`;
    
    // Info
    const totalRows = pagination.total_rows || 0;
    html += `<span class="pagination-info">
             Trang ${currentPage} / ${totalPages} 
             (${totalRows.toLocaleString('vi-VN')} bản ghi)
             </span>`;
    
    paginationControls.innerHTML = html;
}

function closeTableModal() {
    tableModal.style.display = 'none';
}

window.closeTableModal = closeTableModal;

// ===== STATS MODAL =====
async function showStats() {
    const statsContent = document.getElementById('statsContent');
    
    statsContent.innerHTML = '<div class="loading">Đang tải thống kê...</div>';
    statsModal.style.display = 'block';
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/stats`);
        const data = await response.json();
        
        if (data.error) {
            statsContent.innerHTML = `<div class="loading">❌ ${data.error}</div>`;
            return;
        }
        
        let html = '';
        
        if (data.last_updated) {
            html += `<div class="stat-card" style="grid-column: 1 / -1;">
                <div class="stat-card-title">🕒 Cập nhật lần cuối</div>
                <div class="stat-card-value" style="font-size: 1.2rem;">${data.last_updated}</div>
            </div>`;
        }
        
        if (data.stats && data.stats.tables) {
            const tables = data.stats.tables;
            let totalRecords = 0;
            
            for (const [tableName, tableData] of Object.entries(tables)) {
                totalRecords += tableData.count || 0;
                
                html += `<div class="stat-card">
                    <div class="stat-card-title">📊 ${tableName}</div>
                    <div class="stat-card-value">${(tableData.count || 0).toLocaleString('vi-VN')}</div>
                    <div style="margin-top: 0.5rem; color: #64748b; font-size: 0.85rem;">bản ghi</div>
                </div>`;
            }
            
            html = `<div class="stat-card" style="background: linear-gradient(135deg, #4f46e5, #6366f1); color: white; grid-column: 1 / -1;">
                <div class="stat-card-title" style="color: #e0e7ff;">📈 TỔNG SỐ BẢN GHI</div>
                <div class="stat-card-value" style="color: white;">${totalRecords.toLocaleString('vi-VN')}</div>
            </div>` + html;
        }
        
        if (!html) {
            html = '<div class="loading">Không có thống kê</div>';
        }
        
        statsContent.innerHTML = html;
        
    } catch (error) {
        console.error('Lỗi:', error);
        statsContent.innerHTML = `<div class="loading">❌ Lỗi: ${error.message}</div>`;
    }
}

function closeStatsModal() {
    statsModal.style.display = 'none';
}

window.closeStatsModal = closeStatsModal;

// ===== FORCE REFRESH =====
async function forceRefresh() {
    const refreshIcon = document.getElementById('refreshIcon');
    refreshIcon.style.animation = 'spin 1s linear infinite';
    refreshBtn.disabled = true;
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/refresh`, {
            method: 'POST'
        });
        
        const data = await response.json();
        
        if (data.success) {
            await loadTables();
            addAI37Message('ai', '✅ Đã làm mới dữ liệu thành công!');
        } else {
            addAI37Message('ai', '❌ Lỗi làm mới dữ liệu');
        }
        
    } catch (error) {
        console.error('Lỗi:', error);
        addAI37Message('ai', `❌ Lỗi: ${error.message}`);
    } finally {
        refreshIcon.style.animation = '';
        refreshBtn.disabled = false;
    }
}

// ===== HEALTH CHECK =====
async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        
        updateAI37Status(data.status === 'healthy' ? 'online' : 'warning');
        
    } catch (error) {
        console.error('Lỗi health check:', error);
        updateAI37Status('offline');
    }
}

function updateAI37Status(status) {
    const ai37Status = document.getElementById('ai37Status');
    const statusDot = ai37Status.querySelector('.status-dot');
    
    if (status === 'online') {
        statusDot.style.background = '#10b981';
        ai37Status.innerHTML = '<span class="status-dot"></span> Trực tuyến';
        ai37Status.className = 'ai37-status ai37-status-online';
    } else if (status === 'warning') {
        statusDot.style.background = '#10b981';
        ai37Status.innerHTML = '<span class="status-dot"></span> Trực tuyến';
        ai37Status.className = 'ai37-status ai37-status-online';
    } else {
        statusDot.style.background = '#ef4444';
        ai37Status.innerHTML = '<span class="status-dot"></span> Ngoại tuyến';
        ai37Status.className = 'ai37-status';
    }
}

// ===== LOAD STATS ON START =====
async function loadStats() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/stats`);
        const data = await response.json();
        
        if (data.stats && data.stats.tables) {
            let totalRecords = 0;
            for (const tableData of Object.values(data.stats.tables)) {
                totalRecords += tableData.count || 0;
            }
            document.getElementById('totalRecords').textContent = totalRecords.toLocaleString('vi-VN');
        }
    } catch (error) {
        console.error('Lỗi load stats:', error);
    }
}

// ===== AUTO CHECK HEALTH =====
setInterval(checkHealth, 30000);

console.log('🤖 AI 37 sẵn sàng!');
