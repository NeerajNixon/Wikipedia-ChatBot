async function ingestPage() {
    const pageInput = document.getElementById('page-input').value;
    try {
        const response = await fetch('http://localhost:8000/ingest', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ page: pageInput }),
        });
        const data = await response.json();
        if (data.status === 'success') {
            document.getElementById('chat-section').style.display = 'block';
            document.getElementById('chat-history').innerHTML = ''; // Clear chat history in UI
        } else {
            alert('Error ingesting page: ' + data.error);
        }
    } catch (error) {
        console.error('Error:', error);
    }
}

async function askQuestion() {
    const questionInput = document.getElementById('question-input').value;
    const chatHistoryElement = document.getElementById('chat-history');
    const chatHistory = [];

    chatHistoryElement.querySelectorAll('.chat-message').forEach((message) => {
        const userMessage = message.classList.contains('user');
        const text = message.innerText;
        chatHistory.push([userMessage ? 'user' : 'ai', text]);
    });

    try {
        const response = await fetch('http://localhost:8000/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question: questionInput, chat_history: chatHistory }),
        });
        const data = await response.json();
        if (data.response) {
            chatHistoryElement.innerHTML += `<div class="chat-message user">${questionInput}</div>`;
            chatHistoryElement.innerHTML += `<div class="chat-message ai">${data.response}</div>`;
            document.getElementById('question-input').value = '';
            chatHistoryElement.scrollTop = chatHistoryElement.scrollHeight;
        } else {
            alert('Error getting response: ' + data.error);
        }
    } catch (error) {
        console.error('Error:', error);
    }
}
