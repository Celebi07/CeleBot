const chatbox = document.getElementById('chatbox');
const form = document.getElementById('translate-form');
const input = document.getElementById('english-input');

form.addEventListener('submit', (event) => {
    event.preventDefault();
    const userMessage = input.value;

    // Display user message
    displayMessage(userMessage, 'user-message');

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ data: userMessage }) // Ensure the key 'data' matches the Flask endpoint
    })
    .then(response => response.json())
    .then(data => {
        const translatedSentence = data.output;

        // Display bot message with bot-message class
        displayMessage(translatedSentence, 'bot-message');

        // Scroll to the bottom of the chatbox after adding a new message
        autoScroll();
    })
    .catch(error => {
        console.error('Error:', error);
    });

    // Clear input field
    input.value = '';
});

function displayMessage(message, className) {
    const messageElement = document.createElement('div');
    messageElement.classList.add('message');
    messageElement.innerHTML = `<p class="${className}">${message}</p>`;
    chatbox.appendChild(messageElement);
}

function autoScroll() {
    // Scroll to the bottom of the chatbox
    chatbox.scrollTop = chatbox.scrollHeight;
}
