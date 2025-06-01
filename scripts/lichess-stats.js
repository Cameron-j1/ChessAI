async function fetchBotStats() {
    const botUsername = 'AI_in_robo_engine';
    const statsContainer = document.querySelector('.lichess-embed');
    
    try {
        // Fetch user data
        const response = await fetch(`https://lichess.org/api/user/${botUsername}`, {
            headers: {
                'Accept': 'application/json'
            }
        });

        if (!response.ok) {
            throw new Error('Failed to fetch bot statistics');
        }

        const data = await response.json();
        
        // Create stats HTML
        const statsHTML = `
            <div class="bot-rating-card">
                <div class="rating-header">
                    <img src="${data.profile?.avatar || 'https://lichess1.org/assets/images/placeholder.png'}" 
                         alt="Bot Avatar" class="bot-avatar">
                    <h4>${data.username}</h4>
                </div>
                <div class="rating-details">
                    <div class="rating-item">
                        <span class="label">Rapid Rating</span>
                        <span class="value">${data.perfs.rapid?.rating || 'Unrated'}</span>
                    </div>
                    <div class="rating-item">
                        <span class="label">Blitz Rating</span>
                        <span class="value">${data.perfs.blitz?.rating || 'Unrated'}</span>
                    </div>
                    <div class="rating-item">
                        <span class="label">Classical Rating</span>
                        <span class="value">${data.perfs.classical?.rating || 'Unrated'}</span>
                    </div>
                    <div class="rating-item">
                        <span class="label">Games Played</span>
                        <span class="value">${data.count.all || 0}</span>
                    </div>
                </div>
            </div>
        `;

        // Update the container
        statsContainer.innerHTML = statsHTML;
        statsContainer.classList.add('loaded');

    } catch (error) {
        console.error('Error fetching bot statistics:', error);
        statsContainer.innerHTML = `
            <div class="error-message">
                <p>Unable to load bot statistics at the moment.</p>
                <a href="https://lichess.org/@/AI_in_robo_engine" target="_blank" class="lichess-profile-link">
                    <i class="fas fa-chess-knight"></i>
                    View Profile on Lichess
                </a>
            </div>
        `;
    }
}

// Call the function when the page loads
document.addEventListener('DOMContentLoaded', fetchBotStats); 