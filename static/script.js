// 游戏状态
let gameId = 'default';
let bestScore = localStorage.getItem('bestScore') || 0;
document.getElementById('best-score').textContent = bestScore;

// 初始化游戏
function initGame() {
    resetGame();
    setupEventListeners();
}

// 重置游戏
async function resetGame() {
    try {
        const response = await fetch('/api/reset', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ game_id: gameId })
        });
        
        const data = await response.json();
        updateBoard(data.board);
        updateScore(data.score);
    } catch (error) {
        console.error('Error resetting game:', error);
    }
}

// 更新棋盘
function updateBoard(board) {
    const gameBoard = document.getElementById('game-board');
    gameBoard.innerHTML = '';
    
    for (let i = 0; i < 4; i++) {
        for (let j = 0; j < 4; j++) {
            const tile = document.createElement('div');
            tile.className = 'tile';
            
            const value = board[i][j];
            if (value > 0) {
                tile.className += ` tile-${value}`;
                tile.textContent = value;
            }
            
            gameBoard.appendChild(tile);
        }
    }
}

// 更新分数
function updateScore(score) {
    document.getElementById('score').textContent = score;
    
    if (score > bestScore) {
        bestScore = score;
        document.getElementById('best-score').textContent = bestScore;
        localStorage.setItem('bestScore', bestScore);
    }
}

// 执行动作
async function makeMove(action) {
    try {
        console.log('Making move:', action); // 调试信息
        
        const response = await fetch('/api/step', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                game_id: gameId,
                action: action 
            })
        });
        
        const data = await response.json();
        console.log('Response from server:', data); // 调试信息
        
        updateBoard(data.board);
        updateScore(data.score);
        
        if (data.done) {
            alert(`游戏结束！最终得分：${data.score}`);
        }
    } catch (error) {
        console.error('Error making move:', error);
    }
}

// 设置事件监听器
function setupEventListeners() {
    // 键盘事件
    document.addEventListener('keydown', function(e) {
        console.log('Key pressed:', e.key); // 调试信息
        
        switch (e.key) {
            case 'ArrowUp':
            case 'Up':
                e.preventDefault();
                makeMove(0); // 上
                break;
            case 'ArrowDown':
            case 'Down':
                e.preventDefault();
                makeMove(1); // 下
                break;
            case 'ArrowLeft':
            case 'Left':
                e.preventDefault();
                makeMove(2); // 左
                break;
            case 'ArrowRight':
            case 'Right':
                e.preventDefault();
                makeMove(3); // 右
                break;
            case 'w':
            case 'W':
                e.preventDefault();
                makeMove(0); // 上 (W)
                break;
            case 's':
            case 'S':
                e.preventDefault();
                makeMove(1); // 下 (S)
                break;
            case 'a':
            case 'A':
                e.preventDefault();
                makeMove(2); // 左 (A)
                break;
            case 'd':
            case 'D':
                e.preventDefault();
                makeMove(3); // 右 (D)
                break;
        }
    });
    
    // 触摸事件
    let touchStartX = 0;
    let touchStartY = 0;
    
    document.addEventListener('touchstart', function(e) {
        touchStartX = e.touches[0].clientX;
        touchStartY = e.touches[0].clientY;
    });
    
    document.addEventListener('touchend', function(e) {
        if (!touchStartX || !touchStartY) return;
        
        const touchEndX = e.changedTouches[0].clientX;
        const touchEndY = e.changedTouches[0].clientY;
        
        const dx = touchEndX - touchStartX;
        const dy = touchEndY - touchStartY;
        
        if (Math.abs(dx) > Math.abs(dy)) {
            // 水平移动
            if (Math.abs(dx) > 50) {
                if (dx > 0) {
                    makeMove(3); // 右
                } else {
                    makeMove(2); // 左
                }
            }
        } else {
            // 垂直移动
            if (Math.abs(dy) > 50) {
                if (dy > 0) {
                    makeMove(1); // 下
                } else {
                    makeMove(0); // 上
                }
            }
        }
        
        touchStartX = 0;
        touchStartY = 0;
    });
}

// 启动游戏
window.addEventListener('DOMContentLoaded', initGame);
