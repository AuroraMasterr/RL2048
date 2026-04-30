// 演示状态
let isPlaying = false;
let isRecording = false;
let gameInterval = null;
let currentSpeed = 500;
let steps = 0;
let episodes = 0;
let recordedFrames = [];
let mediaRecorder = null;
let recordedChunks = [];

// 动作映射
const actionNames = ['上', '下', '左', '右'];

// 初始化
document.addEventListener('DOMContentLoaded', function() {
    initEventListeners();
    resetGame();
});

// 事件监听
function initEventListeners() {
    document.getElementById('startBtn').addEventListener('click', startGame);
    document.getElementById('pauseBtn').addEventListener('click', pauseGame);
    document.getElementById('resetBtn').addEventListener('click', resetGame);
    document.getElementById('recordBtn').addEventListener('click', toggleRecording);
    
    const speedSlider = document.getElementById('speedSlider');
    speedSlider.addEventListener('input', function() {
        currentSpeed = parseInt(speedSlider.value);
        document.getElementById('speedValue').textContent = currentSpeed + 'ms';
        
        // 如果正在播放，更新速度
        if (isPlaying) {
            pauseGame();
            startGame();
        }
    });
}

// 开始游戏
function startGame() {
    if (isPlaying) return;
    
    isPlaying = true;
    document.getElementById('startBtn').disabled = true;
    document.getElementById('pauseBtn').disabled = false;
    
    playGame();
}

// 暂停游戏
function pauseGame() {
    isPlaying = false;
    document.getElementById('startBtn').disabled = false;
    document.getElementById('pauseBtn').disabled = true;
    
    if (gameInterval) {
        clearInterval(gameInterval);
        gameInterval = null;
    }
}

// 重置游戏
async function resetGame() {
    pauseGame();
    steps = 0;
    recordedFrames = [];
    
    try {
        const response = await fetch('/api/reset', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({game_id: 'agent'})
        });
        
        const data = await response.json();
        updateBoard(data.board);
        updateScore(data.score, data.max_tile);
    } catch (error) {
        console.error('重置游戏失败:', error);
    }
}

// 播放游戏
function playGame() {
    if (!isPlaying) return;
    
    gameInterval = setInterval(agentMove, currentSpeed);
}

// 智能体移动
async function agentMove() {
    if (!isPlaying) return;
    
    try {
        const response = await fetch('/api/agent/step', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({game_id: 'agent'})
        });
        
        const data = await response.json();
        
        if (data.error) {
            console.error('智能体错误:', data.error);
            pauseGame();
            return;
        }
        
        updateBoard(data.board);
        updateScore(data.score, data.max_tile);
        
        // 显示动作
        const actionName = actionNames[data.action] || '未知';
        document.getElementById('actionLabel').textContent = `上一步: ${actionName} (+${data.reward}分)`;
        
        steps++;
        document.getElementById('steps').textContent = steps;
        
        // 游戏结束
        if (data.done) {
            episodes++;
            document.getElementById('episodes').textContent = episodes;
            console.log(`游戏${episodes}结束! 得分: ${data.score}, 最大块: ${data.max_tile}`);
            
            // 自动重新开始
            setTimeout(async function() {
                await resetGame();
                if (isPlaying) {
                    startGame();
                }
            }, 2000);
        }
    } catch (error) {
        console.error('智能体移动失败:', error);
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
function updateScore(score, maxTile) {
    document.getElementById('score').textContent = score;
    document.getElementById('maxTile').textContent = maxTile;
}

// 切换录制
async function toggleRecording() {
    const recordBtn = document.getElementById('recordBtn');
    
    if (!isRecording) {
        // 开始录制
        try {
            const stream = await navigator.mediaDevices.getDisplayMedia({
                video: {
                    mediaSource: "browser",
                    cursor: "motion",
                    width: 1280,
                    height: 720
                },
                audio: false
            });
            
            // 使用 MP4 格式
            mediaRecorder = new MediaRecorder(stream, {
                mimeType: 'video/mp4'
            });
            
            mediaRecorder.ondataavailable = function(event) {
                if (event.data.size > 0) {
                    recordedChunks.push(event.data);
                }
            };
            
            mediaRecorder.onstop = function() {
                const blob = new Blob(recordedChunks, { type: 'video/mp4' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.style.display = 'none';
                a.href = url;
                a.download = '2048-ppo-agent-' + new Date().toISOString().slice(0, 10) + '.mp4';
                document.body.appendChild(a);
                a.click();
                URL.revokeObjectURL(url);
                recordedChunks = [];
            };
            
            mediaRecorder.start();
            isRecording = true;
            recordBtn.textContent = '⏹️ 停止录制';
            recordBtn.style.backgroundColor = '#f65e3b';
            
        } catch (error) {
            console.error('屏幕录制失败:', error);
            alert('录制功能需要浏览器支持，并且需要授予屏幕录制权限');
        }
        
    } else {
        // 停止录制
        if (mediaRecorder) {
            mediaRecorder.stop();
            mediaRecorder.stream.getTracks().forEach(track => track.stop());
        }
        isRecording = false;
        recordBtn.textContent = '📹 开始录制';
        recordBtn.style.backgroundColor = '#8f7a66';
    }
}