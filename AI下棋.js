// 安卓模拟器 分辨率1600x900 适用
var left = 759
var right = 1529
var down = 883
var up = 115
var columns = 8;
var rows = 8;

var cellWidth = (right - left) / columns;
var cellHeight = (down - up) / rows;

var centers = [];

for (var i = 0; i < rows; i++) {
    for (var j = 0; j < columns; j++) {
        var centerX = left + j * cellWidth + cellWidth / 2;
        var centerY = up + i * cellHeight + cellHeight / 2;
        centers.push({x: centerX, y: centerY});
    }
}

// 修复1: 使用对象数组而不是Python风格的元组
const start_qi1 = [
    {x: 730, y: 454},  // 激活屏幕
    {x: 1528, y: 241},  // 打开右侧栏目
    {x: 1439, y: 247}, // 打开包裹
    {x: 1350, y: 195},  // 点击搜索
    {x: 935, y: 201}  // 输入框
];

// 中间输入 无常棋 到 729，63 
const start_qi2 = [
    {x: 1466, y: 62}, // 确定
    {x: 1350, y: 195}, // 点击搜索
    {x: 840, y: 342}, // 点击第一个无常棋
    {x: 782, y: 236}, // 邀请对弈
    {x: 371, y: 213}, // 世界频道
    {x: 593, y: 216}, // 当前频道
    {x: 374, y: 283}, // 帮派频道
    {x: 590, y: 280},  // 门派频道
    //{x: 1529, y: 42},  // 右上角关闭
    //{x: 1529, y: 42} // 右上角关闭
];

if (!requestScreenCapture(false)) {
    toast("请授权截图权限！");
    exit();
}

function getColorType(color) {
    var r = (color >> 16) & 0xff;
    var g = (color >> 8) & 0xff;
    var b = color & 0xff;

    var maxDiff = Math.max(Math.abs(r - g), Math.abs(r - b), Math.abs(g - b));

    // 判断是否接近黑色
    if (r < 60 && g < 60 && b < 60 && maxDiff < 10) {  
        return "B";  // Black
    }

    if (r > 205 && g > 205 && b > 205) {
        return "W";  // White
    }

    return "O";  // Other
}

function read_img(img){
    var grid = Array(8).fill().map(() => Array(8).fill(''));

    for (var i = 0; i < 8; i++) {
        for (var j = 0; j < 8; j++) {
            var point = centers[i * 8 + j];
            var color = images.pixel(img, point.x, point.y);
            var colorType = getColorType(color);

            //log(point.x+" "+point.y+" "+colorType);
            grid[i][j] = colorType;
        }
    }
    var result = grid.map(row => row.join(' ')).join('\n');
    log('\n'+result);

    return [grid,result];
}

// 将棋盘状态转换为API需要的格式
function convertBoardToNumbers(grid) {
    var board = [];
    for (var i = 0; i < 8; i++) {
        var row = [];
        for (var j = 0; j < 8; j++) {
            if (grid[i][j] === 'B') {
                row.push(1);   // 黑子
            } else if (grid[i][j] === 'W') {
                row.push(-1);  // 白子
            } else {
                row.push(0);   // 空位
            }
        }
        board.push(row);
    }
    return board;
}

// 调用AI API获取最佳落子位置
function getAIMove(board, playerColor) {
    try {
        var apiUrl = "http://localhost:8082/api/ai_move";
        
        var payload = {
            "board": board,
            "player": playerColor  // 1表示黑子(AI执黑)，-1表示白子(AI执白)
        };
        
        toast("正在请求AI思考...");
        log("发送数据: " + JSON.stringify(payload));
        
        var response = http.postJson(apiUrl, payload, {
            headers: {
                "Content-Type": "application/json"
            },
            timeout: 15000  // 15秒超时
        });
        
        if (response.statusCode == 200) {
            var result = response.body.json();
            toast("AI返回结果: " + JSON.stringify(result));
            
            if (result.success && result.ai_move) {
                return {
                    row: result.ai_move.row,
                    col: result.ai_move.col
                };
            } else {
                log("AI返回错误: " + result.message);
                return null;
            }
        } else {
            log("HTTP请求失败，状态码: " + response.statusCode);
            return null;
        }
        
    } catch (e) {
        toast("调用AI API出错: " + e);
        return null;
    }
}

function randomClickPos(x, y, maxOffset) {
    let offsetX = Math.floor(Math.random() * (2 * maxOffset + 1)) - maxOffset;
    let offsetY = Math.floor(Math.random() * (2 * maxOffset + 1)) - maxOffset;
    log("✔️ 点击坐标: " + Math.round(x + offsetX) +" "+Math.round(y + offsetY));
    return {
        x: x + offsetX,
        y: y + offsetY
    };
}

function randomClick(pos, maxOffset) {
    // 修复2: 确保pos是对象格式
    if (typeof pos === 'object' && pos.x !== undefined && pos.y !== undefined) {
        let click_pos = randomClickPos(pos.x, pos.y, maxOffset);
        log("点击" + click_pos.x + " " + click_pos.y);
        click(click_pos.x, click_pos.y);
    } else {
        log("❌ randomClick: 传入的坐标格式不正确");
    }
}

function randomSleep(minMs, maxMs) {
    let delay = minMs + Math.random() * (maxMs - minMs);
    sleep(delay);
}

function findImageCenter(template, screenshot, threshold) {
    let offsetx = 0.5, offsety = 0.5;
    // 3. 搜索图像
    let result = images.findImage(screenshot, template, {
        threshold: threshold
    });

    let center = null;
    if (result) {
        center = {
            x: result.x + template.getWidth() * offsetx,
            y: result.y + template.getHeight() * offsety
        };
        
        log("✅ 找到坐标 " + " @ (" + Math.round(center.x) + ", " + Math.round(center.y) + ")");
       
    } else{
        log("🔍 未找到 ");
    }

    return center;
}

function clickImage(imagetemplate, screen, threshold) {
    let pos = findImageCenter(imagetemplate, screen, threshold);
    if (pos) {
        randomClick(pos, 10);
        return true;
    }
    return false;
}

function getScreen() {
    // 使用Auto.js内置的截图功能，无需root权限
    var img = captureScreen();
    if (img && img.width == 1600 && img.height == 900) {
        // 可选：如果你想保存一份到文件用于调试，可以取消下面这行注释
        // images.save(img, "/storage/emulated/0/Pictures/sc.png");
        return img;
    } else {
        log("截图失败!");
        toast("截图失败，请检查Auto.js权限");
        return null;
    }
}

// 该位置是白色，说明正在下棋
function is_white(start_w, start_h, img) {
    var color = images.pixel(img, start_w, start_h);
    var r = (color >> 16) & 0xff;
    var g = (color >> 8) & 0xff;
    var b = color & 0xff;
    if (r > 205 && g > 205 && b > 205) {
        return true;  // White
    }
    return false;
}

// 修复3: 修改isMyTurn函数，不要在这里回收模板图片
function isMyTurn(img, templateImg) {
    // 修复4: region应该是数组格式
    var region = [1068, 91, 1185 - 1068, 123 - 91]; // [x, y, width, height]
    
    try {
        // 在指定区域中查找模板图片
        var match = images.findImage(img, templateImg);
        
        // 如果找到匹配，match不为null
        if (match) {
            log("检测到 '己方回合' 提示");
            return true;
        } else {
            // log("未检测到 '己方回合' 提示"); // 取消注释可以每次都打印
            return false;
        }
    } catch (e) {
        log("图像匹配出错: " + e);
        return false;
    }
    // 修复5: 不要在这里回收模板图片，因为后续可能还需要使用
}

// 模板图片路径
var templatePath = files.path("./己方回合.png"); // 获取脚本同目录下的图片路径
var qipanPath = files.path("./棋盘.png"); 
var guanbiPath = files.path("./关闭.png"); 
// 读取模板图片
var templatejifang = images.read(templatePath);
var templateqipan = images.read(qipanPath);
var templateguanbi = images.read(guanbiPath);

if (!templatejifang) {
    log("错误：无法读取模板图片 '己方回合.png'");
    exit();
}

let timecount = 0;
while (true) {
    sleep(1000);
    // 1. 截图
    let sc_img = getScreen();
    if (!sc_img) {
        log("⚠️ 截图失败");
        sleep(1000);
        continue;
    }
    let qipan = findImageCenter(templateqipan,sc_img,0.8);
    let guanbi = findImageCenter(templateguanbi,sc_img,0.7);

    // 如果正在下棋，就检测是否是自己的回合
    if (is_white(825, 54, sc_img)) {
        log("检测棋盘已打开");
        if (isMyTurn(sc_img, templatejifang)) {
            log("检测轮到己方回合 AI 下棋");
            let re = read_img(sc_img);
            // 修复9: 正确访问返回值的属性
            log("棋盘网格:", re[0]);
            log("棋盘文本:", re[1]);
            let board_numbers = convertBoardToNumbers(re[0]);
            let aiMove = getAIMove(board_numbers, 1); // 1 表示自己拿黑子
            if (aiMove) {
                        var x = aiMove.row;
                        var y = aiMove.col;
                        var point = centers[x * 8 + y];
                        
                        log("AI决定落子位置: (" + x + ", " + y + ")");
                        log("点击坐标: (" + point.x + ", " + point.y + ")");
                        
                        click(point.x, point.y);
                        sleep(2000);  // 等待落子动画
                    } else {
                        log("AI未返回有效落子，跳过");
                        sleep(3000);
                }
                
            }
    } else if(guanbi){
        log("找到关闭");
        randomClick(guanbi, 10);
        randomSleep(800,1000);
    }
    else if(qipan){
        log("找到棋盘");
        // 多次进入不了棋盘，重新邀请
        if(timecount<23){
            randomClick(qipan, 10);
            randomSleep(800,1000);
        }
        timecount = timecount +1;      
    }
    else{
        log("开始重新启动进入下棋的流程");
        // 开始重新启动进入下棋的流程
        // 修复8: 使用正确的循环语法
        for (let i = 0; i < start_qi1.length; i++) {
            randomClick(start_qi1[i], 10);
            randomSleep(1000, 1200);
        }
        
        // 输入文字
        input("无常棋");
        randomSleep(500, 800);
        
        for (let i = 0; i < start_qi2.length; i++) {
            randomClick(start_qi2[i], 10);
            randomSleep(1000, 1200);
        }
        timecount = 0;
    }
    
    // 回收截图
    sc_img.recycle();
    sleep(1000);
}
