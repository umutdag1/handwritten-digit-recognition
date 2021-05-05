
let drawingCanvas = document.getElementById('input-canvas');
drawingCanvas.width = 84;
drawingCanvas.height = 84;
let drawingContext = drawingCanvas.getContext('2d');

let feedCanvas = document.getElementById('feed-image');
feedCanvas.width = 28;
feedCanvas.height = 28;
let feedContext = feedCanvas.getContext('2d');

function getMousePos(evt) {
    var rect = drawingCanvas.getBoundingClientRect();
    return {
        x: evt.clientX - rect.left,
        y: evt.clientY - rect.top
    };
}
var mouseIsDown = false;
drawingCanvas.onmousedown = ev => {
    mouseIsDown = true;
}
drawingCanvas.onmouseup = ev => {
    mouseIsDown = false;
}
drawingCanvas.onmousemove = e => {
    if (!mouseIsDown) return;

    drawingContext.beginPath();
    drawingContext.arc(getMousePos(e).x, getMousePos(e).y, 3, 0, 2 * Math.PI, false);
    drawingContext.fillStyle = '#000000';
    drawingContext.fill();

    return false;
}
drawingCanvas.addEventListener('mouseout', function () {
    mouseIsDown = false;
}, false);

document.getElementById('recognise-button').onclick = function () {
    feedContext.drawImage(drawingCanvas, 0, 0, feedCanvas.width, feedCanvas.height);
    let rgbaArray = feedContext.getImageData(0, 0, feedCanvas.width, feedCanvas.height).data;
    let pixels = [];
    for (let i = 3; i < rgbaArray.length; i = i + 4)
        pixels.push(rgbaArray[i]);
    console.log(pixels)
    document.getElementById('output-region-text').innerText = "Waiting";
    $.ajax({
        method : "POST",
        url: "http://127.0.0.1:5000/api/getResult",
        contentType: 'application/json; charset=utf-8',
        data : JSON.stringify({pixels : pixels})
    }).done(function (res) {
        let predicted_result = JSON.parse(res)
        document.getElementById('output-region-text').innerText = predicted_result;
    });
};

document.getElementById('clear-button').onclick = function () {
    drawingContext.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);
    feedContext.clearRect(0, 0, feedCanvas.width, feedCanvas.height);
    document.getElementById('output-region-text').innerText = '';
};

