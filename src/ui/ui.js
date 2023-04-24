
// Get the input image and canvas elements
const upload = document.getElementById('upload');
upload.onclick = uploadImage();
const uploadLabel = document.getElementById('uploadLabel');

const canvas = document.getElementById('canvas');
const context = canvas.getContext('2d');


const clear = document.getElementById('clear');
const colorizeBtn = document.getElementById('colorize');


const selectedColor = document.getElementById('selectedColor');

const outputImage = document.getElementById('outputImage');

function calculateAspectRatioFit(srcWidth, srcHeight, maxWidth, maxHeight) {

    var ratio = Math.min(maxWidth / srcWidth, maxHeight / srcHeight);

    return { width: srcWidth*ratio, height: srcHeight*ratio };
 }

// When an image is selected, display it on the canvas
function uploadImage() {
upload.addEventListener('change', function() {
  const file = upload.files[0];
  const reader = new FileReader();
  reader.onload = function(event) {
    const image = new Image();
    image.onload = function() {
      canvas.width = 400;
      canvas.height = 350;

    var ratio = calculateAspectRatioFit(image.width, image.height, canvas.width, canvas.height);
    var xOffset = ratio.width < canvas.width ? ((canvas.width - ratio.width) / 2) : 0;
    var yOffset = ratio.height < canvas.height ? ((canvas.height - ratio.height) / 2) : 0;

      context.drawImage(image, xOffset, yOffset, ratio.width, ratio.height);
      colorPalette.style.visibility="visible";
      clear.style.visibility="visible";
      colorizeBtn.style.visibility="visible";
      uploadLabel.style.visibility="hidden";

    };
    image.src = event.target.result;

  };
  reader.readAsDataURL(file);

});
}

function draw() {
  var color = getSelectedColor();
  canvas.addEventListener('mousedown', function(event) {
    const x = event.offsetX;
    const y = event.offsetY;
    context.fillStyle = color;
    context.fillRect(x, y, 5, 5);
  });
}

// Get the selected color from the color palette
function getSelectedColor() {
    return selectedColor.value;
}


// Clear the canvas
function clearCanvas() {
  context.clearRect(0, 0, canvas.width, canvas.height);
  colorPalette.style.visibility="hidden";
  clear.style.visibility="hidden";
  colorizeBtn.style.visibility="hidden";
  uploadLabel.style.visibility="visible";
}
