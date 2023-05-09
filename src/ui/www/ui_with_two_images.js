
const upload = document.getElementById('upload');
upload.onclick = uploadImage();
const uploadLabel = document.getElementById('uploadLabel');

const canvas = document.getElementById('canvas');
const context = canvas.getContext('2d');

const canvasOutput = document.getElementById('canvasOutput');
const context2 = canvasOutput.getContext('2d');

const clear = document.getElementById('clear');
const colorizeBtn = document.getElementById('colorize');
const groundTruthBtn = document.getElementById('groundTruth');


const selectedColor = document.getElementById('selectedColor');
const suggestedColors = document.querySelectorAll('.color');
const suggestedColorsDiv = document.getElementById('suggestedColors');

const main = document.getElementById("main");




const outputImage = document.getElementById('outputImage');
var real_width = 0
var rescaled_width = 0
var real_height = 0
var rescaled_height = 0
var dx = 0
var dy = 0


// Source: https://stackoverflow.com/questions/3971841/how-to-resize-images-proportionally-keeping-the-aspect-ratio
function calculateAspectRatioFit(srcWidth, srcHeight, maxWidth, maxHeight) {
    var ratio = Math.min(maxWidth / srcWidth, maxHeight / srcHeight);
    return { width: srcWidth*ratio, height: srcHeight*ratio };
 }


// When an image is selected, display it on the canvas
// Upon upload, an API call is made to get the suggested colors for that image
function uploadImage() {
upload.addEventListener('change', function() {
  const file = upload.files[0];
  formData.append("file", file);

  const reader = new FileReader();
  reader.onload = function(event) {
    const image = new Image();
    image.onload = function() {
      canvas.width = 400;
      canvas.height = 350;

    var ratio = calculateAspectRatioFit(image.width, image.height, canvas.width, canvas.height);
    var xOffset = ratio.width < canvas.width ? ((canvas.width - ratio.width) / 2) : 0;
    var yOffset = ratio.height < canvas.height ? ((canvas.height - ratio.height) / 2) : 0;

    real_width = image.width;
    real_height = image.height;
    rescaled_width=ratio.width;
    rescaled_height = ratio.height;

    dx = xOffset
    dy = yOffset

      context.drawImage(image, xOffset, yOffset, ratio.width, ratio.height);
      // colorPalette.style.visibility="visible";
      colorPalette.style.display="block";
      main.style.marginTop="20px"

      clear.style.visibility="visible";
      colorizeBtn.style.visibility="visible";

      // uploadLabel.style.visibility="hidden";
      uploadLabel.style.display="none";
      groundTruthBtn.style.visibility="hidden";


    };
    image.src = event.target.result;

  };
  reader.readAsDataURL(file);
  let i = 0;


  fetch('/suggested_colors', {
    method: 'POST',
    body: formData,
  })

  .then(response => response.json())
  .then(data => {  const arr = data['data'];

    suggestedColors.forEach(color => {
  
    let r = arr[i][0];
    let g = arr[i][1];
    let b = arr[i][2];
    suggestedColorsDiv.style.visibility="visible";

    // let r = Math.min(Math.round(arr[i][0] * 1.2), 255);
    // let g = Math.min(Math.round(arr[i][1] * 1.2), 255);
    // let b = Math.min(Math.round(arr[i][2] * 1.2), 255);
   
    i = i + 1;
      color.style.background=rgbToHex(r,g,b)    
    });
console.log(data); console.log("received");})

    
});
}
const formData = new FormData();
const coords = []

// Draw on canvas
// Everytime a point is drawn into the picture, an API call is made to send picture, x and y coordinates, and selected color
canvas.addEventListener('mousedown', function(event) {
  const canvas_x = event.offsetX;
  const canvas_y = event.offsetY;
  const x = Math.floor((canvas_x - dx)  / (rescaled_width/real_width) )
  const y = Math.floor((canvas_y - dy)  / (rescaled_height/real_height) )
  console.log("here")
  console.log(x, y)
  if (x >=0 & y >=0 & x <= real_width & y <= real_height){

  var color = getSelectedColor();
  coords.push({x:x, y:y, color:color})
  context.fillStyle = color;
  context.beginPath();
  context.moveTo(canvas_x, canvas_y);
  context.arc(canvas_x, canvas_y, 2.5, 0, Math.PI * 2, false);
  context.fill();

  formData.set('coords', JSON.stringify(coords));
  console.log(JSON.stringify(coords))
  formData.append("color", color);

  fetch('/input_image', {
  method: 'POST',
  body: formData,
})

.then(response => response.blob())
.then(data => {
    var img = new Image();
  img.onload = function() {
    var ratio = calculateAspectRatioFit(img.width, img.height, canvasOutput.width, canvasOutput.height);
    var xOffset = ratio.width < canvasOutput.width ? ((canvasOutput.width - ratio.width) / 2) : 0;
    var yOffset = ratio.height < canvasOutput.height ? ((canvasOutput.height - ratio.height) / 2) : 0;

    context2.drawImage(img, xOffset, yOffset, ratio.width, ratio.height);
  };
  img.src = URL.createObjectURL(data);
})

}});


function show_inception_image(){fetch('/inception_image', {
  method: 'POST',
  body: formData,
})

.then(response => response.blob())
.then(data => {
    var img = new Image();
  img.onload = function() {
    var ratio = calculateAspectRatioFit(img.width, img.height, canvasOutput.width, canvasOutput.height);
    var xOffset = ratio.width < canvasOutput.width ? ((canvasOutput.width - ratio.width) / 2) : 0;
    var yOffset = ratio.height < canvasOutput.height ? ((canvasOutput.height - ratio.height) / 2) : 0;

    context2.drawImage(img, xOffset, yOffset, ratio.width, ratio.height);
  };
  img.src = URL.createObjectURL(data);
})}





// Convert color from RGB format to Hex
function rgbToHex(r, g, b) {
  return "#" + (1 << 24 | r << 16 | g << 8 | b).toString(16).slice(1);}


// Update selected color when a suggested color is selected
suggestedColors.forEach(color => {
  color.addEventListener('click', () => {
    let rgb = color.style.background;
    let r = parseInt(rgb.match(/\d+/g)[0]);
    let g = parseInt(rgb.match(/\d+/g)[1]);
    let b = parseInt(rgb.match(/\d+/g)[2]);
    selectedColor.value=rgbToHex(r,g,b)    
  });
});


// Get the selected color from the color palette
function getSelectedColor() {
    return selectedColor.value;
}


// Clear the canvas
function clearCanvas() {
  // context.clearRect(0, 0, canvas.width, canvas.height);
  // context2.clearRect(0, 0, canvas.width, canvas.height);

  // colorPalette.style.visibility="hidden";
  // clear.style.visibility="hidden";
  // colorizeBtn.style.visibility="hidden";
  // suggestedColorsDiv.style.visibility="hidden";

  // uploadLabel.style.visibility="visible";
  location.reload()
}
