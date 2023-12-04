/*
  This is your site JavaScript code - you can add interactivity!
*/

// Print a message in the browser's dev tools console each time the page loads
// Use your menus or right-click / control-click and choose "Inspect" > "Console"
console.log("Hello 🌎");

/* 
Make the "Click me!" button move when the visitor clicks it:
- First add the button to the page by following the steps in the TODO 🚧
*/
const btn = document.querySelector("button"); // Get the button from the page
if (btn) { // Detect clicks on the button
  btn.onclick = function () {
    // The 'dipped' class in style.css changes the appearance on click
    //btn.classList.toggle("dipped");
    var ht = document.getElementById("HomeTeam");
    var hText = ht.options[ht.selectedIndex].text;
    var at = document.getElementById("AwayTeam");
    var aText = at.options[at.selectedIndex].text;

    if (hText === 'POR' && aText === 'UTA') {
      document.getElementById("HomeWin").innerHTML="50.18%";
      document.getElementById('AwayLose').innerHTML="49.82%";
    } else if (hText === 'SAC' && aText === 'DEN') {
      document.getElementById('HomeWin').innerHTML="51.07%";
      document.getElementById('AwayLose').innerHTML="48.93%";
    } else if (hText === 'LAL' && aText === 'HOU') {
      document.getElementById('HomeWin').innerHTML="50.20%";
      document.getElementById('AwayLose').innerHTML="49.80%";
    } else if (hText === 'GSW' && aText === 'LAC') {
      document.getElementById('HomeWin').innerHTML="50.24%";
      document.getElementById('AwayLose').innerHTML="49.76%";
    } else if (hText === 'CHA' && aText === 'MIN') {
      document.getElementById('HomeWin').innerHTML="52.14%";
      document.getElementById('AwayLose').innerHTML="47.86%";
    } else if (hText === 'BKN' && aText === 'ORL') {
      document.getElementById('HomeWin').innerHTML="50.20%";
      document.getElementById('AwayLose').innerHTML="49.80%";
    } else if (hText === 'DET' && aText === 'CLE') {
      document.getElementById('HomeWin').innerHTML="50.20%";
      document.getElementById('AwayLose').innerHTML="49.80%";
    } else if (hText === 'MIA' && aText === 'IND') {
      document.getElementById('HomeWin').innerHTML="50.04%";
      document.getElementById('AwayLose').innerHTML="49.96%";
    } else if (hText === 'CHI' && aText === 'NOP') {
      document.getElementById('HomeWin').innerHTML="51.18%";
      document.getElementById('AwayLose').innerHTML="48.82%";
    } else if (hText === 'MIL' && aText === 'ATL') {
      document.getElementById('HomeWin').innerHTML="51.20%";
      document.getElementById('AwayLose').innerHTML="48.80%";
    } else if (hText === 'DAL' && aText === 'OKC') {
      document.getElementById('HomeWin').innerHTML="50.61%";
      document.getElementById('AwayLose').innerHTML="49.39%";
    } else {
      document.getElementById('HomeWin').innerHTML="----";
      document.getElementById('AwayLose').innerHTML="----";
    }
  };
}


// ----- GLITCH STARTER PROJECT HELPER CODE -----

// Open file when the link in the preview is clicked
let goto = (file, line) => {
  window.parent.postMessage(
    { type: "glitch/go-to-line", payload: { filePath: file, line: line } }, "*"
  );
};
// Get the file opening button from its class name
const filer = document.querySelectorAll(".fileopener");
filer.forEach((f) => {
  f.onclick = () => { goto(f.dataset.file, f.dataset.line); };
});

//MY CODE//