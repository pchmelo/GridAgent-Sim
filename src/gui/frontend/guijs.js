const terminalContainer = document.getElementById("terminalContainer");
const toggleBtn = document.getElementById("toggleTerminal");

toggleBtn.addEventListener("click", () => {
  const minimized = terminalContainer.classList.toggle("minimized");
  toggleBtn.classList.toggle("minimized", minimized);
});
