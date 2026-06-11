(function () {
  const body = document.body;
  const toggleBtn = document.getElementById("themeToggleBtn");
  const toggleIcon = document.querySelector(".toggle-icon");
  const toggleLabel = document.querySelector(".toggle-label");
  const logo = document.getElementById("logo");
  const highlightTheme = document.getElementById("highlight-theme");

  function updateAssets(theme) {
    if (logo) {
      logo.src =
        theme === "light"
          ? "images/rllm_logo/rllm_logo_light.png"
          : "images/rllm_logo/rllm_logo_dark.png";
    }
    const setIcon = (id, light, dark) => {
      const el = document.getElementById(id);
      if (el) el.src = theme === "light" ? light : dark;
    };
    setIcon("email-icon", "images/contact_icons/email_light.png", "images/contact_icons/email_dark.png");
    setIcon("github-icon", "images/contact_icons/github_light.png", "images/contact_icons/github_dark.png");
    setIcon("x-icon", "images/contact_icons/x_light.png", "images/contact_icons/x_dark.png");
    if (highlightTheme) {
      highlightTheme.href =
        theme === "light"
          ? "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css"
          : "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css";
    }
  }

  function applyTheme(theme) {
    if (theme === "light") {
      body.classList.add("light-mode");
      if (toggleIcon) toggleIcon.textContent = "☾";
      if (toggleLabel) toggleLabel.textContent = "Dark Mode";
    } else {
      body.classList.remove("light-mode");
      if (toggleIcon) toggleIcon.textContent = "☀\uFE0E";
      if (toggleLabel) toggleLabel.textContent = "Light Mode";
    }
    updateAssets(theme);
  }

  let savedTheme = localStorage.getItem("theme");
  if (!savedTheme) {
    savedTheme = "light";
    localStorage.setItem("theme", "light");
  }
  applyTheme(savedTheme);

  if (toggleBtn) {
    toggleBtn.addEventListener("click", () => {
      const newTheme = body.classList.contains("light-mode") ? "dark" : "light";
      localStorage.setItem("theme", newTheme);
      applyTheme(newTheme);
    });
  }
})();
