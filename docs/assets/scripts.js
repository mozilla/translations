document.addEventListener("DOMContentLoaded", () => {
  const a = document.querySelector(".navbar-brand");
  const baseDir = a.href;

  {
    const logo = document.createElement("img");
    logo.src = `${baseDir}/assets/translations.svg`;
    logo.className = "translations-logo";
    a.insertBefore(logo, a.firstChild);
  }
  {
    const container = document.querySelector(".bs-sidebar");
    const noodles = document.createElement("img");
    noodles.src = `${baseDir}/assets/noodles.png`;
    noodles.className = "translations-noodles";
    container.insertBefore(noodles, container.firstChild);
  }

  // "Edit on Github" is quite long for the menu, shorten it.
  document.querySelector(".fa-github").nextSibling.textContent = " Edit";
});
