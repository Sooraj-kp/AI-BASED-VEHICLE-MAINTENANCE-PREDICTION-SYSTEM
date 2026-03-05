/* VehicleAI — Main JavaScript */

// Page load animations
document.addEventListener('DOMContentLoaded', () => {
  // Fade in main content
  document.querySelector('.main-wrap').style.opacity = '0';
  document.querySelector('.main-wrap').style.transition = 'opacity .4s ease';
  setTimeout(() => {
    document.querySelector('.main-wrap').style.opacity = '1';
  }, 50);

  // Animate stat values
  animateCounters();
});

function animateCounters() {
  document.querySelectorAll('.stat-val').forEach(el => {
    const raw = el.textContent.replace(/[^0-9.]/g, '');
    const target = parseFloat(raw);
    if (!isNaN(target) && target > 0) {
      const prefix = el.textContent.match(/^[^0-9]*/)?.[0] || '';
      const suffix = el.textContent.match(/[^0-9.]*$/)?.[0] || '';
      let start = 0;
      const duration = 900;
      const step = 16;
      const increment = target / (duration / step);
      const timer = setInterval(() => {
        start = Math.min(start + increment, target);
        const display = Number.isInteger(target)
          ? Math.round(start).toLocaleString('en-IN')
          : start.toFixed(1);
        el.textContent = prefix + display + suffix;
        if (start >= target) clearInterval(timer);
      }, step);
    }
  });
}

// Smooth scroll anchor links
document.querySelectorAll('a[href^="#"]').forEach(a => {
  a.addEventListener('click', e => {
    e.preventDefault();
    const target = document.querySelector(a.getAttribute('href'));
    if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
  });
});
