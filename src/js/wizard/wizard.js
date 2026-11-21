/** 傻瓜式向导状态机：把复杂流程拆成步骤，记录进度，供 UI 高亮当前步骤。 */
class Wizard {
  constructor(steps) {
    this.steps = steps;
    this.current = 0;
  }
  next() { if (this.current < this.steps.length - 1) this.current++; return this.current; }
  prev() { if (this.current > 0) this.current--; return this.current; }
  goto(i) { this.current = Math.max(0, Math.min(i, this.steps.length - 1)); return this.current; }
  render(containerId) {
    const el = document.getElementById(containerId);
    if (!el) return;
    el.innerHTML = this.steps
      .map((s, i) => `<div class="wizard-step ${i === this.current ? 'active' : ''}">${i + 1}. ${s}</div>`)
      .join('');
  }
}
window.Wizard = Wizard;
