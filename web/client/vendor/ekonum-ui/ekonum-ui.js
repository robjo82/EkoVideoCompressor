/*! ekonum-ui 3.7.0 — optional components. Without this file, every native element still works. */

/* ── src/js/audio.js ── */
/* <ekn-audio> — an audio (or video) player in the brand's colours, with an optional
 * excerpt to keep: "listen and trim", as transcript does it.
 *
 *   <ekn-audio><audio src="meeting.mp3" preload="metadata" controls></audio></ekn-audio>
 *   <ekn-audio trim start="12" end="200" label="Écouter et rogner">
 *     <audio src="meeting.mp3" preload="metadata" controls></audio>
 *   </ekn-audio>
 *
 * Progressive enhancement: the native <audio> or <video> stays the media element; the
 * component hides its controls and draws its own. Without JavaScript the browser's player
 * remains. It enhances on phones too: trimming with the thumb has no native equivalent.
 *
 * Without `trim`: play, position, time, speed — transcript's player. Arrows move by five
 * seconds (thirty with Shift), Space plays or pauses.
 * With `trim`: one timeline holds everything — the kept part highlighted, the playhead, and
 * two handles at its edges; a click on the timeline moves the playhead. A single play
 * button plays from the handle touched last: from the start of the kept part, or the five
 * seconds before its end, to hear whether the cut falls right. Playback stops at the end of
 * the kept part. Start and end can also be typed ("1:02:03", "12:30", "45"); the kept part
 * is never shorter than one second. "Tout reprendre" keeps everything again. Every change
 * fires `ekn-trim` with { start, end } in seconds; the `start` / `end` attributes follow.
 * A video shows its picture (above the controls, or beside the timeline with `trim`); an
 * audio file, even in a <video>, reserves no space. A format the browser cannot read is
 * said in words, and play is disabled.
 *
 * Handles and the position slider are native range inputs: a handle's arrows move it by one
 * second (ten with Shift), Page Up / Page Down by a tenth of the length, Home / End to the
 * ends. `player.seek(seconds, true)` jumps there and plays — for a transcript that leads to
 * the moment it shows; the media element's `timeupdate` says where playback is. Labels
 * follow the page's lang.
 */
(() => {
  if (typeof customElements === "undefined" || customElements.get("ekn-audio")) return;

  const PLAY = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5.5v13l10.5-6.5z" fill="currentColor"/></svg>';
  const PAUSE = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 5h3.5v14H7zM13.5 5H17v14h-3.5z" fill="currentColor"/></svg>';
  const SPEEDS = [1, 1.25, 1.5, 2];
  const PREVIEW = 5;   // seconds heard before the end of the kept part
  const MIN_KEPT = 1;  // an empty excerpt cannot be transcribed
  const STEP = 5, LONG_STEP = 30;   // position slider: arrows, Shift + arrows
  const HANDLE_STEP = 10;           // a handle with Shift + arrows
  const HANDLE = 16;   // handle width in px: the timeline runs between the handles' centres

  const clock = (s) => {
    s = Math.max(0, Math.round(s || 0));
    const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), r = String(s % 60).padStart(2, "0");
    return h ? `${h}:${String(m).padStart(2, "0")}:${r}` : `${m}:${r}`;
  };
  // "1:02:03", "12:30" or "45" → seconds; null if unreadable.
  const parse = (text) => {
    const parts = String(text || "").trim().split(":").map((p) => p.trim());
    if (!parts.length || parts.some((p) => p === "" || Number.isNaN(Number(p)))) return null;
    return parts.reduce((total, p) => total * 60 + Number(p), 0);
  };
  const spoken = (s) => {
    s = Math.round(s || 0);
    const m = Math.floor(s / 60), r = s % 60;
    return `${m ? `${m} min ` : ""}${r} s`;
  };

  class AudioPlayer extends HTMLElement {
    connectedCallback() {
      if (this.dataset.enhanced !== undefined) return;
      this.media = this.querySelector("audio, video");
      if (!this.media) return;
      const fr = (this.closest("[lang]")?.lang || document.documentElement.lang || "fr").startsWith("fr");
      this.t = fr
        ? { play: "Lire", pause: "Pause", fromStart: "Écouter depuis le début de la partie gardée",
            fromEnd: "Écouter la fin de la partie gardée", position: "Position", speed: "Vitesse de lecture",
            start: "Début de la partie gardée", end: "Fin de la partie gardée", startField: "Début", endField: "Fin",
            kept: (k, d) => `${k} gardées sur ${d}`, all: (d) => `${d} — tout est gardé`, reset: "Tout reprendre",
            of: "sur", sep: ",", unreadable: "Format non lisible par ce navigateur." }
        : { play: "Play", pause: "Pause", fromStart: "Play from the start of the kept part",
            fromEnd: "Play the end of the kept part", position: "Position", speed: "Playback speed",
            start: "Start of the kept part", end: "End of the kept part", startField: "Start", endField: "End",
            kept: (k, d) => `${k} kept of ${d}`, all: (d) => `${d}, all kept`, reset: "Keep everything",
            of: "of", sep: ".", unreadable: "This browser cannot read this format." };
      this.trim = this.hasAttribute("trim");
      this.last = "start";
      this.media.removeAttribute("controls");
      this.dataset.enhanced = "";
      const label = this.getAttribute("label");
      const play = `<button type="button" class="ekn-audio__play" aria-label="${this.trim ? this.t.fromStart : this.t.play}">${PLAY}</button>`;
      this.insertAdjacentHTML("beforeend", this.trim ? `
        <div class="ekn-audio__head">
          ${label ? `<strong class="ekn-audio__title"></strong>` : ""}
          <span class="ekn-audio__status"><span data-status aria-live="polite"></span>
            <button type="button" class="ekn-audio__reset" hidden>${this.t.reset}</button></span>
        </div>
        <div class="ekn-audio__body">
          <div class="ekn-audio__picture"></div>
          <div class="ekn-audio__main">
            <div class="ekn-audio__bar">
              ${play}
              <div class="ekn-audio__timeline">
                <span class="ekn-audio__kept"></span>
                <span class="ekn-audio__playhead"></span>
                <input type="range" class="ekn-audio__range ekn-audio__handle" data-handle="start" min="0" max="0" step="1" value="0" aria-label="${this.t.start}">
                <input type="range" class="ekn-audio__range ekn-audio__handle" data-handle="end" min="0" max="0" step="1" value="0" aria-label="${this.t.end}">
              </div>
            </div>
            <div class="ekn-audio__fields">
              <label class="ekn-audio__field">${this.t.startField} <input class="ekn-input ekn-audio__clock" data-field="start" inputmode="numeric" autocomplete="off"></label>
              <span class="ekn-audio__now" data-now></span>
              <label class="ekn-audio__field">${this.t.endField} <input class="ekn-input ekn-audio__clock" data-field="end" inputmode="numeric" autocomplete="off"></label>
            </div>
          </div>
        </div>` : `
        <div class="ekn-audio__bar">
          ${play}
          <input type="range" class="ekn-audio__range ekn-audio__seek" min="0" max="0" step="1" value="0" aria-label="${this.t.position}">
          <span class="ekn-audio__time"><span data-now>0:00</span> / <span data-total>0:00</span></span>
          <button type="button" class="ekn-audio__speed" aria-label="${this.t.speed} : 1×">1×</button>
        </div>`);
      this.insertAdjacentHTML("beforeend", `<p class="ekn-audio__error" hidden></p>`);
      const $ = (s) => this.querySelector(s);
      if (label) $(".ekn-audio__title").textContent = label;   // text, never markup
      this.ui = { play: $(".ekn-audio__play"), now: $("[data-now]"), total: $("[data-total]"), seek: $(".ekn-audio__seek"),
                  speed: $(".ekn-audio__speed"), timeline: $(".ekn-audio__timeline"), status: $("[data-status]"),
                  reset: $(".ekn-audio__reset"), start: $('[data-handle="start"]'), end: $('[data-handle="end"]'),
                  startField: $('[data-field="start"]'), endField: $('[data-field="end"]'),
                  picture: $(".ekn-audio__picture"), error: $(".ekn-audio__error") };

      const m = this.media;
      if (this.trim && m.tagName === "VIDEO") $(".ekn-audio__picture").append(m);
      m.addEventListener("loadedmetadata", () => this.ready());
      m.addEventListener("durationchange", () => this.ready());
      m.addEventListener("timeupdate", () => this.tick());
      m.addEventListener("play", () => this.state());
      m.addEventListener("pause", () => this.state());
      m.addEventListener("ended", () => this.state());
      m.addEventListener("error", () => this.unreadable());
      if (m.error) this.unreadable();
      this.ui.play.addEventListener("click", () => this.toggle());
      if (this.trim) this.wireTrim();
      else this.wireSimple();
      if (m.readyState >= 1) this.ready();
    }

    wireSimple() {
      const m = this.media;
      const seek = this.ui.seek;
      seek.addEventListener("input", () => { m.currentTime = +seek.value; this.tick(); });
      // While the thumb is held, playback does not pull it back.
      seek.addEventListener("pointerdown", () => { this.holding = true; });
      for (const type of ["pointerup", "pointercancel", "change"]) seek.addEventListener(type, () => { this.holding = false; });
      // A meeting is long: an arrow moves by five seconds, not one; Space plays.
      seek.addEventListener("keydown", (e) => {
        const dir = { ArrowRight: 1, ArrowUp: 1, ArrowLeft: -1, ArrowDown: -1 }[e.key];
        if (dir) { e.preventDefault(); this.seek(m.currentTime + dir * (e.shiftKey ? LONG_STEP : STEP)); }
        else if (e.key === " ") { e.preventDefault(); this.toggle(); }
      });
      this.ui.speed.addEventListener("click", () => {
        const next = SPEEDS[(SPEEDS.indexOf(m.playbackRate) + 1) % SPEEDS.length] || 1;
        m.playbackRate = next;
        const text = `${String(next).replace(".", this.t.sep)}×`;
        this.ui.speed.textContent = text;
        this.ui.speed.setAttribute("aria-label", `${this.t.speed} : ${text}`);
      });
    }

    wireTrim() {
      const touched = (h) => { this.last = h; this.paint(); };
      for (const h of ["start", "end"]) {
        const input = this.ui[h], field = this.ui[`${h}Field`];
        input.addEventListener("input", () => { this.place(h, +input.value); this.seek(+this.ui[h].value); });
        input.addEventListener("pointerdown", () => touched(h));
        input.addEventListener("focus", () => touched(h));
        input.addEventListener("change", () => this.emit());
        input.addEventListener("keydown", (e) => {
          const dir = { ArrowRight: 1, ArrowUp: 1, ArrowLeft: -1, ArrowDown: -1 }[e.key];
          if (!dir || !e.shiftKey) return;
          e.preventDefault();
          this.place(h, +input.value + dir * HANDLE_STEP);
          this.seek(+input.value);
          this.emit();
        });
        field.addEventListener("focus", () => touched(h));
        field.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); field.blur(); } });
        field.addEventListener("blur", () => {
          const t = parse(field.value);
          if (t === null) { this.paint(); return; }
          this.place(h, t);
          this.emit();
        });
      }
      // A click on the timeline itself (not on a handle) moves the playhead there.
      this.ui.timeline.addEventListener("pointerdown", (e) => {
        if (e.target.classList.contains("ekn-audio__handle")) return;
        const box = this.ui.timeline.getBoundingClientRect();
        const ratio = Math.min(Math.max((e.clientX - box.left - HANDLE / 2) / (box.width - HANDLE), 0), 1);
        this.seek(ratio * this.duration);
      });
      this.ui.reset.addEventListener("click", () => {
        this.ui.start.value = 0;
        this.ui.end.value = this.max;
        this.paint();
        this.emit();
      });
    }

    get duration() { return isFinite(this.media.duration) ? this.media.duration : 0; }

    ready() {
      const d = this.duration;
      if (d <= 0) return;
      this.max = Math.floor(d);
      // A <video> holding a sound only has no picture to show.
      if (this.media.tagName === "VIDEO" && !this.media.videoWidth) {
        if (this.trim) this.ui.picture.hidden = true;
        else this.media.hidden = true;
      }
      if (!this.trim) {
        this.ui.total.textContent = clock(d);
        this.ui.seek.max = this.max;
      } else {
        for (const h of ["start", "end"]) this.ui[h].max = this.max;
        const start = Math.min(Math.max(0, +(this.getAttribute("start") ?? 0)), this.max - MIN_KEPT);
        const end = Math.min(Math.max(start + MIN_KEPT, +(this.getAttribute("end") ?? this.max)), this.max);
        this.ui.start.value = start;
        this.ui.end.value = end;
        this.paint();
      }
      this.tick();
    }

    range() {
      return { start: +this.ui.start.value, end: +this.ui.end.value };
    }

    // Moves one edge, keeping at least MIN_KEPT seconds between them.
    place(edge, t) {
      const { start, end } = this.range();
      t = Math.round(t);
      if (edge === "start") this.ui.start.value = Math.min(Math.max(t, 0), end - MIN_KEPT);
      else this.ui.end.value = Math.max(Math.min(t, this.max), start + MIN_KEPT);
      this.last = edge;
      this.paint();
    }

    // Public: moves playback to `t` seconds, and plays if asked.
    seek(t, play = false) {
      this.media.currentTime = Math.min(Math.max(t, 0), this.duration || t);
      this.tick();
      if (play && this.media.paused) this.media.play()?.catch(() => {});
    }

    unreadable() {
      this.ui.play.disabled = true;
      this.ui.error.textContent = this.t.unreadable;
      this.ui.error.hidden = false;
    }

    at(t) {   // timeline position, aligned with the centre of a handle
      const p = this.max ? Math.min(Math.max(t, 0), this.max) / this.max : 0;
      return `calc(${HANDLE / 2}px + (100% - ${HANDLE}px) * ${p})`;
    }

    paint() {
      const { start, end } = this.range(), max = this.max || 0;
      const kept = this.querySelector(".ekn-audio__kept");
      kept.style.left = this.at(start);
      kept.style.width = `calc((100% - ${HANDLE}px) * ${max ? (end - start) / max : 0})`;
      for (const h of ["start", "end"]) {
        const v = h === "start" ? start : end;
        this.ui[h].setAttribute("aria-valuetext", spoken(v));
        this.ui[h].toggleAttribute("data-last", this.last === h);
        if (document.activeElement !== this.ui[`${h}Field`]) this.ui[`${h}Field`].value = clock(v);
      }
      const trimmed = start > 0 || end < max;
      this.ui.status.textContent = trimmed ? this.t.kept(clock(end - start), clock(max)) : this.t.all(clock(max));
      this.ui.reset.hidden = !trimmed;
      if (this.media.paused) this.ui.play.setAttribute("aria-label", this.last === "end" ? this.t.fromEnd : this.t.fromStart);
      this.setAttribute("start", start);
      this.setAttribute("end", end);
    }

    emit() {
      this.dispatchEvent(new CustomEvent("ekn-trim", { bubbles: true, detail: this.range() }));
    }

    toggle() {
      const m = this.media;
      if (!m.paused) { m.pause(); return; }
      if (this.trim) {
        const { start, end } = this.range();
        m.currentTime = this.last === "end" ? Math.max(end - PREVIEW, start) : start;
      }
      m.play()?.catch(() => {});
    }

    tick() {
      const m = this.media, now = m.currentTime || 0;
      if (this.trim) {
        // Playback stops at the edge of the kept part: what lies beyond will not be kept.
        if (!m.paused && now >= +this.ui.end.value) { m.pause(); m.currentTime = +this.ui.end.value; }
        this.querySelector(".ekn-audio__playhead").style.left = this.at(m.currentTime);
        this.ui.now.textContent = !m.paused || now > 0 ? clock(m.currentTime) : "";
        return;
      }
      this.ui.now.textContent = clock(now);
      if (!this.holding) this.ui.seek.value = Math.floor(now);
      this.ui.seek.style.setProperty("--ekn-progress", `${(now / (+this.ui.seek.max || 1)) * 100}%`);
      this.ui.seek.setAttribute("aria-valuetext", `${spoken(now)} ${this.t.of} ${spoken(this.duration)}`);
    }

    state() {
      const playing = !this.media.paused;
      this.ui.play.innerHTML = playing ? PAUSE : PLAY;
      if (playing) this.ui.play.setAttribute("aria-label", this.t.pause);
      else if (this.trim) this.paint();
      else this.ui.play.setAttribute("aria-label", this.t.play);
    }
  }

  customElements.define("ekn-audio", AudioPlayer);
})();

/* ── src/js/date-picker.js ── */
/* <ekn-date-picker> — a calendar for <input type="date">, on desktop only.
 *
 *   <ekn-date-picker><input class="ekn-input" type="date" name="day" min="2026-01-01"></ekn-date-picker>
 *
 * Progressive enhancement: the native input stays the source of truth (its value, its
 * name, its min and max, form submission, typing the date by hand). The element only adds
 * a button that opens a calendar in the brand's colours. It does nothing:
 *   - on a touch screen (pointer: coarse), where the phone's own calendar is the best one;
 *   - when JavaScript does not load, or the browser has no popover support.
 * Add the `always` attribute to enhance on touch screens too.
 *
 * Keyboard, in the calendar: arrows move by day and week, Page Up / Page Down by month
 * (with Shift, by year), Home / End to the start / end of the week, Enter picks,
 * Escape closes. Picking fires `input` and `change` on the native input.
 * Month and day names follow the page's lang (French by default).
 */
(() => {
  if (typeof customElements === "undefined" || customElements.get("ekn-date-picker")) return;

  const ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><rect x="3" y="5" width="18" height="16" rx="3"/><path d="M3 10h18M8 3v4M16 3v4"/></svg>';
  const pad = (n) => String(n).padStart(2, "0");
  // Dates are handled in UTC: a local midnight can shift a day across a DST change.
  const toISO = (d) => `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())}`;
  const fromISO = (s) => {
    const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(s || "");
    return m ? new Date(Date.UTC(+m[1], +m[2] - 1, +m[3])) : null;
  };
  const addDays = (d, n) => new Date(Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate() + n));
  const addMonths = (d, n) => {
    const target = new Date(Date.UTC(d.getUTCFullYear(), d.getUTCMonth() + n, 1));
    const last = new Date(Date.UTC(target.getUTCFullYear(), target.getUTCMonth() + 1, 0)).getUTCDate();
    return new Date(Date.UTC(target.getUTCFullYear(), target.getUTCMonth(), Math.min(d.getUTCDate(), last)));
  };
  const today = () => { const n = new Date(); return new Date(Date.UTC(n.getFullYear(), n.getMonth(), n.getDate())); };

  let count = 0;

  class DatePicker extends HTMLElement {
    connectedCallback() {
      if (this.dataset.enhanced !== undefined) return;
      this.input = this.querySelector('input[type="date"]');
      const touch = matchMedia("(pointer: coarse)").matches && !this.hasAttribute("always");
      if (!this.input || touch || !HTMLElement.prototype.hasOwnProperty("popover")) return;

      this.lang = (this.closest("[lang]")?.lang || document.documentElement.lang || "fr").trim() || "fr";
      this.labels = this.lang.startsWith("fr")
        ? { open: "Ouvrir le calendrier", prev: "Mois précédent", next: "Mois suivant", today: "Aujourd'hui", dialog: "Choisir une date" }
        : { open: "Open the calendar", prev: "Previous month", next: "Next month", today: "Today", dialog: "Choose a date" };
      // Monday, unless the locale says otherwise (weekInfo: 1 = Monday … 7 = Sunday).
      this.firstDay = 1;
      try {
        const locale = new Intl.Locale(this.lang);
        const week = locale.getWeekInfo?.() ?? locale.weekInfo;
        if (week?.firstDay) this.firstDay = week.firstDay % 7;
      } catch { /* keep Monday */ }

      const id = `ekn-calendar-${++count}`;
      this.toggle = Object.assign(document.createElement("button"), {
        type: "button", className: "ekn-date-picker__toggle", innerHTML: ICON,
      });
      this.toggle.setAttribute("aria-label", this.labels.open);
      this.toggle.setAttribute("aria-haspopup", "dialog");
      this.toggle.setAttribute("aria-controls", id);
      this.toggle.setAttribute("aria-expanded", "false");
      this.calendar = Object.assign(document.createElement("div"), { id, className: "ekn-calendar" });
      this.calendar.setAttribute("popover", "");
      this.calendar.setAttribute("role", "dialog");
      this.calendar.setAttribute("aria-label", this.labels.dialog);
      this.append(this.toggle, this.calendar);
      this.dataset.enhanced = "";

      this.toggle.addEventListener("click", () => this.open());
      this.calendar.addEventListener("toggle", (e) => {
        this.toggle.setAttribute("aria-expanded", String(e.newState === "open"));
        // Closed by Escape or a click outside: focus goes back to the button, not to <body>.
        if (e.newState === "closed" && this.calendar.contains(document.activeElement)) this.toggle.focus();
      });
      this.calendar.addEventListener("keydown", (e) => this.key(e));
      this.calendar.addEventListener("click", (e) => {
        const day = e.target.closest("[data-date]");
        if (day) this.pick(day.dataset.date);
      });
    }

    bounds() {
      return { min: fromISO(this.input.min), max: fromISO(this.input.max) };
    }

    allowed(d) {
      const { min, max } = this.bounds();
      return !(min && d < min) && !(max && d > max);
    }

    open() {
      const { min, max } = this.bounds();
      let start = fromISO(this.input.value) || today();
      if (min && start < min) start = min;
      if (max && start > max) start = max;
      this.focused = start;
      this.render();
      this.calendar.showPopover();
      this.place();
      this.calendar.querySelector('[tabindex="0"]')?.focus();
    }

    // Under the field, aligned on its right edge, kept inside the screen.
    place() {
      const r = this.getBoundingClientRect(), c = this.calendar.getBoundingClientRect();
      const margin = 8;
      let top = r.bottom + 4;
      if (top + c.height > innerHeight - margin && r.top - 4 - c.height > margin) top = r.top - 4 - c.height;
      const left = Math.max(margin, Math.min(r.right - c.width, innerWidth - c.width - margin));
      Object.assign(this.calendar.style, { top: `${top}px`, left: `${left}px` });
    }

    render() {
      const f = this.focused, lang = this.lang;
      const selected = this.input.value, now = toISO(today());
      const first = new Date(Date.UTC(f.getUTCFullYear(), f.getUTCMonth(), 1));
      const month = new Intl.DateTimeFormat(lang, { month: "long", year: "numeric", timeZone: "UTC" }).format(first);
      const long = new Intl.DateTimeFormat(lang, { weekday: "long", day: "numeric", month: "long", year: "numeric", timeZone: "UTC" });
      const weekday = new Intl.DateTimeFormat(lang, { weekday: "short", timeZone: "UTC" });
      const weekdayLong = new Intl.DateTimeFormat(lang, { weekday: "long", timeZone: "UTC" });
      // 2023-01-01 was a Sunday: day i of the week is that date + i.
      const days = [...Array(7)].map((_, i) => new Date(Date.UTC(2023, 0, 1 + ((this.firstDay + i) % 7))));
      const lead = (first.getUTCDay() - this.firstDay + 7) % 7;
      const inMonth = new Date(Date.UTC(f.getUTCFullYear(), f.getUTCMonth() + 1, 0)).getUTCDate();
      const cells = [...Array(lead).fill(null), ...[...Array(inMonth)].map((_, i) => addDays(first, i))];
      while (cells.length % 7) cells.push(null);
      const rows = [];
      for (let i = 0; i < cells.length; i += 7) {
        rows.push(`<tr>${cells.slice(i, i + 7).map((d) => {
          if (!d) return "<td></td>";
          const iso = toISO(d), ok = this.allowed(d);
          return `<td><button type="button" class="ekn-calendar__day" data-date="${iso}" tabindex="${iso === toISO(f) ? 0 : -1}"`
            + ` aria-label="${long.format(d)}"${iso === selected ? ' aria-pressed="true"' : ""}${iso === now ? ' aria-current="date"' : ""}`
            + `${ok ? "" : ' aria-disabled="true"'}>${d.getUTCDate()}</button></td>`;
        }).join("")}</tr>`);
      }
      const { min, max } = this.bounds();
      const prevOk = !min || addMonths(first, -1) >= new Date(Date.UTC(min.getUTCFullYear(), min.getUTCMonth(), 1));
      const nextOk = !max || addMonths(first, 1) <= max;
      this.calendar.innerHTML = `
        <div class="ekn-calendar__head">
          <button type="button" class="ekn-calendar__nav" data-move="-1" aria-label="${this.labels.prev}"${prevOk ? "" : " disabled"}>‹</button>
          <strong aria-live="polite">${month}</strong>
          <button type="button" class="ekn-calendar__nav" data-move="1" aria-label="${this.labels.next}"${nextOk ? "" : " disabled"}>›</button>
        </div>
        <table class="ekn-calendar__grid" role="presentation">
          <thead><tr>${days.map((d) => `<th scope="col" abbr="${weekdayLong.format(d)}">${weekday.format(d).replace(".", "")}</th>`).join("")}</tr></thead>
          <tbody>${rows.join("")}</tbody>
        </table>
        ${this.allowed(today()) ? `<div class="ekn-calendar__foot"><button type="button" class="ekn-calendar__today" data-date="${now}">${this.labels.today}</button></div>` : ""}`;
      this.calendar.querySelectorAll("[data-move]").forEach((b) => b.addEventListener("click", () => {
        this.focused = addMonths(this.focused, +b.dataset.move);
        this.render();
        // The same button, unless it just became disabled (end of the allowed range).
        const again = this.calendar.querySelector(`[data-move="${b.dataset.move}"]:not(:disabled)`);
        (again || this.calendar.querySelector('[tabindex="0"]'))?.focus();
      }));
    }

    key(e) {
      const day = e.target.closest(".ekn-calendar__day");
      if (!day) return;
      const f = fromISO(day.dataset.date);
      const weekStart = (f.getUTCDay() - this.firstDay + 7) % 7;
      const moves = {
        ArrowLeft: () => addDays(f, -1), ArrowRight: () => addDays(f, 1),
        ArrowUp: () => addDays(f, -7), ArrowDown: () => addDays(f, 7),
        PageUp: () => addMonths(f, e.shiftKey ? -12 : -1), PageDown: () => addMonths(f, e.shiftKey ? 12 : 1),
        Home: () => addDays(f, -weekStart), End: () => addDays(f, 6 - weekStart),
      };
      if (!moves[e.key]) return;
      e.preventDefault();
      this.focused = moves[e.key]();
      this.render();
      this.calendar.querySelector(`[data-date="${toISO(this.focused)}"].ekn-calendar__day`)?.focus();
    }

    pick(iso) {
      if (!this.allowed(fromISO(iso))) return;
      this.input.value = iso;
      this.input.dispatchEvent(new Event("input", { bubbles: true }));
      this.input.dispatchEvent(new Event("change", { bubbles: true }));
      this.calendar.hidePopover();
      this.toggle.focus();
    }
  }

  customElements.define("ekn-date-picker", DatePicker);
})();
