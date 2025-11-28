const BUCKET_NAME = "moz-fx-translations-data--303e-prod-translations-data";
const STORAGE_URL = `https://storage.googleapis.com/${BUCKET_NAME}`;
const DEFAULT_DB_URL = `${STORAGE_URL}/db/db.sqlite`;

class Config {
  static resolveDbUrl() {
    const param = new URLSearchParams(location.search).get("db");
    if (!param) return DEFAULT_DB_URL;

    if (param.startsWith("/")) {
      return new URL(param, location.origin).toString();
    }

    try {
      return new URL(param, location.href).toString();
    } catch {
      return DEFAULT_DB_URL;
    }
  }
}

const DB_URL = Config.resolveDbUrl();

class Database {
  constructor(db) {
    this.db = db;
  }

  static async open() {
    async function loadSqlJsGlobal() {
      if (globalThis.initSqlJs) return globalThis.initSqlJs;
      await new Promise((res, rej) => {
        const s = document.createElement("script");
        s.src = "https://cdn.jsdelivr.net/npm/sql.js@1.10.3/dist/sql-wasm.js";
        s.onload = res;
        s.onerror = rej;
        document.head.appendChild(s);
      });
      return globalThis.initSqlJs;
    }

    const initSqlJs = await loadSqlJsGlobal();
    const SQL = await initSqlJs({
      locateFile: (f) => "https://cdn.jsdelivr.net/npm/sql.js@1.10.3/dist/" + f,
    });

    const cacheBustUrl = `${DB_URL}?t=${Date.now()}`;
    const response = await fetch(cacheBustUrl, {
      cache: "no-store",
      headers: { "Cache-Control": "no-cache, no-store, must-revalidate" },
    });
    if (!response.ok) {
      throw new Error(`Failed to fetch database: ${response.statusText}`);
    }
    const buffer = await response.arrayBuffer();
    const db = new SQL.Database(new Uint8Array(buffer));
    return new Database(db);
  }

  getLanguagePairs() {
    const results = this.db.exec(`
      SELECT DISTINCT source_lang, target_lang
      FROM final_evals
      ORDER BY source_lang, target_lang
    `);
    if (!results.length) return [];
    return results[0].values.map(([src, trg]) => `${src}-${trg}`);
  }

  getDatasets(src, trg) {
    const results = this.db.exec(`
      SELECT DISTINCT dataset
      FROM final_evals
      WHERE source_lang = '${src}' AND target_lang = '${trg}'
      ORDER BY dataset
    `);
    if (!results.length) return [];
    return results[0].values.map(([dataset]) => dataset);
  }

  getMetrics(src, trg, dataset) {
    const results = this.db.exec(`
      SELECT DISTINCT m.metric_name
      FROM final_eval_metrics m
      JOIN final_evals e ON m.eval_id = e.id
      WHERE e.source_lang = '${src}' AND e.target_lang = '${trg}' AND e.dataset = '${dataset}'
      ORDER BY m.metric_name
    `);
    if (!results.length) return [];
    return results[0].values.map(([metric]) => metric);
  }

  getLeaderboard(src, trg, dataset) {
    const evalResults = this.db.exec(`
      SELECT e.id, e.translator, e.model_name, e.translations_url
      FROM final_evals e
      WHERE e.source_lang = '${src}' AND e.target_lang = '${trg}' AND e.dataset = '${dataset}'
    `);
    if (!evalResults.length) return [];

    const entries = evalResults[0].values.map(([id, translator, model_name, translations_url]) => ({
      id,
      translator,
      model_name,
      translations_url,
      metrics: {},
      llm_scores: {},
    }));

    for (const entry of entries) {
      const metricResults = this.db.exec(`
        SELECT metric_name, corpus_score, details_json, scores_url
        FROM final_eval_metrics
        WHERE eval_id = ${entry.id}
      `);
      if (metricResults.length) {
        for (const [metric_name, corpus_score, details_json, scores_url] of metricResults[0].values) {
          entry.metrics[metric_name] = { score: corpus_score, details_json, scores_url };
        }
      }

      const llmResults = this.db.exec(`
        SELECT l.criterion, l.score, l.summary
        FROM final_eval_llm_scores l
        JOIN final_eval_metrics m ON l.metric_id = m.id
        WHERE m.eval_id = ${entry.id}
      `);
      if (llmResults.length) {
        for (const [criterion, score, summary] of llmResults[0].values) {
          entry.llm_scores[criterion] = { score, summary };
        }
      }
    }

    return entries;
  }
}

class URLStateManager {
  constructor() {
    this.state = {
      langpair: null,
    };
    this.listeners = [];
    this._loadFromUrl();
    window.addEventListener("popstate", () => {
      this._loadFromUrl();
      this._notifyListeners();
    });
  }

  _loadFromUrl() {
    const params = new URLSearchParams(window.location.search);
    this.state.langpair = params.get("langpair");
  }

  _saveToUrl() {
    const params = new URLSearchParams(window.location.search);
    if (this.state.langpair) params.set("langpair", this.state.langpair);
    else params.delete("langpair");
    const newUrl = params.toString() ? `?${params.toString()}` : window.location.pathname;
    window.history.pushState({}, "", newUrl);
  }

  update(key, value) {
    this.state[key] = value;
    this._saveToUrl();
    this._notifyListeners();
  }

  onChange(callback) {
    this.listeners.push(callback);
  }

  _notifyListeners() {
    this.listeners.forEach((cb) => cb(this.state));
  }
}

class DatasetLeaderboard {
  constructor(container, dataset, data, srcLang, trgLang, diffMode = false) {
    this.container = container;
    this.dataset = dataset;
    this.data = data;
    this.srcLang = srcLang;
    this.trgLang = trgLang;
    this.sortMetric = null;
    this.sortDirection = "desc";
    this.diffMode = diffMode;
    this.googleV2Baseline = this._findGoogleV2Baseline();
  }

  _findGoogleV2Baseline() {
    const googleEntry = this.data.find(
      (e) => e.translator === "google" && e.model_name === "v2"
    );
    if (!googleEntry) return null;
    const baseline = {};
    for (const [metric, data] of Object.entries(googleEntry.metrics)) {
      baseline[metric] = data.score;
    }
    return baseline;
  }

  setDiffMode(diffMode) {
    this.diffMode = diffMode;
  }

  render() {
    const allMetrics = new Set();
    for (const entry of this.data) {
      Object.keys(entry.metrics).forEach((m) => allMetrics.add(m));
    }
    const metrics = Array.from(allMetrics).sort();

    if (!this.sortMetric || !metrics.includes(this.sortMetric)) {
      const chrfMetric = metrics.find(m => m.toLowerCase() === "chrf");
      this.sortMetric = chrfMetric || metrics[0] || null;
    }
    const hasLlmScores = this.data.some((e) => Object.keys(e.llm_scores).length > 0);

    let html = `<div class="dataset-section">
      <h2 class="dataset-title">${this.dataset}</h2>
      <div class="table-container">
        <table class="leaderboard-table">
          <thead><tr>
            <th class="rank-col">Rank</th>
            <th>Translator</th>
            <th>Model</th>`;

    for (const metric of metrics) {
      const sortClass = metric === this.sortMetric
        ? (this.sortDirection === "desc" ? "sorted-desc" : "sorted-asc")
        : "";
      html += `<th class="score-col ${sortClass}" data-metric="${metric}">${metric}</th>`;
    }
    if (hasLlmScores) {
      html += `<th>LLM Details</th>`;
    }
    html += `</tr></thead><tbody>`;

    const sortedData = this._sortData(this.data, this.sortMetric, this.sortDirection);

    sortedData.forEach((entry, index) => {
      const rank = index + 1;
      const rankClass = rank <= 3 ? `rank-${rank}` : "";
      const modelLink = this._getModelRegistryLink(entry);

      html += `<tr>
        <td class="rank-cell ${rankClass}">${rank}</td>
        <td class="translator-cell">${entry.translator}</td>`;
      if (modelLink) {
        html += `<td class="model-cell" title="${entry.model_name}"><a href="${modelLink}" target="_blank">${entry.model_name}</a></td>`;
      } else {
        html += `<td class="model-cell" title="${entry.model_name}">${entry.model_name}</td>`;
      }

      for (const metric of metrics) {
        const metricData = entry.metrics[metric];
        if (metricData) {
          const score = this._formatScore(metricData.score, metric);
          const hasScores = metricData.scores_url || entry.translations_url;
          const clickable = hasScores ? "clickable" : "";
          html += `<td class="score-cell ${clickable}" data-eval-id="${entry.id}" data-metric="${metric}">${score}</td>`;
        } else {
          html += `<td class="score-cell">-</td>`;
        }
      }

      if (hasLlmScores) {
        if (Object.keys(entry.llm_scores).length > 0) {
          html += `<td class="llm-scores-cell"><div class="llm-scores-container">`;
          for (const [criterion, d] of Object.entries(entry.llm_scores)) {
            const scoreClass = `llm-score-${Math.round(d.score)}`;
            html += `<span class="llm-score-badge ${scoreClass}"><span class="criterion">${criterion.slice(0, 3)}</span>${d.score.toFixed(1)}</span>`;
          }
          html += `</div></td>`;
        } else {
          html += `<td>-</td>`;
        }
      }
      html += `</tr>`;
    });

    html += `</tbody></table></div></div>`;
    this.container.innerHTML = html;

    this.container.querySelectorAll("th[data-metric]").forEach((th) => {
      th.addEventListener("click", () => {
        const metric = th.dataset.metric;
        if (metric === this.sortMetric) {
          this.sortDirection = this.sortDirection === "desc" ? "asc" : "desc";
        } else {
          this.sortMetric = metric;
          this.sortDirection = "desc";
        }
        this.render();
      });
    });

    this.container.querySelectorAll(".score-cell.clickable").forEach((td) => {
      td.addEventListener("click", () => {
        const evalId = parseInt(td.dataset.evalId);
        const metric = td.dataset.metric;
        const entry = this.data.find((e) => e.id === evalId);
        if (entry) {
          this._showOverlay(entry, metric);
        }
      });
    });
  }

  _sortData(data, metric, direction) {
    return [...data].sort((a, b) => {
      let aScore = a.metrics[metric]?.score ?? -Infinity;
      let bScore = b.metrics[metric]?.score ?? -Infinity;

      if (this.diffMode && this.googleV2Baseline) {
        const baseline = this.googleV2Baseline[metric];
        if (baseline !== undefined) {
          aScore = aScore === -Infinity ? -Infinity : aScore - baseline;
          bScore = bScore === -Infinity ? -Infinity : bScore - baseline;
        }
      }

      return direction === "desc" ? bScore - aScore : aScore - bScore;
    });
  }

  _formatScore(score, metric) {
    if (score === null || score === undefined) return "-";

    if (this.diffMode && this.googleV2Baseline) {
      const baseline = this.googleV2Baseline[metric];
      if (baseline !== undefined) {
        const diff = score - baseline;
        const sign = diff >= 0 ? "+" : "";
        return `${sign}${diff.toFixed(2)}`;
      }
    }

    return score.toFixed(2);
  }

  async _showOverlay(entry, metricName) {
    const overlay = document.getElementById("overlay");
    const content = document.getElementById("overlay-content");

    content.innerHTML = "<p>Loading translations...</p>";
    overlay.classList.add("visible");

    try {
      let translations = [];
      const allScores = {};
      let googleV2Scores = null;

      if (entry.translations_url) {
        const resp = await fetch(entry.translations_url);
        if (resp.ok) translations = await resp.json();
      }

      const fetchPromises = Object.entries(entry.metrics).map(async ([metric, data]) => {
        if (data.scores_url) {
          try {
            const resp = await fetch(data.scores_url);
            if (resp.ok) allScores[metric] = await resp.json();
          } catch {}
        }
      });
      await Promise.all(fetchPromises);

      if (this.diffMode) {
        googleV2Scores = await this._fetchGoogleV2Scores();
      }

      this._renderOverlayContent(entry, metricName, translations, allScores, null, "desc", googleV2Scores);
    } catch (error) {
      content.innerHTML = `<p>Error loading data: ${error.message}</p>`;
    }
  }

  async _fetchGoogleV2Scores() {
    const googleEntry = this.data.find(
      (e) => e.translator === "google" && e.model_name === "v2"
    );
    if (!googleEntry) return null;

    const scores = {};
    const fetchPromises = Object.entries(googleEntry.metrics).map(async ([metric, data]) => {
      if (data.scores_url) {
        try {
          const resp = await fetch(data.scores_url);
          if (resp.ok) scores[metric] = await resp.json();
        } catch {}
      }
    });
    await Promise.all(fetchPromises);
    return scores;
  }

  _renderOverlayContent(entry, metricName, translations, allScores, sortCol = null, sortDir = "desc", googleV2Scores = null) {
    const content = document.getElementById("overlay-content");

    const regularMetrics = Object.keys(allScores).filter(m => !m.startsWith("llm")).sort();
    const llmMetrics = Object.keys(allScores).filter(m => m.startsWith("llm")).sort();
    const llmCriteria = llmMetrics.length > 0 && allScores[llmMetrics[0]]?.[0]
      ? Object.keys(allScores[llmMetrics[0]][0])
      : [];

    let indexedData = translations.map((t, i) => {
      const segmentScores = {};
      for (const [metric, scores] of Object.entries(allScores)) {
        segmentScores[metric] = scores[i];
      }
      return { idx: i, t, scores: segmentScores };
    });

    if (sortCol !== null) {
      indexedData.sort((a, b) => {
        let aVal, bVal;
        if (llmCriteria.includes(sortCol)) {
          const llmMetric = llmMetrics[0];
          const aData = a.scores[llmMetric]?.[sortCol];
          const bData = b.scores[llmMetric]?.[sortCol];
          aVal = Array.isArray(aData) ? aData[0] : (aData ?? -Infinity);
          bVal = Array.isArray(bData) ? bData[0] : (bData ?? -Infinity);
        } else {
          aVal = a.scores[sortCol] ?? -Infinity;
          bVal = b.scores[sortCol] ?? -Infinity;
          if (this.diffMode && googleV2Scores?.[sortCol]) {
            const aBaseline = googleV2Scores[sortCol][a.idx];
            const bBaseline = googleV2Scores[sortCol][b.idx];
            if (typeof aBaseline === "number") aVal = aVal === -Infinity ? -Infinity : aVal - aBaseline;
            if (typeof bBaseline === "number") bVal = bVal === -Infinity ? -Infinity : bVal - bBaseline;
          }
        }
        return sortDir === "desc" ? bVal - aVal : aVal - bVal;
      });
    }

    let html = `
      <div class="overlay-header">
        <h2>${entry.translator}</h2>
        <div class="meta">Model: ${entry.model_name} | Dataset: ${this.dataset}</div>
      </div>
    `;

    if (Object.keys(entry.llm_scores).length > 0) {
      html += `<div class="llm-summary"><h3>LLM Overall Scores</h3><div class="llm-summary-grid">`;
      for (const [criterion, d] of Object.entries(entry.llm_scores)) {
        const scoreClass = `llm-score-${Math.round(d.score)}`;
        html += `<div class="llm-summary-item">
          <div class="criterion-header">
            <span class="criterion-name">${criterion}</span>
            <span class="criterion-score ${scoreClass}">${d.score.toFixed(1)}</span>
          </div>
          <div class="criterion-text">${d.summary || ""}</div>
        </div>`;
      }
      html += `</div></div>`;
    }

    if (translations.length > 0) {
      const hasSrc = translations.some(t => t.src);
      const hasRef = translations.some(t => t.ref);
      const textCols = 1 + (hasSrc ? 1 : 0) + 1 + (hasRef ? 1 : 0);
      const totalCols = textCols + regularMetrics.length + llmCriteria.length;

      html += `<table class="translations-table"><thead><tr>`;
      html += `<th>#</th>`;
      if (hasSrc) html += `<th>Source</th>`;
      html += `<th>Translation</th>`;
      if (hasRef) html += `<th>Reference</th>`;

      for (const metric of regularMetrics) {
        const sortClass = sortCol === metric ? (sortDir === "desc" ? "sorted-desc" : "sorted-asc") : "";
        html += `<th class="score-col sortable ${sortClass}" data-sort-col="${metric}">${metric}</th>`;
      }

      for (const criterion of llmCriteria) {
        const sortClass = sortCol === criterion ? (sortDir === "desc" ? "sorted-desc" : "sorted-asc") : "";
        html += `<th class="score-col sortable ${sortClass}" data-sort-col="${criterion}">${criterion.slice(0, 3)}</th>`;
      }
      html += `</tr></thead><tbody>`;

      for (const { idx, t, scores } of indexedData) {
        const llmScore = llmMetrics.length > 0 ? scores[llmMetrics[0]] : null;
        const hasCommentary = llmScore && typeof llmScore === "object" &&
          Object.values(llmScore).some(v => Array.isArray(v) && v[1]);

        html += `<tr class="${hasCommentary ? 'expandable' : ''}" data-row-idx="${idx}"><td>${idx + 1}</td>`;
        if (hasSrc) html += `<td class="text-cell">${this._escapeHtml(t.src)}</td>`;
        html += `<td class="text-cell">${this._escapeHtml(t.trg)}</td>`;
        if (hasRef) html += `<td class="text-cell">${this._escapeHtml(t.ref)}</td>`;

        for (const metric of regularMetrics) {
          const val = scores[metric];
          if (typeof val === "number") {
            const baselineVal = googleV2Scores?.[metric]?.[idx];
            if (this.diffMode && typeof baselineVal === "number") {
              const diff = val - baselineVal;
              const sign = diff >= 0 ? "+" : "";
              html += `<td class="score-col"><span class="segment-score metric-score">${sign}${diff.toFixed(2)}</span></td>`;
            } else {
              html += `<td class="score-col"><span class="segment-score metric-score">${val.toFixed(2)}</span></td>`;
            }
          } else {
            html += `<td class="score-col">-</td>`;
          }
        }

        if (llmScore && typeof llmScore === "object") {
          for (const criterion of llmCriteria) {
            const scoreData = llmScore[criterion];
            const [score, explanation] = Array.isArray(scoreData) ? scoreData : [scoreData, ""];
            if (score !== undefined && score !== null) {
              const scoreClass = `llm-score-${Math.round(score)}`;
              html += `<td class="score-col"><span class="segment-score ${scoreClass}">${score}</span></td>`;
            } else {
              html += `<td class="score-col">-</td>`;
            }
          }
        } else {
          for (const criterion of llmCriteria) {
            html += `<td class="score-col">-</td>`;
          }
        }
        html += `</tr>`;

        if (hasCommentary) {
          html += `<tr class="commentary-row hidden" data-commentary-for="${idx}">
            <td colspan="${totalCols}">
              <div class="commentary-content">`;
          for (const [criterion, scoreData] of Object.entries(llmScore)) {
            const [score, explanation] = Array.isArray(scoreData) ? scoreData : [scoreData, ""];
            if (explanation) {
              const scoreClass = `llm-score-${Math.round(score)}`;
              html += `<div class="commentary-item">
                <span class="commentary-criterion">${criterion}</span>
                <span class="segment-score ${scoreClass}">${score}</span>
                <span class="commentary-text">${this._escapeHtml(explanation)}</span>
              </div>`;
            }
          }
          html += `</div></td></tr>`;
        }
      }
      html += `</tbody></table>`;
    }

    content.innerHTML = html;

    content.querySelectorAll("th.sortable").forEach((th) => {
      th.addEventListener("click", () => {
        const col = th.dataset.sortCol;
        const newDir = sortCol === col && sortDir === "desc" ? "asc" : "desc";
        this._renderOverlayContent(entry, metricName, translations, allScores, col, newDir, googleV2Scores);
      });
    });

    content.querySelectorAll("tr.expandable").forEach((tr) => {
      tr.addEventListener("click", () => {
        const idx = tr.dataset.rowIdx;
        const commentaryRow = content.querySelector(`tr[data-commentary-for="${idx}"]`);
        if (commentaryRow) {
          commentaryRow.classList.toggle("hidden");
          tr.classList.toggle("expanded");
        }
      });
    });
  }

  _escapeHtml(text) {
    if (!text) return "";
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  _getModelRegistryLink(entry) {
    if (entry.translator !== "bergamot") return null;
    const modelName = entry.model_name;
    if (!modelName) return null;

    const experimentName = modelName.replace(/_[A-Za-z0-9_-]{22}$/, "");
    const langpair = `${this.srcLang}-${this.trgLang}`;
    const params = new URLSearchParams({
      searchString: "",
      showModels: "true",
      modelName: experimentName,
      modelLangpair: langpair,
      modelModelName: "student_exported",
      score: "vs-google",
    });
    return `https://mozilla.github.io/translations/model-registry/?${params.toString()}`;
  }
}

class App {
  constructor() {
    this.db = null;
    this.urlState = new URLStateManager();
    this.datasetLeaderboards = [];
    this.diffMode = false;
  }

  async init() {
    try {
      this.db = await Database.open();
      this._setupUI();
      this._setupEventListeners();
      this._loadInitialData();
    } catch (error) {
      document.getElementById("loading").style.display = "none";
      document.getElementById("error").textContent = `Error: ${error.message}`;
    }
  }

  _setupUI() {
    const langpairs = this.db.getLanguagePairs();
    const langpairSelect = document.getElementById("langpair-select");
    langpairSelect.innerHTML = langpairs.map((lp) => `<option value="${lp}">${lp}</option>`).join("");

    if (langpairs.length === 0) {
      document.getElementById("loading").textContent = "No evaluation data available.";
      return;
    }

    document.getElementById("loading").style.display = "none";
    document.getElementById("controls-container").style.display = "block";
    document.getElementById("datasets-container").style.display = "block";
  }

  _setupEventListeners() {
    document.getElementById("langpair-select").addEventListener("change", (e) => {
      this.urlState.update("langpair", e.target.value);
      this._renderAllDatasets();
    });

    document.getElementById("diff-mode-toggle").addEventListener("change", (e) => {
      this.diffMode = e.target.checked;
      for (const leaderboard of this.datasetLeaderboards) {
        leaderboard.setDiffMode(this.diffMode);
        leaderboard.render();
      }
    });

    document.getElementById("overlay-close").addEventListener("click", () => {
      document.getElementById("overlay").classList.remove("visible");
    });

    document.getElementById("overlay").addEventListener("click", (e) => {
      if (e.target.id === "overlay") {
        document.getElementById("overlay").classList.remove("visible");
      }
    });
  }

  _loadInitialData() {
    const langpairs = this.db.getLanguagePairs();
    if (langpairs.length === 0) return;

    const initialLangpair = this.urlState.state.langpair || langpairs[0];
    document.getElementById("langpair-select").value = initialLangpair;
    this.urlState.state.langpair = initialLangpair;

    this._renderAllDatasets();
  }

  _renderAllDatasets() {
    const [src, trg] = (this.urlState.state.langpair || "").split("-");
    const datasets = this.db.getDatasets(src, trg);
    const container = document.getElementById("datasets-container");

    container.innerHTML = "";
    this.datasetLeaderboards = [];

    for (const dataset of datasets) {
      const section = document.createElement("div");
      container.appendChild(section);

      const data = this.db.getLeaderboard(src, trg, dataset);
      const leaderboard = new DatasetLeaderboard(section, dataset, data, src, trg);
      leaderboard.render();
      this.datasetLeaderboards.push(leaderboard);
    }
  }
}

const app = new App();
app.init();