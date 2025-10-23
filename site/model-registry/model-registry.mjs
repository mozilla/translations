// @ts-check
import {
  create,
  exposeAsGlobal,
  formatBytes,
  getElement,
  isNever,
  jsonToYAML,
  parseSearchQuery,
} from "../utils.mjs";

/**
 * @import {
 *   Corpus,
 *   TrainingRun,
 *   ModelRun,
 *   ModelReference,
 *   ModelName,
 *   WordAlignedCorpus
 * } from '../@types/training-run.d.ts'
 */

const BUCKET_NAME = "moz-fx-translations-data--303e-prod-translations-data";
const STORAGE_URL = `https://storage.googleapis.com/${BUCKET_NAME}`;
const DOCS_URL = "https://mozilla.github.io/translations/docs";
const REGISTRY_URL = "https://mozilla.github.io/translations/model-registry";
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

const elements = {
  table: getElement("table", HTMLTableElement),
  tbody: getElement("table-body"),
  thead: getElement("table-thead"),
  tableContainer: getElement("table-container", HTMLDivElement),
  loading: getElement("loading"),
  error: getElement("error"),
  searchFilter: getElement("search-filter", HTMLInputElement),
  overlay: getElement("overlay"),
  overlayCloseButton: getElement("overlay-close-button"),
  overlayContent: getElement("overlay-content"),
  scrollContainer: getElement("scroll-container"),
  scores: getElement("scores"),
};

class ModelUtils {
  static displayName = new Intl.DisplayNames("en", { type: "language" });

  /** @param {string | null | undefined} modelName @returns {ModelName | null} */
  static toModelName(modelName) {
    switch (modelName) {
      case "backwards":
      case "teacher_1":
      case "teacher_2":
      case "student":
      case "student_finetuned":
      case "student_quantized":
      case "student_exported":
        return modelName;
      default:
        return null;
    }
  }

  /** @param {ModelName} modelName */
  static getLabel(modelName) {
    switch (modelName) {
      case "backwards":
        return "Backwards";
      case "teacher_1":
        return "Teacher 1";
      case "teacher_2":
        return "Teacher 2";
      case "student":
        return "Student";
      case "student_finetuned":
        return "Student Finetuned";
      case "student_quantized":
        return "Student Quantized";
      case "student_exported":
        return "Student Exported";
      default:
        isNever(modelName);
        throw new Error("Could not convert model name to label: " + modelName);
    }
  }

  static getLanguageTag(source_lang, target_lang) {
    return source_lang === "en" ? target_lang : source_lang;
  }

  static getDisplayName(languageTag) {
    return ModelUtils.displayName.of(languageTag) ?? languageTag;
  }
}

class ScoreManager {
  static setupHandlers() {
    for (const radio of elements.scores.querySelectorAll("input[type=radio]")) {
      radio.addEventListener("change", () => {
        urlStateManager.update({ score: ScoreManager.getChecked() });
      });
    }
  }

  static getChecked() {
    for (const input of elements.scores.querySelectorAll("input")) {
      if (input.checked) return input.id.replace("score-", "");
    }
    return "vs-google";
  }

  /** @param {TrainingRun} trainingRun @param {ModelRun} [modelRun] */
  static getGoogleFloresComparison(trainingRun, modelRun) {
    const googleFlores = trainingRun.comet_flores_comparison?.google;
    if (!googleFlores || !modelRun?.flores?.comet) return null;

    const percentage = 100 * (1 - googleFlores / (modelRun.flores.comet / 100));
    const sign = percentage >= 0 ? "+" : "";
    return {
      percentage,
      difference: `${sign}${percentage.toFixed(2)}`,
      score: `${(googleFlores * 100).toFixed(2)}`,
    };
  }
}

/** @typedef {ReturnType<typeof URLStateManager.prototype.getInitialState>} State */

class URLStateManager {
  /** @type {State} */
  state = this.getInitialState();

  constructor() {
    addEventListener("popstate", (event) => {
      this.state = event.state.state;
      this.updateUI();
    });
  }

  getInitialState() {
    const urlParams = new URLSearchParams(window.location.search);

    /** @type {ModelReference | null} */
    let modelReference = null;
    const name = urlParams.get("modelName");
    const langpair = urlParams.get("modelLangpair");
    const modelName = ModelUtils.toModelName(urlParams.get("modelModelName"));

    if (name && langpair && modelName) {
      modelReference = { name, langpair, modelName };
    }

    return {
      searchString: urlParams.get("searchString") ?? "",
      showModels: urlParams.get("showModels") === "true",
      showCorpora: urlParams.get("showCorpora") === "true",
      score: urlParams.get("score") || "vs-google",
      modelReference,
    };
  }

  stateToURLSearchParams() {
    const urlParams = new URLSearchParams();
    urlParams.set("searchString", this.state.searchString);
    if (this.state.showModels) urlParams.set("showModels", "true");
    if (this.state.showCorpora) urlParams.set("showCorpora", "true");
    if (this.state.modelReference) {
      urlParams.set("modelName", this.state.modelReference.name);
      urlParams.set("modelLangpair", this.state.modelReference.langpair);
      urlParams.set("modelModelName", this.state.modelReference.modelName);
    }
    urlParams.set("score", this.state.score);
    return urlParams;
  }

  /** @param {Partial<State>} partialState */
  replaceState(partialState) {
    this.state = { ...this.state, ...partialState };
  }

  /** @param {Partial<State>} partialState */
  update(partialState) {
    this.replaceState(partialState);
    this.pushHistory();
    this.updateUI();
  }

  pushHistory() {
    const urlParams = this.stateToURLSearchParams();
    const url = new URL(window.location.href);
    const newLocation = `${url.origin}${url.pathname}?${urlParams}`;
    history.pushState(urlStateManager, "", newLocation);
  }

  updateUI() {
    SearchFilter.onStateChange(this.state.searchString);
    ModelCardOverlay.onStateChange(this.state.modelReference);

    elements.table.classList.toggle("show-models", this.state.showModels);
    elements.table.classList.toggle("show-corpora", this.state.showCorpora);
    elements.searchFilter.value = this.state.searchString;

    const scoreRadio = elements.scores.querySelector("#score-" + this.state.score);
    if (scoreRadio) scoreRadio.setAttribute("checked", "");
    document.body.dataset["score"] = this.state.score;
  }
}

const urlStateManager = new URLStateManager();
exposeAsGlobal("urlStateManager", urlStateManager);

/** @type {TrainingRun[] | null} */
let trainingRuns = null;

document.addEventListener("DOMContentLoaded", async () => {
  TableSorter.setupHandlers();

  trainingRuns = await loadTrainingRuns();
  exposeAsGlobal("trainingRuns", trainingRuns);

  SearchFilter.setupHandlers();
  ModelCardOverlay.setupHandlers();
  ScoreManager.setupHandlers();

  urlStateManager.updateUI();
  TableSorter.sortByDate();

  elements.tableContainer.style.display = "block";
  elements.loading.style.display = "none";
});

class TableSorter {
  static prevColumnIndex = -1;
  static prevDirection = 1;

  static setupHandlers() {
    elements.table.querySelectorAll("th button").forEach((button, index) => {
      button.addEventListener("click", () => TableSorter.sort(index));
    });
  }

  static sortByDate() {
    const tr = elements.thead.querySelector("tr");
    if (!tr) throw new Error("Could not find the tr");
    for (let index = 0; index < tr.children.length; index++) {
      if (tr.children[index].getAttribute("data-key") === "date") {
        TableSorter.sort(index, -1);
        break;
      }
    }
  }

  static sort(columnIndex, defaultDirection = 1) {
    const rows = Array.from(elements.tbody.children);
    const direction =
      TableSorter.prevColumnIndex === columnIndex ? -TableSorter.prevDirection : defaultDirection;
    TableSorter.prevDirection = direction;
    TableSorter.prevColumnIndex = columnIndex;

    rows.sort((rowA, rowB) => {
      const valueA = rowA.querySelectorAll("td")[columnIndex].innerText;
      const valueB = rowB.querySelectorAll("td")[columnIndex].innerText;
      return String(valueA).localeCompare(String(valueB)) * direction;
    });

    rows.forEach((row) => elements.tbody.appendChild(row));
  }
}

class SearchFilter {
  static setupHandlers() {
    elements.searchFilter.addEventListener("keyup", () => {
      SearchFilter.onStateChange(elements.searchFilter.value);
    });

    const pushSearchFilter = () => {
      urlStateManager.replaceState({ searchString: elements.searchFilter.value });
      urlStateManager.pushHistory();
    };

    elements.searchFilter.addEventListener("keyup", (event) => {
      if (event.key === "Enter") pushSearchFilter();
    });
    elements.searchFilter.addEventListener("blur", pushSearchFilter);
  }

  /** @param {string} search */
  static onStateChange(search) {
    search = search.trim();
    const trs = Array.from(elements.tbody.querySelectorAll("tr"));

    for (const tr of trs) tr.style.display = "table-row";
    if (!search) return;

    const { filters, terms } = parseSearchQuery(search);

    for (const tr of elements.tbody.querySelectorAll("tr")) {
      const rowText = tr.innerText.toLowerCase();
      for (const term of terms) {
        if (!rowText.includes(term)) {
          tr.style.display = "none";
          break;
        }
      }
    }

    for (const filter of filters) {
      if (!filter.key.match(/^[a-z-]+$/)) continue;

      const ths = elements.thead.querySelectorAll("th");
      let columnIndex = null;
      for (let i = 0; i < ths.length; i++) {
        if (ths[i].dataset.key === filter.key) {
          columnIndex = i;
          break;
        }
      }
      if (columnIndex === null) continue;

      for (const tr of trs) {
        const td = /** @type {HTMLElement} */ (tr.children[columnIndex]);
        const rowText = td.innerText.toLowerCase();
        if (filter.negated) {
          if (rowText.includes(filter.value)) tr.style.display = "none";
        } else if (!rowText.includes(filter.value)) {
          tr.style.display = "none";
        }
      }
    }
  }
}

class DetailsTable {
  constructor(parent, title) {
    this.tbody = create.tbody();
    create.li({
      parent,
      children: [
        title,
        create.table({
          className: "details-table",
          children: [
            create.thead({
              children: [
                create.tr({
                  children: [
                    create.th({ children: "Label" }),
                    create.th({ children: "Details" }),
                  ],
                }),
              ],
            }),
            this.tbody,
          ],
        }),
      ],
    });
  }

  addRow(label, value) {
    create.tr({
      parent: this.tbody,
      children: [create.td({ children: label }), create.td({ children: value || "-" })],
    });
  }
}

class ModelCardOverlay {
  /** @type {TrainingRun} */
  trainingRun;
  /** @type {ModelRun} */
  modelRun;
  /** @type {ModelReference} */
  modelReference;

  constructor(trainingRun, modelRun, modelReference) {
    this.trainingRun = trainingRun;
    this.modelRun = modelRun;
    this.modelReference = modelReference;
  }

  static setupHandlers() {
    const hideOverlay = () => urlStateManager.update({ modelReference: null });
    elements.overlayCloseButton.addEventListener("click", hideOverlay);
    document.body.addEventListener("keyup", (event) => {
      if (event.key === "Escape") hideOverlay();
    });
    elements.overlay.addEventListener("click", (event) => {
      if (event.target === elements.overlay) hideOverlay();
    });
  }

  /** @param {ModelReference | null} modelReference */
  static onStateChange(modelReference) {
    if (!modelReference) {
      document.body.classList.remove("overlay-show");
      elements.scrollContainer.removeAttribute("inert");
      return null;
    }
    if (!trainingRuns || document.body.classList.contains("overlay-show")) {
      return null;
    }

    const { name, langpair, modelName } = modelReference;
    const trainingRun = trainingRuns.find(
      (tr) => tr.name === name && tr.langpair === langpair
    );

    if (!trainingRun) {
      elements.error.style.display = "block";
      elements.error.innerText = `Could not find the model "${name}" (${langpair})`;
      return null;
    }

    const modelRun = trainingRun[modelName];
    if (!modelRun) {
      elements.error.style.display = "block";
      elements.error.innerText = `That model couldn't be found for "${name}" (${langpair})`;
      return null;
    }

    new ModelCardOverlay(trainingRun, modelRun, modelReference).initialize();
  }

  initialize() {
    elements.overlayContent.innerText = "";
    this.createHeaders();
    const detailsUL = create.ul({ parent: elements.overlayContent });
    this.initModelDetails(detailsUL);
    this.initArtifacts(detailsUL);
    this.initTrainingContinuation();
    this.initTrainingConfig();
    elements.scrollContainer.setAttribute("inert", "");
    document.body.classList.add("overlay-show");
  }

  createHeaders() {
    const { name, langpair, modelName } = this.modelReference;
    create.h1({ children: `${name} (${langpair})`, parent: elements.overlayContent });
    create.h2({ children: ModelUtils.getLabel(modelName), parent: elements.overlayContent });
  }

  initModelDetails(parent) {
    const table = new DetailsTable(parent, "Model Details");
    const { task_group_id: taskGroupId, task_id: taskId } = this.modelRun;
    const { langpair, name } = this.trainingRun;

    table.addRow("Date", (this.modelRun.date || "").slice(0, 10));
    table.addRow(
      "TaskGroup",
      create.a({
        children: this.modelRun.task_group_id,
        href: `https://firefox-ci-tc.services.mozilla.com/tasks/groups/${taskGroupId}`,
      })
    );
    table.addRow(
      "Task",
      create.a({
        children: this.modelRun.task_name,
        href: `https://firefox-ci-tc.services.mozilla.com/tasks/${taskId}`,
      })
    );
    table.addRow("W&B", this.createWandBLinks(langpair, name, taskGroupId));
  }

  createWandBLinks(langpair, name, taskGroupId) {
    const modelName = this.modelReference.modelName.replace("_", "-");
    const idPart = (this.modelRun.task_group_id || "").slice(0, 6);

    return create.ul({
      children: [
        create.li({
          children: create.a({
            children: "Model Run",
            href: `https://wandb.ai/moz-translations/${langpair}/runs/${modelName}_${idPart}`,
          }),
        }),
        create.li({
          children: create.a({
            children: `Task Group ${taskGroupId}`,
            href: `https://wandb.ai/moz-translations/${langpair}/groups/${name}_${taskGroupId}/workspace`,
          }),
        }),
        create.li({
          children: [
            create.a({
              children: langpair,
              href: `https://wandb.ai/moz-translations/${langpair}/`,
            }),
            create.div({
              className: "wandb-filter",
              children: [
                "Group by ",
                create.span({ children: "Group" }),
                ", filter by ",
                create.span({ children: name }),
              ],
            }),
            create.div({
              className: "wandb-filter",
              children: [
                `Or filter by regex: `,
                create.span({
                  children: this.trainingRun.task_group_ids.map((t) => t.slice(0, 6)).join("|"),
                }),
              ],
            }),
          ],
        }),
      ],
    });
  }

  initArtifacts(parent) {
    create.li({ parent, children: "Artifacts" });
    const artifactsUL = create.ul({ parent });
    for (const url of this.modelRun.artifact_urls || []) {
      const fileName = url.split("/").pop();
      create.li({
        parent: artifactsUL,
        children: create.a({ children: fileName, href: url }),
      });
    }
  }

  initTrainingContinuation() {
    const { name, langpair, modelName } = this.modelReference;
    const continuations = new Continuations(this.trainingRun, this.modelRun);

    switch (modelName) {
      case "backwards":
        continuations.createSection({
          header: "Re-use this model for backtranslations",
          lines: [
            continuations.docs([
              `Use the "${langpair}" model from the "${name}" training run for back translations.`,
            ]),
            continuations.vocab(),
            "  models:",
            continuations.model("backwards", "use", this.modelRun),
          ],
        });
        break;
      case "teacher_1":
      case "teacher_2": {
        const corpora = [
          ...continuations.backTranslationsCorpus(),
          ...continuations.parallelCorpus(),
        ];

        if (corpora.length) {
          corpora.unshift("  corpora:");
          continuations.createSection({
            header: "Train a new teacher with the existing corpora",
            lines: [
              continuations.docs([
                `Train a new teacher ${langpair} model from the "${name}" training run.`,
              ]),
              continuations.vocab(),
              corpora,
              "  models:",
              continuations.backwardsModel(),
            ],
          });
        }

        continuations.createSection({
          header: "Generate new distillation data and train a new student",
          lines: [
            continuations.docs([
              `Use the existing ${langpair} model from the "${name}" training run to `,
              "generate more distillation data",
            ]),
            corpora,
            continuations.vocab(),
            "  models:",
            continuations.ensemble("train-teacher", "use"),
            continuations.backwardsModel(),
          ],
        });

        continuations.createSection({
          header: "Fine-tune the teacher",
          lines: [
            continuations.docs([
              `Fine tune the ${langpair} model from the "${name}" training run.`,
            ]),
            corpora,
            continuations.vocab(),
            "  models:",
            continuations.ensemble("train-teacher", "continue"),
            continuations.backwardsModel(),
          ],
        });

        const distillationCorpus = continuations.distillationCorpus();
        if (distillationCorpus.length) {
          continuations.createSection({
            header: "Train new student from this teacher's distillation data",
            lines: [
              continuations.docs([
                `Train a new ${langpair} student from just the disillation corpus from `,
                `the "${name}" training run.`,
              ]),
              continuations.vocab(),
              "  corpora:",
              distillationCorpus,
              "  models:",
              continuations.backwardsModel(),
            ],
          });
        }
        break;
      }
      case "student": {
        const distillationCorpus = continuations.distillationCorpus();
        if (distillationCorpus.length) {
          continuations.createSection({
            header: "Train new student",
            lines: [
              continuations.docs([
                `Train a new ${langpair} student from the "${name}" training run.`,
              ]),
              continuations.vocab(),
              "  corpora:",
              distillationCorpus,
              "models:",
              continuations.backwardsModel(),
            ],
          });
        }

        continuations.createSection({
          header: "Train teacher using this model for back translations",
          lines: [
            continuations.docs([
              `Use the ${langpair} model from the "${name}" training run for back translations.`,
            ]),
            "models:",
            continuations.model("backwards", "use", this.modelRun),
          ],
        });
        break;
      }
      case "student_finetuned":
      case "student_quantized":
      case "student_exported":
        break;
      default:
        isNever(modelName);
    }
  }

  initTrainingConfig() {
    create.h2({ parent: elements.overlayContent, children: "Training Config" });

    const taskGroupId = this.modelRun.task_group_id;
    let config = this.trainingRun.experiment_config || {};

    if (taskGroupId) {
      const taskGroupConfig = Database.instance.queryOne(
        `SELECT experiment_config FROM task_groups WHERE task_group_id=?`,
        [taskGroupId]
      );
      if (taskGroupConfig?.experiment_config) {
        try {
          config = JSON.parse(taskGroupConfig.experiment_config);
        } catch (e) {
          console.log(`Failed to parse task group config:`, e);
        }
      }
    }

    create.pre({
      parent: elements.overlayContent,
      children: jsonToYAML(config),
    });
  }
}



class Database {
  static instance = null;
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
    const resp = await fetch(cacheBustUrl, {
      cache: "no-store",
      headers: { "Cache-Control": "no-cache, no-store, must-revalidate" },
    });
    if (!resp.ok) {
      throw new Error(`Failed to fetch DB: ${resp.status} ${resp.statusText}`);
    }
    const buf = await resp.arrayBuffer();
    const db = new Database(new SQL.Database(new Uint8Array(buf)));
    Database.instance = db;
    return db;
  }

  queryAll(sql, params = []) {
    const stmt = this.db.prepare(sql);
    try {
      if (params && params.length) stmt.bind(params);
      const out = [];
      while (stmt.step()) out.push(stmt.getAsObject());
      return out;
    } finally {
      stmt.free();
    }
  }

  queryOne(sql, params = []) {
    const rows = this.queryAll(sql, params);
    return rows.length ? rows[0] : null;
  }

  getComparisons(runId, metric) {
    const results = {};
    for (const row of this.queryAll(
      `SELECT provider, score FROM run_comparisons WHERE run_id=? AND metric=?`,
      [runId, metric]
    )) {
      results[row.provider] = row.score;
    }
    return results;
  }

  getCorpus(runId, type, aligned) {
    const row = this.queryOne(
      `SELECT type, aligned, source_url, target_url, alignments_url,
              source_bytes, target_bytes, alignments_bytes
       FROM corpora WHERE run_id=? AND type=? AND aligned=? LIMIT 1`,
      [runId, type, aligned ? 1 : 0]
    );
    if (!row) return null;

    if (row.aligned) {
      return {
        source_url: row.source_url,
        target_url: row.target_url,
        alignments_url: row.alignments_url,
        source_bytes: row.source_bytes ?? 0,
        target_bytes: row.target_bytes ?? 0,
        alignments_bytes: row.alignments_bytes ?? 0,
      };
    }
    return {
      source_url: row.source_url,
      target_url: row.target_url,
      source_bytes: row.source_bytes ?? 0,
      target_bytes: row.target_bytes ?? 0,
    };
  }

  getModel(runId, kind) {
    const m = this.queryOne(
      `SELECT m.id, m.date, m.task_group_id, m.task_id, t.task_name, m.artifact_folder
       FROM models m
       LEFT JOIN tasks t ON m.task_id = t.task_id
       WHERE m.run_id=? AND m.kind=? LIMIT 1`,
      [runId, kind]
    );
    if (!m) return null;

    const e = this.queryOne(
      `SELECT chrf, bleu, comet FROM evaluations WHERE model_id=?`,
      [m.id]
    );

    const artifacts = this.queryAll(
      `SELECT url FROM artifacts WHERE model_id=?`,
      [m.id]
    ).map((x) => x.url);

    let config = {};
    try {
      const maybe = this.queryOne(
        `SELECT json(config_json) AS cfg FROM models WHERE id=?`,
        [m.id]
      );
      if (maybe?.cfg) {
        try {
          config = JSON.parse(maybe.cfg);
        } catch {}
      }
    } catch {}

    return {
      date: m.date || null,
      config,
      task_group_id: m.task_group_id || null,
      task_id: m.task_id || null,
      task_name: m.task_name || null,
      flores: e ? { chrf: e.chrf ?? null, bleu: e.bleu ?? null, comet: e.comet ?? null } : null,
      artifact_folder: m.artifact_folder || null,
      artifact_urls: artifacts,
    };
  }

  getTaskGroupIds(runId) {
    return this.queryAll(
      `SELECT DISTINCT task_group_id FROM models WHERE run_id=? AND task_group_id IS NOT NULL`,
      [runId]
    ).map((x) => x.task_group_id);
  }
}

class TrainingRunLoader {
  constructor(db) {
    this.db = db;
  }

  async loadAll() {
    const runs = this.fetchRuns();
    return runs.map((r) => this.buildTrainingRun(r));
  }

  fetchRuns() {
    return this.db.queryAll(
      `SELECT id, name, source_lang, target_lang, date_created
       FROM training_runs
       ORDER BY COALESCE(date_created, '') DESC`
    );
  }

  buildTrainingRun(r) {
    const runId = r.id;

    const taskGroupConfigs = this.db.queryAll(
      `SELECT task_group_id, experiment_config FROM task_groups WHERE run_id=?`,
      [runId]
    );

    let experimentConfig = {};
    if (taskGroupConfigs.length > 0 && taskGroupConfigs[0].experiment_config) {
      try {
        experimentConfig = JSON.parse(taskGroupConfigs[0].experiment_config);
      } catch (e) {
        console.log(`Failed to parse experiment config for ${r.name}:`, e);
      }
    }

    /** @type {TrainingRun} */
    const tr = {
      name: r.name,
      langpair: `${r.source_lang}-${r.target_lang}`,
      source_lang: r.source_lang,
      target_lang: r.target_lang,
      task_group_ids: this.db.getTaskGroupIds(runId),
      date_started: r.date_created || null,
      experiment_config: experimentConfig,
      comet_flores_comparison: this.db.getComparisons(runId, "comet"),
      bleu_flores_comparison: this.db.getComparisons(runId, "bleu"),
      parallel_corpus_aligned: /** @type {any} */ (this.db.getCorpus(runId, "parallel", true)),
      backtranslations_corpus_aligned: /** @type {any} */ (
        this.db.getCorpus(runId, "backtranslations", true)
      ),
      distillation_corpus_aligned: /** @type {any} */ (
        this.db.getCorpus(runId, "distillation", true)
      ),
      parallel_corpus: /** @type {any} */ (this.db.getCorpus(runId, "parallel", false)),
      backtranslations_corpus: /** @type {any} */ (
        this.db.getCorpus(runId, "backtranslations", false)
      ),
      distillation_corpus: /** @type {any} */ (this.db.getCorpus(runId, "distillation", false)),
      backwards: /** @type {any} */ (
        this.db.getModel(runId, "backward") || this.db.getModel(runId, "backwards")
      ),
      teacher_1: /** @type {any} */ (this.db.getModel(runId, "teacher_1")),
      teacher_2: /** @type {any} */ (this.db.getModel(runId, "teacher_2")),
      student: /** @type {any} */ (this.db.getModel(runId, "student")),
      student_finetuned: /** @type {any} */ (this.db.getModel(runId, "student_finetuned")),
      student_quantized: /** @type {any} */ (this.db.getModel(runId, "student_quantized")),
      student_exported: /** @type {any} */ (this.db.getModel(runId, "student_exported")),
    };

    try {
      new TrainingRunRow(tr).build();
    } catch (error) {
      elements.error.style.display = "block";
      elements.error.innerText = "Error building training run row.";
      console.error(error);
    }

    return tr;
  }
}

/** @returns {Promise<TrainingRun[]>} */
async function loadTrainingRuns() {
  const db = await Database.open();
  const loader = new TrainingRunLoader(db);
  return loader.loadAll();
}

class TrainingRunRow {
  /** @type {TrainingRun} */
  trainingRun;
  /** @type {HTMLTableRowElement} */
  tr;

  /** @param {TrainingRun} trainingRun */
  constructor(trainingRun) {
    this.trainingRun = trainingRun;
    this.tr = create.tr({ parent: elements.tbody });
  }

  build() {
    this.createInitialColumns();
    this.createModelButtons();
    this.createCorporaLinks();
  }

  createInitialColumns() {
    const { source_lang, target_lang, name, langpair, date_started } = this.trainingRun;
    const languageTag = ModelUtils.getLanguageTag(source_lang, target_lang);

    this.createFilterableButton("name", name);
    this.createFilterableButton("language", ModelUtils.getDisplayName(languageTag));
    this.createFilterableButton("langpair", langpair);
    create.td({
      parent: this.tr,
      children: ((date_started ?? "–") || "–").slice(0, 10),
    });
  }

  /** @param {string} key @param {string} value */
  createFilterableButton(key, value) {
    create.td({
      parent: this.tr,
      children: create.button({
        className: "button-text",
        children: value,
        onClick() {
          elements.searchFilter.value = value.includes(" ")
            ? `${key}:"${value}"`
            : `${key}:${value}`;
          urlStateManager.update({ searchString: elements.searchFilter.value });
        },
      }),
    });
  }

  /** @param {ModelName} modelName */
  createModelOverlayButton(modelName) {
    const { trainingRun } = this;
    const modelRun = trainingRun[modelName];
    const googleFlores = ScoreManager.getGoogleFloresComparison(trainingRun, modelRun);
    const comet = modelRun?.flores?.comet;
    const bleu = modelRun?.flores?.bleu;

    const style = {};
    let shippable = "Shippable";
    if (googleFlores && googleFlores.percentage < -5) {
      style.background = "#ffa537";
      shippable = "Not shippable";
    }

    const title =
      `${shippable} - COMET ${comet?.toFixed?.(2)} ` +
      `vs Google Comet ${googleFlores?.score} (${googleFlores?.difference})`;

    create.td({
      parent: this.tr,
      className: "models-td",
      style,
      children: create.div({
        children: !modelRun
          ? "–"
          : create.button({
              title,
              className: "button-text",
              children: [
                create.span({
                  className: "score-vs-google",
                  children: googleFlores ? googleFlores.difference : "view",
                }),
                create.span({
                  className: "score-comet",
                  children: comet != null ? Number(comet).toFixed(2) : "view",
                }),
                create.span({
                  className: "score-bleu",
                  children: bleu != null ? Number(bleu).toFixed(2) : "view",
                }),
              ],
              onClick() {
                urlStateManager.update({
                  modelReference: {
                    name: trainingRun.name,
                    langpair: trainingRun.langpair,
                    modelName,
                  },
                });
              },
            }),
      }),
    });
  }

  createModelButtons() {
    create.td({
      parent: this.tr,
      children: create.button({
        onClick() {
          urlStateManager.update({ showModels: !urlStateManager.state.showModels });
        },
        children: [
          create.span({ className: "toggle-models-show", children: "Show" }),
          create.span({ className: "toggle-models-hide", children: "Hide" }),
        ],
      }),
    });

    ["backwards", "teacher_1", "teacher_2", "student", "student_finetuned", "student_quantized", "student_exported"].forEach((model) =>
      this.createModelOverlayButton(model)
    );
  }

  /** @param {Corpus} [corpus] */
  createCorpusLink(corpus) {
    const { source_lang, target_lang } = this.trainingRun;
    create.td({
      parent: this.tr,
      className: "corpus-td",
      children: corpus
        ? [
            create.a({
              children: source_lang,
              href: corpus.source_url,
              title: formatBytes(corpus.source_bytes),
            }),
            create.a({
              children: target_lang,
              href: corpus.target_url,
              title: formatBytes(corpus.target_bytes),
            }),
          ]
        : "–",
    });
  }

  createCorporaLinks() {
    const { trainingRun } = this;

    create.td({
      parent: this.tr,
      children: create.button({
        onClick() {
          urlStateManager.update({ showCorpora: !urlStateManager.state.showCorpora });
        },
        children: [
          create.span({ className: "toggle-corpora-show", children: "Show" }),
          create.span({ className: "toggle-corpora-hide", children: "Hide" }),
        ],
      }),
    });

    this.createCorpusLink(
      /** @type {Corpus} */ (trainingRun.parallel_corpus_aligned || trainingRun.parallel_corpus)
    );
    this.createCorpusLink(
      /** @type {Corpus} */ (trainingRun.backtranslations_corpus_aligned || trainingRun.backtranslations_corpus)
    );
    this.createCorpusLink(
      /** @type {Corpus} */ (trainingRun.distillation_corpus_aligned || trainingRun.distillation_corpus)
    );
    this.createCorpusLink(trainingRun.parallel_corpus);
    this.createCorpusLink(trainingRun.backtranslations_corpus);
    this.createCorpusLink(trainingRun.distillation_corpus);

    create.td({
      parent: this.tr,
      children: create.button({
        children: "log",
        title: "View this run in the console.log",
        onClick() {
          alert("View this run in the console.log");
          console.log(trainingRun.name, trainingRun.langpair);
          console.log(trainingRun);
        },
      }),
    });
  }
}

class Continuations {
  /** @type {TrainingRun} */
  trainingRun;
  /** @type {ModelRun} */
  modelRun;

  constructor(trainingRun, modelRun) {
    this.trainingRun = trainingRun;
    this.modelRun = modelRun;
  }

  /** @param {string} name @param {Corpus} [corpus] @param {WordAlignedCorpus} [aligned] */
  #corpusYaml(name, corpus, aligned) {
    if (!corpus) return [];

    const lines = [`    ${name}:`, "      src: " + corpus.source_url, "      trg: " + corpus.target_url];

    if (aligned?.alignments_url) {
      lines.push(
        "      tok-src: " + aligned.source_url,
        "      tok-trg: " + aligned.target_url,
        "      alignments: " + aligned.alignments_url
      );
    }
    return lines;
  }

  backTranslationsCorpus() {
    return this.#corpusYaml(
      "backtranslations",
      this.trainingRun.backtranslations_corpus,
      /** @type {WordAlignedCorpus} */ (this.trainingRun.backtranslations_corpus_aligned)
    );
  }

  parallelCorpus() {
    return this.#corpusYaml(
      "parallel",
      this.trainingRun.parallel_corpus,
      /** @type {WordAlignedCorpus} */ (this.trainingRun.parallel_corpus_aligned)
    );
  }

  distillationCorpus() {
    return this.#corpusYaml(
      "distillation",
      this.trainingRun.distillation_corpus,
      /** @type {WordAlignedCorpus} */ (this.trainingRun.distillation_corpus_aligned)
    );
  }

  vocab() {
    const urls = (this.modelRun.artifact_urls || []).filter((url) => url.endsWith(".spm"));
    if (urls.length === 0) return [];

    let srcVocab, trgVocab;
    if (urls.length === 1) {
      srcVocab = trgVocab = urls[0];
    } else {
      const { source_lang, target_lang } = this.trainingRun;
      srcVocab = urls.find((url) => url.endsWith(`vocab.${source_lang}.spm`));
      trgVocab = urls.find((url) => url.endsWith(`vocab.${target_lang}.spm`));
    }

    if (!srcVocab || !trgVocab) return [];
    return ["  vocab:", "    src: " + srcVocab, "    trg: " + trgVocab];
  }

  /** @returns {string[]} */
  backwardsModel() {
    if (this.trainingRun.student && this.trainingRun.backwards) {
      return [
        "    # Use the (typically higher quality) student model for the backwards model:",
        ...this.model("backwards", "use", this.trainingRun.student),
        "    # Or use original backwards model if the student model is not good:",
        ...this.model("backwards", "use", this.trainingRun.backwards)
          .slice(1)
          .map((line) => line.replace("      ", "    #   ")),
      ];
    }

    const model = this.trainingRun.student ?? this.trainingRun.backwards;
    return model ? this.model("backwards", "use", model) : [];
  }

  /** @param {"backwards" | "teacher"} name @param {string} mode @param {ModelRun} [modelRun] */
  model(name, mode, modelRun) {
    if (!modelRun) return [];
    return [
      `    ${name}:`,
      "      url: " + (modelRun.artifact_folder || ""),
      `      mode: ${mode}`,
      "      type: default",
    ];
  }

  /** @param {string} name @param {string} mode */
  ensemble(name, mode) {
    return [
      `    ${name}:`,
      "      urls:",
      "        - " + (this.modelRun.artifact_folder || ""),
      `      mode: ${mode}`,
      "      type: default",
    ];
  }

  headerGenerated = false;

  /** @param {{ header: string, lines: Array<string[] | string> }} options */
  createSection({ header, lines }) {
    if (!this.headerGenerated) {
      this.headerGenerated = true;
      create.h2({ parent: elements.overlayContent, children: "Training Continuation" });
      create.p({
        parent: elements.overlayContent,
        children: [
          "Re-use this model in another training run. See the ",
          create.a({
            children: "training continuation docs",
            href: "../docs/training/using-pretrained-models/",
          }),
          " for more information.",
        ],
      });
    }

    create.h4({ parent: elements.overlayContent, children: header });
    create.pre({
      parent: elements.overlayContent,
      children: "continuation:\n" + lines.flatMap((value) => value).join("\n"),
    });
  }

  /** @param {string[]} lines */
  docs(lines) {
    const params = new URLSearchParams();
    params.set(
      "searchString",
      `name:${this.trainingRun.name} langpair:${this.trainingRun.langpair}`
    );
    return [
      ...lines.map((line) => `  # ${line}`),
      `  #   Docs:     ${DOCS_URL}/training/using-pretrained-models/`,
      `  #   Registry: ${REGISTRY_URL}/?${params}`,
    ];
  }
}
