// @ts-check
import {
  changeLocation,
  replaceLocation,
  exposeAsGlobal,
  getElement,
} from "../utils.mjs";

/**
 * @import { ModelRecord, ReleaseInfo, ModelMetadata } from "../@types/models"
 */

const BUCKET_NAME = "moz-fx-translations-data--303e-prod-translations-data";
const STORAGE_URL = `https://storage.googleapis.com/${BUCKET_NAME}`;
const DEFAULT_DB_URL = `${STORAGE_URL}/db/db.sqlite`;

function normalizeLangCode(lang) {
  if (lang === "zh-Hans") return "zh";
  // Convert BCP 47 script tags (e.g., zh-Hant) to underscore format (zh_hant)
  const match = lang.match(/^([a-z]{2,3})-([A-Z][a-z]{3})$/);
  if (match) return `${match[1]}_${match[2].toLowerCase()}`;
  return lang;
}

const displayNames = new Intl.DisplayNames("en", {
  type: "language",
  languageDisplay: "standard",
});
function getDisplayName(lang) {
  const bcp47 = lang.replace(/_/g, "-");
  try {
    return displayNames.of(bcp47) ?? lang;
  } catch {
    return lang;
  }
}

function splitLangPair(pair) {
  if (pair.startsWith("en-")) return ["en", pair.slice(3)];
  if (pair.endsWith("-en")) return [pair.slice(0, -3), "en"];
  return pair.split("-");
}

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
    this._googleScoresCache = null;
    this._exportsByHashCache = null;
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

  getGoogleScores() {
    if (this._googleScoresCache) return this._googleScoresCache;

    this._googleScoresCache = {};
    const rows = this.queryAll(
      `SELECT fe.source_lang, fe.target_lang, fem.metric_name, fem.corpus_score
       FROM final_evals fe
       JOIN final_eval_metrics fem ON fe.id = fem.eval_id
       WHERE fe.dataset = 'flores200-plus'
         AND fe.translator = 'google'
         AND fe.model_name = 'v2'
         AND fem.metric_name IN ('chrf', 'bleu', 'comet22')`
    );
    for (const row of rows) {
      const langpair = `${row.source_lang}-${row.target_lang}`;
      if (!this._googleScoresCache[langpair]) {
        this._googleScoresCache[langpair] = {};
      }
      const metricName = row.metric_name === "comet22" ? "comet" : row.metric_name;
      // comet22 is stored in 0-1 scale, convert to 0-100 for display
      const score = row.metric_name === "comet22" ? row.corpus_score * 100 : row.corpus_score;
      this._googleScoresCache[langpair][metricName] = score;
    }
    return this._googleScoresCache;
  }

  getExportsByHash() {
    if (this._exportsByHashCache) return this._exportsByHashCache;

    this._exportsByHashCache = {};
    const rows = this.queryAll(
      `SELECT ex.hash, ex.architecture, ex.byte_size, ex.model_config, ex.model_statistics,
              fem.metric_name, fem.corpus_score
       FROM exports ex
       JOIN models m ON ex.model_id = m.id
       LEFT JOIN final_evals fe ON fe.model_id = m.id
         AND fe.dataset = 'flores200-plus' AND fe.translator = 'bergamot'
       LEFT JOIN final_eval_metrics fem ON fem.eval_id = fe.id
         AND fem.metric_name IN ('chrf', 'bleu', 'comet22')`
    );

    for (const row of rows) {
      const hash = row.hash;
      if (!this._exportsByHashCache[hash]) {
        let modelConfig = null;
        let modelStatistics = null;
        try {
          if (row.model_config) modelConfig = JSON.parse(row.model_config);
          if (row.model_statistics) modelStatistics = JSON.parse(row.model_statistics);
        } catch {}

        this._exportsByHashCache[hash] = {
          architecture: row.architecture,
          byteSize: row.byte_size,
          hash: row.hash,
          modelConfig,
          modelStatistics,
          flores: {},
        };
      }

      if (row.metric_name && row.corpus_score != null) {
        const metricName = row.metric_name === "comet22" ? "comet" : row.metric_name;
        const score = row.metric_name === "comet22" ? row.corpus_score * 100 : row.corpus_score;
        this._exportsByHashCache[hash].flores[metricName] = score;
      }
    }

    // Convert empty flores objects to null
    for (const entry of Object.values(this._exportsByHashCache)) {
      if (Object.keys(entry.flores).length === 0) {
        entry.flores = null;
      }
    }

    return this._exportsByHashCache;
  }
}

main().catch((error) => {
  console.error(error);
  getElement("error").style.display = "block";
});

const aLessThanB = "a".localeCompare("b");
const aGreaterThanB = aLessThanB * -1;
const aEqualToB = 0;

/**
 * @param {string} url
 * @returns {Promise<any>}
 */
async function fetchJSON(url) {
  const response = await fetch(url);

  if (!response.ok) {
    console.error(response);
    throw new Error("Response failed.");
  }

  return await response.json();
}

function setupRemoteSettingsPreview() {
  const remoteSettingsPreviewCheckbox = /** @type {HTMLInputElement} */ (
    getElement("remoteSettingsPreview")
  );
  const urlParams = new URLSearchParams(window.location.search);
  const isPreview = urlParams.get("preview") === "true";
  remoteSettingsPreviewCheckbox.checked = isPreview;
  remoteSettingsPreviewCheckbox.addEventListener("change", () => {
    const urlParams = new URLSearchParams(window.location.search);
    if (remoteSettingsPreviewCheckbox.checked) {
      urlParams.set("preview", "true");
    } else {
      urlParams.delete("preview");
    }
    changeLocation(urlParams);
  });
  return isPreview;
}

function setupReleaseChannelCheckbox() {
  const releasedModelsCheckbox = /** @type {HTMLInputElement} */ (
    getElement("releasedModels")
  );
  const nightlyModelsCheckbox = /** @type {HTMLInputElement} */ (
    getElement("nightlyModels")
  );
  const urlParams = new URLSearchParams(window.location.search);
  const releasedModelsUrlValue = urlParams.get("releasedModels");
  const isReleasedModels =
    releasedModelsUrlValue === "true" || !releasedModelsUrlValue;

  releasedModelsCheckbox.checked = isReleasedModels;
  releasedModelsCheckbox.addEventListener("change", () => {
    const urlParams = new URLSearchParams(window.location.search);
    if (releasedModelsCheckbox.checked) {
      urlParams.delete("releasedModels");
      nightlyModelsCheckbox.checked = false;
      urlParams.delete("nightlyModels");
    } else {
      urlParams.set("releasedModels", "false");
    }
    changeLocation(urlParams);
  });

  const nightlyModelsUrlValue = urlParams.get("nightlyModels");
  const isNightlyModels = !isReleasedModels && nightlyModelsUrlValue === "true";
  nightlyModelsCheckbox.checked = isNightlyModels;
  nightlyModelsCheckbox.addEventListener("change", () => {
    const urlParams = new URLSearchParams(window.location.search);
    if (nightlyModelsCheckbox.checked) {
      urlParams.set("nightlyModels", "true");
      urlParams.set("releasedModels", "false");
      releasedModelsCheckbox.checked = false;
    } else {
      urlParams.delete("nightlyModels");
    }
    changeLocation(urlParams);
  });

  return [isReleasedModels, isNightlyModels];
}

function setupShowAdditionalDetails() {
  const additionalDetailsCheckbox = /** @type {HTMLInputElement} */ (
    getElement("showAdditionalDetails")
  );
  const urlParams = new URLSearchParams(window.location.search);
  const urlValue = urlParams.get("showAdditionalDetails");
  const showAdditionalDetails = urlValue === "true";
  additionalDetailsCheckbox.checked = showAdditionalDetails;

  if (showAdditionalDetails) {
    document.body.classList.add("showAdditionalDetails");
  }

  additionalDetailsCheckbox.addEventListener("change", () => {
    const urlParams = new URLSearchParams(window.location.search);
    if (additionalDetailsCheckbox.checked) {
      urlParams.set("showAdditionalDetails", "true");
      document.body.classList.add("showAdditionalDetails");
    } else {
      urlParams.delete("showAdditionalDetails");
      document.body.classList.remove("showAdditionalDetails");
    }
    replaceLocation(urlParams);
  });
}

/**
 * @param {ModelRecord[]} models
 */
function getReleasedModels(models) {
  /** @type {Map<string, ModelRecord>} */
  const byKey = new Map();
  for (const model of models) {
    const channels = getReleaseChannels(model);
    if (channels?.release) {
      const langPair = model.sourceLanguage + "-" + model.targetLanguage;
      const key = langPair + "|" + (model.filter_expression ?? "");
      const existingModel = byKey.get(key);
      if (
        !existingModel ||
        versionCompare(model.version, existingModel.version) > 0
      ) {
        byKey.set(key, model);
      }
    }
  }
  return [...byKey.values()];
}

/**
 * @param {ModelRecord[]} models
 */
function getNightlyModels(models) {
  /** @type {Map<string, ModelRecord>} */
  const langPairs = new Map();
  for (const model of models) {
    const langPair = model.sourceLanguage + "-" + model.targetLanguage;
    const existingModel = langPairs.get(langPair);
    if (
      !existingModel ||
      versionCompare(model.version, existingModel.version) > 0
    ) {
      langPairs.set(langPair, model);
    }
  }
  return (models = [...langPairs.values()]);
}

async function main() {
  getElement("counts").style.display = "table";

  const isPreview = setupRemoteSettingsPreview();
  const bucket = isPreview ? "main-preview" : "main";

  setupShowAdditionalDetails();
  const [isReleasedModels, isNightlyModels] = setupReleaseChannelCheckbox();

  const db = await Database.open();
  exposeAsGlobal("db", db);

  /** @type {{ data: ModelRecord[] }} */
  const records = await fetchJSON(
    `https://firefox.settings.services.mozilla.com/v1/buckets/${bucket}/collections/translations-models-v2/records`
  );
  exposeAsGlobal("records", records.data);

  const attachmentsByKey = getAttachmentsByKey(records.data);

  const googleScores = db.getGoogleScores();
  exposeAsGlobal("googleScores", googleScores);

  const exportsByHash = db.getExportsByHash();
  exposeAsGlobal("exportsByHash", exportsByHash);

  /**
   * @typedef {Object} ModelEntry
   * @property {string} lang
   * @property {string} display
   * @property {ModelRecord[]} fromEn
   * @property {ModelRecord[]} toEn
   */

  /** @type {Map<string, ModelEntry>} */
  const modelsMap = new Map();
  let models = records.data.filter((record) => record.fileType === "model");
  const releasedModels = getReleasedModels(models);

  countModels(models, releasedModels);
  logModelTableData(googleScores, models, exportsByHash);

  if (isReleasedModels) {
    models = releasedModels;
  } else if (isNightlyModels) {
    models = getNightlyModels(models);
  }
  exposeAsGlobal("models", models);

  /**
   * @param {ModelRecord} model
   */
  function getModelKey(model) {
    const lang =
      model.sourceLanguage === "en" ? model.targetLanguage : model.sourceLanguage;
    if (isNightlyModels || isReleasedModels) {
      return lang;
    }
    return lang + " " + model.version;
  }

  for (const model of models) {
    const key = getModelKey(model);
    let entry = modelsMap.get(key);

    if (!entry) {
      const lang =
        model.sourceLanguage === "en" ? model.targetLanguage : model.sourceLanguage;
      entry = {
        lang,
        display: getDisplayName(lang) ?? lang,
        toEn: [],
        fromEn: [],
      };
      modelsMap.set(key, entry);
    }

    if (model.sourceLanguage === "en") {
      entry.fromEn.push(model);
    } else {
      entry.toEn.push(model);
    }
  }

  const tbody = getElement("tbody");

  /**
   * @param {ModelEntry} entry
   * @returns {string}
   */
  function getModelVersion(entry) {
    if (entry.fromEn.length) {
      return entry.fromEn[0].version;
    }
    if (entry.toEn.length) {
      return entry.toEn[0].version;
    }
    throw new Error("Could not find the model version");
  }

  // Sort the models by language, and then version number.
  const modelEntries = [...modelsMap.values()];
  modelEntries.sort(
    (a, b) => -versionCompare(getModelVersion(a), getModelVersion(b))
  );
  modelEntries.sort((a, b) => a.display.localeCompare(b.display));

  for (const { lang, toEn, fromEn } of modelEntries) {
    const rowCount = Math.max(toEn.length, fromEn.length, 1);

    for (let i = 0; i < rowCount; i++) {
      const tr = document.createElement("tr");
      /**
       * @param {string} [text]
       */
      const td = (text = "") => {
        const el = document.createElement("td");
        el.innerText = text;
        tr.appendChild(el);
        return el;
      };
      td(getDisplayName(lang));

      addToRow(
        td,
        `${lang}-en`,
        records.data,
        googleScores,
        exportsByHash,
        attachmentsByKey,
        toEn[i] ?? null
      );
      addToRow(
        td,
        `en-${lang}`,
        records.data,
        googleScores,
        exportsByHash,
        attachmentsByKey,
        fromEn[i] ?? null
      );
      tbody.append(tr);
    }
  }
  getElement("loading").style.display = "none";
  getElement("table").style.display = "table";
}

/**
 * @param {(text?: string) => HTMLTableCellElement} td
 * @param {string} pair
 * @param {ModelRecord[]} records
 * @param {Record<string, Record<string, number>>} googleScores
 * @param {Record<string, ModelMetadata>} exportsByHash
 * @param {Map<string, Array<[string, string]>>} attachmentsByKey
 * @param {ModelRecord | null} model
 */
function addToRow(
  td,
  pair,
  records,
  googleScores,
  exportsByHash,
  attachmentsByKey,
  model
) {
  if (!model) {
    const classes = [
      "modelColumn",
      "versionColumn",
      "sizeColumn",
      "releaseColumn",
      "scoreColumn",
      "architectureColumn",
      "parametersColumn",
    ];
    for (const className of classes) {
      let el = td();
      el.classList.add("empty", className);
    }
    return;
  }

  const modelNameTD = td();
  modelNameTD.className = "modelColumn";
  /** @type {HTMLDivElement | null} */
  let attachmentsDiv = null;
  const attachments = attachmentsByKey.get(getAttachmentKey(model));
  if (attachments) {
    const div = document.createElement("div");
    div.className = "attachments";
    attachmentsDiv = div;
    for (const [name, url] of attachments) {
      const a = document.createElement("a");
      a.innerText = name;
      a.href = url;
      div.appendChild(a);
    }
    const button = document.createElement("button");
    button.innerText = pair;

    document.body.addEventListener("click", (event) => {
      const target = /** @type {Node | null} */ (event.target);
      if (target && !div.contains(target) && target !== button) {
        div.style.display = "none";
      }
    });

    button.addEventListener("click", () => {
      if (div.style.display === "block") {
        div.style.display = "none";
      } else {
        div.style.display = "block";
      }
    });

    modelNameTD.appendChild(button);
    modelNameTD.appendChild(div);
  } else {
    modelNameTD.innerText = pair;
  }

  const versionEl = td(model.version);
  versionEl.className = "versionColumn";

  const [totalSize, sizeByType] = getModelSize(records, model);
  const sizeElement = td(totalSize);
  sizeElement.className = "sizeColumn";
  sizeElement.title = sizeByType;

  const releaseChannels = getReleaseChannels(model);
  const releaseEl = td(releaseChannels?.label ?? "Custom");
  releaseEl.className = "releaseColumn";
  releaseEl.title = model.filter_expression ?? "";

  const scoreEl = td();
  scoreEl.className = "scoreColumn";
  const architectureEl = td();
  architectureEl.className = "architectureColumn";
  const parametersEl = td();
  parametersEl.className = "parametersColumn";

  const modelMetadata = model.decompressedHash ? exportsByHash[model.decompressedHash] : null;

  if (modelMetadata) {
    architectureEl.innerText = modelMetadata.architecture || "";
    if (modelMetadata.modelStatistics?.parameters) {
      parametersEl.innerText =
        modelMetadata.modelStatistics.parameters.toLocaleString("en-US", {
          maximumFractionDigits: 0,
        });
    }

    if (attachmentsDiv) {
      const metadataPre = document.createElement("pre");
      metadataPre.className = "metadataPre";
      metadataPre.innerText = JSON.stringify(modelMetadata, null, 2);
      attachmentsDiv.appendChild(metadataPre);
    }

    const mozillaComet = modelMetadata.flores?.comet;
    const [src, tgt] = splitLangPair(pair);
    const normalizedPair = `${normalizeLangCode(src)}-${normalizeLangCode(tgt)}`;
    const googleComet = googleScores[normalizedPair]?.comet;

    if (mozillaComet && googleComet) {
      const bergamotCometDisplay = mozillaComet.toFixed(2);
      const percentage = 100 * (1 - googleComet / mozillaComet);
      const sign = percentage >= 0 ? "+" : "";
      const percentDisplay = `${sign}${percentage.toFixed(2)}%`.padStart(
        7,
        "\u00A0"
      );
      const scoreDisplay = `${bergamotCometDisplay}${percentDisplay}`;

      scoreEl.innerText = scoreDisplay;

      let shippable = "Shippable";
      if (percentage < -5) {
        scoreEl.style.background = "#ffa537";
        shippable = "Not shippable";
      }

      scoreEl.title =
        `${shippable} - COMET ${mozillaComet.toFixed(2)} ` +
        `vs Google Comet ${googleComet.toFixed(2)} ` +
        `(${scoreDisplay})` +
        "\n\n";
    }
  }
}

/**
 * @param {ModelRecord[]} records
 * @param {ModelRecord | null} model
 * @returns {[string, string]} [totalSize, sizeByType]
 */
function getModelSize(records, model) {
  if (!model) {
    return ["", ""];
  }

  let sizeByTypes = "";

  let size = 0;
  for (const record of records) {
    if (
      record.sourceLanguage === model.sourceLanguage &&
      record.targetLanguage === model.targetLanguage &&
      record.version === model.version &&
      record.filter_expression === model.filter_expression
    ) {
      const stringSize =
        (Number(record.attachment.size) / 1000 / 1000).toFixed(1) + " MB";
      sizeByTypes += `\n${record.fileType}: ${stringSize}`;
      size += Number(record.attachment.size);
    }
  }

  return [(size / 1000 / 1000).toFixed(1) + " MB", sizeByTypes.trim()];
}

/**
 * Compare two versions quickly.
 * @param {string} a
 * @param {string} b
 * @return {number}
 */
export default function versionCompare(a, b) {
  /** @type {any[]} */
  const aParts = a.split(".");
  /** @type {any[]} */
  const bParts = b.split(".");
  while (aParts.length < 3) {
    aParts.unshift("0");
  }
  while (bParts.length < 3) {
    bParts.unshift("0");
  }

  const [, aEnd, aBeta] = aParts[2].match(/(\d+)([a-z]\d?)?/) ?? [
    undefined,
    "0",
    "",
  ];
  const [, bEnd, bBeta] = bParts[2].match(/(\d+)([a-z]\d?)?/) ?? [
    undefined,
    "0",
    "",
  ];
  aParts.pop();
  bParts.pop();
  aParts.push(aEnd);
  bParts.push(bEnd);

  aParts[0] = Number(aParts[0]);
  aParts[1] = Number(aParts[1]);
  aParts[2] = Number(aParts[2]);

  bParts[0] = Number(bParts[0]);
  bParts[1] = Number(bParts[1]);
  bParts[2] = Number(bParts[2]);

  for (const part of aParts) {
    if (isNaN(part)) {
      console.error(aParts);
      throw new Error(a + " had an NaN.");
    }
  }
  for (const part of bParts) {
    if (isNaN(part)) {
      console.error(bParts);
      throw new Error(a + " had an NaN.");
    }
  }

  for (let i = 0; i < 3; i++) {
    const aPart = aParts[i];
    const bPart = bParts[i];
    if (aPart > bPart) return aGreaterThanB;
    if (aPart < bPart) return aLessThanB;
  }
  if (!aBeta && !bBeta) return aEqualToB;
  if (!aBeta) return aGreaterThanB;
  if (!bBeta) return aLessThanB;

  return aBeta.localeCompare(bBeta);
}

/**
 * @param {ModelRecord | null} model
 * @returns {ReleaseInfo | null}
 */
function getReleaseChannels(model) {
  if (!model) {
    return null;
  }

  switch (model.filter_expression) {
    case "env.channel == 'default' || env.channel == 'nightly' || env.channel == 'beta'":
      return {
        release: false,
        beta: true,
        nightly: true,
        android: true,
        label: "Beta",
      };
    case "env.appinfo.OS != 'Android' || env.channel != 'release'":
    case "env.appinfo.OS != 'Android'":
      return {
        release: true,
        beta: true,
        nightly: true,
        android: false,
        label: "Release (Desktop)",
      };
    case "env.appinfo.OS == 'Android'":
      return {
        release: true,
        beta: true,
        nightly: true,
        android: true,
        label: "Release (Android)",
      };
    case "env.channel == 'default' || env.channel == 'nightly'":
      return {
        release: false,
        beta: false,
        nightly: true,
        android: true,
        label: "Nightly",
      };
    case "":
    case undefined:
      return {
        release: true,
        beta: true,
        nightly: true,
        android: true,
        label: "Release",
      };
    default:
      console.log(
        "Come up with a label for this filter_expression:",
        model.filter_expression
      );
      return null;
  }
}

/**
 * @param {string} a
 * @param {string} b
 * @param {number} direction
 */
function assertComparison(a, b, direction) {
  if (versionCompare(a, b) !== direction) {
    throw new Error(`Expected ${a} ${b} to compare to ${direction}`);
  }
}

assertComparison("1.0a", "1.0", aLessThanB);
assertComparison("1.0a1", "1.0", aLessThanB);
assertComparison("1.0a", "1.0a", aEqualToB);
assertComparison("0.1.0a", "1.0a", aEqualToB);
assertComparison("1.0", "1.0a", aGreaterThanB);
assertComparison("1.0", "1.0a1", aGreaterThanB);
assertComparison("1.0", "2.0", aLessThanB);
assertComparison("1.0", "1.1", aLessThanB);
assertComparison("1.0a", "1.1", aLessThanB);

/**
 * @param {Record<string, Record<string, number>>} googleScores
 * @param {ModelRecord[]} models
 * @param {Record<string, ModelMetadata>} exportsByHash
 */
function logModelTableData(googleScores, models, exportsByHash) {
  /** @type {Array<unknown[]>} */
  const xx_en = [];
  const en_xx = [];

  /**
   * @type {Map<string, ModelRecord | null>}
   */
  const modelsByLangPair = new Map();
  for (const langPair of Object.keys(googleScores)) {
    modelsByLangPair.set(langPair, null);
  }
  for (const model of getNightlyModels(models)) {
    const sourceLanguage = normalizeLangCode(model.sourceLanguage);
    const targetLanguage = normalizeLangCode(model.targetLanguage);
    modelsByLangPair.set(`${sourceLanguage}-${targetLanguage}`, model);
  }

  for (const [langPair, model] of modelsByLangPair) {
    const modelMetadata = model?.decompressedHash ? exportsByHash[model.decompressedHash] : null;
    const [fromLang, toLang] = langPair.split("-");
    const row = [
      langPair,
      fromLang,
      toLang,
      googleScores[langPair]?.comet ?? "",
      modelMetadata?.flores?.comet ?? "",
      getReleaseChannels(model)?.label ?? "",
      modelMetadata?.architecture ?? "",
    ];
    if (fromLang === "en") {
      en_xx.push(row);
    } else {
      xx_en.push(row);
    }
  }

  /**
   * @param {any} a
   * @param {any} b
   */
  function sortRow(a, b) {
    return (a[1] + "-" + a[2]).localeCompare(b[1] + "-" + b[2]);
  }
  xx_en.sort(sortRow);
  en_xx.sort(sortRow);

  const rows = [
    ["Lang Pair", "From", "To", "Google", "Mozilla", "Release", "Architecture"],
    ...en_xx,
    ...xx_en,
  ];

  let tsv = "";
  for (const row of rows) {
    tsv += row.join("\t") + "\n";
  }

  console.log(tsv);
}

/**
 * @param {ModelRecord} record
 */
function getAttachmentKey(record) {
  const { sourceLanguage, targetLanguage, version } = record;
  return `${sourceLanguage}-${targetLanguage} ${version}`;
}

/**
 * @param {ModelRecord[]} records
 */
function getAttachmentsByKey(records) {
  /** @type {Map<string, Array<[string, string]>>} */
  const attachmentsByKey = new Map();
  for (const record of records) {
    const key = getAttachmentKey(record);
    let attachments = attachmentsByKey.get(key);
    if (!attachments) {
      attachments = [];
      attachmentsByKey.set(key, attachments);
    }
    attachments.push([
      record.name,
      `https://firefox-settings-attachments.cdn.mozilla.net/${record.attachment.location}`,
    ]);
  }
  return attachmentsByKey;
}

/**
 * @param {ModelRecord[]} allModels
 * @param {ModelRecord[]} releasedModels
 */
function countModels(allModels, releasedModels) {
  const unique = new Set();
  const fromProd = new Set();
  const fromAll = new Set();
  const toProd = new Set();
  const toAll = new Set();

  for (const model of releasedModels) {
    if (model.sourceLanguage == "en") {
      toProd.add(model.targetLanguage);
    } else {
      fromProd.add(model.sourceLanguage);
    }
  }

  for (const model of allModels) {
    unique.add(model.targetLanguage);
    unique.add(model.sourceLanguage);
    if (model.sourceLanguage == "en") {
      toAll.add(model.targetLanguage);
    } else {
      fromAll.add(model.sourceLanguage);
    }
  }

  const toNightly = toAll.difference(toProd);
  const fromNightly = fromAll.difference(fromProd);

  getElement("fromProd").innerText = String(fromProd.size);
  getElement("toProd").innerText = String(toProd.size);
  getElement("fromNightly").innerText = String(fromNightly.size);
  getElement("toNightly").innerText = String(toNightly.size);
  getElement("uniqueLanguages").innerText = String(unique.size);
}
