// @ts-check
import {
  changeLocation,
  replaceLocation,
  exposeAsGlobal,
  getElement,
} from "../utils.mjs";

/**
 * @import { ModelRecord, EvalResults, ReleaseInfo, ModelMetadata } from "../@types/models"
 */

main().catch((error) => {
  console.error(error);
  getElement("error").style.display = "block";
});

const aLessThanB = "a".localeCompare("b");
const aGreaterThanB = aLessThanB * -1;
const aEqualToB = 0;
const REPO_URL =
  "https://raw.githubusercontent.com/mozilla/firefox-translations-models/main/";
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

function setupReleasedModels() {
  const releasedModelsCheckbox = /** @type {HTMLInputElement} */ (
    getElement("releasedModels")
  );
  const urlParams = new URLSearchParams(window.location.search);
  const urlValue = urlParams.get("releasedModels");
  const isReleasedModels = urlValue === "true" || !urlValue;
  releasedModelsCheckbox.checked = isReleasedModels;
  releasedModelsCheckbox.addEventListener("change", () => {
    const urlParams = new URLSearchParams(window.location.search);
    if (releasedModelsCheckbox.checked) {
      urlParams.delete("releasedModels");
    } else {
      urlParams.set("releasedModels", "false");
    }
    changeLocation(urlParams);
  });

  return isReleasedModels;
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
  const langPairs = new Map();
  for (const model of models) {
    if (getReleaseChannels(model)?.release) {
      const langPair = model.fromLang + "-" + model.toLang;
      const existingModel = langPairs.get(langPair);
      if (
        !existingModel ||
        versionCompare(model.version, existingModel.version) > 0
      ) {
        langPairs.set(langPair, model);
      }
    }
  }
  return (models = [...langPairs.values()]);
}

async function main() {
  getElement("counts").style.display = "table";

  const isPreview = setupRemoteSettingsPreview();
  const bucket = isPreview ? "main-preview" : "main";

  setupShowAdditionalDetails();
  const isReleasedModels = setupReleasedModels();

  /** @type {{ data: ModelRecord[] }} */
  const records = await fetchJSON(
    `https://firefox.settings.services.mozilla.com/v1/buckets/${bucket}/collections/translations-models/records`
  );
  exposeAsGlobal("records", records.data);

  const attachmentsByKey = getAttachmentsByKey(records.data);

  /** @type {EvalResults} */
  const cometResults = await fetchJSON(
    "https://raw.githubusercontent.com/mozilla/firefox-translations-models/main/evaluation/comet-results.json"
  );

  logCometResults(cometResults);

  /** @type {Record<string, string>} */
  const byHash = await fetchJSON(REPO_URL + "models/by-hash.json");
  exposeAsGlobal("byHash", byHash);

  /**
   * @typedef {Object} ModelEntry
   * @property {string} lang
   * @property {string} display
   * @property {ModelRecord | null} fromEn
   * @property {ModelRecord | null} toEn
   */

  /** @type {Map<string, ModelEntry>} */
  const modelsMap = new Map();
  let models = records.data.filter((record) => record.fileType === "model");
  const releasedModels = getReleasedModels(models);

  countModels(models, releasedModels);

  if (isReleasedModels) {
    // Get the released model with the latest version.
    models = releasedModels;
  }
  exposeAsGlobal("models", models);

  const dn = new Intl.DisplayNames("en", {
    type: "language",
    fallback: "code",
    languageDisplay: "standard",
  });

  /**
   * Released models are group by lang
   * @param {string} lang
   * @param {string} version
   */
  function getModelKey(lang, version) {
    if (isReleasedModels) {
      return lang;
    }
    return lang + " " + version;
  }

  for (const model of models) {
    /** @type {ModelEntry | undefined} */
    let entry;
    if (model.fromLang === "en") {
      entry = modelsMap.get(getModelKey(model.toLang, model.version));
      if (!entry) {
        entry = {
          lang: model.toLang,
          display: dn.of(model.toLang) ?? model.toLang,
          toEn: null,
          fromEn: null,
        };
      }
      if (entry.fromEn) {
        const message =
          "Multiple models with the same version were found, this is an error that should be fixed in Remote Settings.";
        alert(message);
        console.error(message, entry.fromEn, model);
      }
      entry.fromEn = model;
    } else {
      entry = modelsMap.get(getModelKey(model.fromLang, model.version));
      if (!entry) {
        entry = {
          lang: model.fromLang,
          display: dn.of(model.fromLang) ?? model.fromLang,
          toEn: null,
          fromEn: null,
        };
      }
      if (entry.toEn) {
        const message =
          "Multiple models with the same version were found, this is an error that should be fixed in Remote Settings.";
        alert(message);
        console.error(message, entry.toEn, model);
      }

      entry.toEn = model;
    }
    modelsMap.set(getModelKey(entry.lang, model.version), entry);
  }

  const tbody = getElement("tbody");

  /**
   * @param {ModelEntry} entry
   * @returns {string}
   */
  function getModelVersion(entry) {
    if (entry.fromEn) {
      return entry.fromEn.version;
    }
    if (entry.toEn) {
      return entry.toEn.version;
    }
    throw new Error("Could not find the model version");
  }

  // Sort the models by language, and then version number.
  const modelEntries = [...modelsMap.values()];
  modelEntries.sort(
    (a, b) => -versionCompare(getModelVersion(a), getModelVersion(b))
  );
  modelEntries.sort((a, b) => a.display.localeCompare(b.display));

  // Only add the score once to a langpair.
  const langPairScoreAdded = new Set();

  for (const { lang, toEn, fromEn } of modelEntries) {
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
    td(dn.of(lang));

    addToRow(
      td,
      `${lang}-en`,
      records.data,
      cometResults,
      byHash,
      attachmentsByKey,
      toEn,
      langPairScoreAdded
    );
    addToRow(
      td,
      `en-${lang}`,
      records.data,
      cometResults,
      byHash,
      attachmentsByKey,
      fromEn,
      langPairScoreAdded
    );
    tbody.append(tr);
  }
  getElement("loading").style.display = "none";
  getElement("table").style.display = "table";
}

/**
 * @param {(text?: string) => HTMLTableCellElement} td
 * @param {string} pair
 * @param {ModelRecord[]} records
 * @param {EvalResults} cometResults
 * @param {Record<string, string>} byHash
 * @param {Map<string, Array<[string, string]>>} attachmentsByKey
 * @param {ModelRecord | null} model
 * @param {Set<string>} langPairScoreAdded
 */
function addToRow(
  td,
  pair,
  records,
  cometResults,
  byHash,
  attachmentsByKey,
  model,
  langPairScoreAdded
) {
  if (!model) {
    // When there is no model add in all of the proper classes.
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
  if (model) {
    // Add the attachments.
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

      // Hide when clicking outside of the button and popup.
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
  }

  const versionEl = td(model.version);
  versionEl.className = "versionColumn";

  const [totalSize, sizeByType] = getModelSize(records, model);
  const sizeElement = td(totalSize);
  sizeElement.className = "sizeColumn";
  sizeElement.title = sizeByType;

  const releaseChannels = getReleaseChannels(model);
  let releaseEl;
  if (model) {
    releaseEl = td(releaseChannels?.label ?? "Custom");
    releaseEl.title = model?.filter_expression ?? "";
  } else {
    releaseEl = td();
  }
  releaseEl.className = "releaseColumn";

  const scoreEl = td();
  scoreEl.className = "scoreColumn";
  const architectureEl = td();
  architectureEl.className = "architectureColumn";
  const parametersEl = td();
  parametersEl.className = "parametersColumn";

  getModelMetadata(byHash, model).then((modelMetadata) => {
    if (!modelMetadata) {
      return;
    }
    architectureEl.innerText = modelMetadata.architecture;
    parametersEl.innerText =
      modelMetadata.modelStatistics.parameters.toLocaleString("en-US", {
        maximumFractionDigits: 0,
      });

    if (attachmentsDiv) {
      const metadataPre = document.createElement("pre");
      metadataPre.className = "metadataPre";
      metadataPre.innerText = JSON.stringify(modelMetadata, null, 2);
      attachmentsDiv.appendChild(metadataPre);
    }

    // Add the evals:
    const mozillaComet = modelMetadata.flores["comet"];
    if (!mozillaComet) {
      return;
    }
    const googleComet = cometResults[pair]?.["flores-test"]?.["google"];

    let hasEvals = Boolean(mozillaComet && googleComet);

    // Only show the evals once for the latest model. We have no way to know which is
    // the correct eval to show.
    if (langPairScoreAdded.has(pair)) {
      hasEvals = false;
    }
    if (hasEvals) {
      langPairScoreAdded.add(pair);
    }

    const bergamotCometDisplay = (100 * mozillaComet).toFixed(2);
    const percentage = 100 * (1 - googleComet / mozillaComet);
    const sign = percentage >= 0 ? "+" : "";
    let scoreDisplay = "";
    if (hasEvals) {
      const percentDisplay = `${sign}${percentage.toFixed(2)}%`.padStart(
        7,
        "\u00A0"
      );
      scoreDisplay = `${bergamotCometDisplay}${percentDisplay}`;
    }

    scoreEl.innerText = scoreDisplay;
    if (hasEvals) {
      let shippable = "Shippable";
      // el.style.color = "#fff";
      // el.style.background = "#2ebffc";
      if (percentage < -5) {
        // Does not meet release criteria.
        scoreEl.style.background = "#ffa537";
        // el.style.color = "#000";
        shippable = "Not shippable";
      }

      scoreEl.title =
        `${shippable} - COMET ${(100 * mozillaComet).toFixed(2)} ` +
        `vs Google Comet ${(100 * googleComet).toFixed(2)} ` +
        `(${scoreDisplay})` +
        "\n\n";
    }
  });
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
      record.fromLang === model.fromLang &&
      record.toLang === model.toLang &&
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
      return {
        release: true,
        beta: true,
        nightly: true,
        android: false,
        label: "Release (Desktop)",
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
 * @param {EvalResults} cometResults
 */
function logCometResults(cometResults) {
  /** @type {Array<unknown[]>} */
  const xx_en = [];
  const en_xx = [];

  for (const [langPair, evaluation] of Object.entries(cometResults)) {
    const flores = evaluation["flores-dev"];
    const [fromLang, toLang] = langPair.split("-");
    const row = [
      langPair,
      fromLang,
      toLang,
      flores.google || "",
      flores.bergamot || "",
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
    ["Lang Pair", "From", "To", "Google", "Bergamot"],
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
  const { fromLang, toLang, version } = record;
  return `${fromLang}-${toLang} ${version}`;
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
    if (model.fromLang == "en") {
      toProd.add(model.toLang);
    } else {
      fromProd.add(model.fromLang);
    }
  }

  for (const model of allModels) {
    unique.add(model.toLang);
    unique.add(model.fromLang);
    if (model.fromLang == "en") {
      toAll.add(model.toLang);
    } else {
      fromAll.add(model.fromLang);
    }
  }

  const toNightly = toAll.difference(toProd);
  const fromNightly = fromAll.difference(fromProd);

  getElement("fromProd").innerText = String(fromProd.size);
  getElement("toProd").innerText = String(toProd.size);
  getElement("fromNightly").innerText = String(toNightly.size);
  getElement("toNightly").innerText = String(fromNightly.size);
  getElement("uniqueLanguages").innerText = String(unique.size);
}

/**
 * @param {Record<string, string>} byHash
 * @param {ModelRecord | null} model
 * @return {Promise<ModelMetadata | null>}
 */
async function getModelMetadata(byHash, model) {
  if (!model) {
    return null;
  }
  const metadataUrl = byHash[model.attachment.hash];
  if (!metadataUrl) {
    return null;
  }

  const response = await fetch(REPO_URL + metadataUrl);
  return response.json();
}
